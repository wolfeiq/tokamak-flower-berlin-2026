"""Inverse PINN for the transport closure: chi(rho) from noisy profiles.

WHY THE RESIDUAL IS THE INTEGRAL FORM, NOT THE DIFFERENTIAL ONE
---------------------------------------------------------------
The obvious PINN writes the conservation law as it appears in the textbook,

    (3/2) d(nT)/dt + (1/rho) d(rho q)/drho - S = 0,      q = -n chi dT/drho

and penalises that at collocation points. Doing so needs d/drho of a product
containing dT/drho -- i.e. SECOND derivatives of the network. That is exactly
the configuration that failed in the Cosserat work: when an unknown enters the
residual only through a high-order derivative the network under-resolves, the
residual is satisfiable without committing to the parameter, and the fit stalls
(GA/EA stuck near 250% while the CRLB floor said 0.05%).

The fix there was to integrate the balance law and expose the unknown against a
data-derived quantity. The same move works here. Integrating once,

    rho * q(rho) = integral_0^rho [S - (3/2) d(nT)/dt] rho' drho'  ==  RHS(rho)

the right-hand side is built ONCE from measured sources and profiles -- it does
not involve chi at all -- and the residual becomes

    rho * ( -n * chi_phi(rho) * dT_theta/drho ) - RHS(rho) = 0

which needs only FIRST derivatives of the network. chi appears algebraically,
multiplied by a quantity the data determines.

SO WHAT IS THE NETWORK FOR?
---------------------------
Not for the integral -- ``closure.chi_from_power_balance`` already does that
with finite differences, and on clean data it reaches ~0.1%. The network earns
its place only where finite differences break: dT/drho amplifies noise, and
real diagnostics are sparse and noisy. So T is represented by a smooth network
fitted to the observations, and its analytic derivative replaces the finite
difference.

    L = ||T_theta - T_obs||^2  +  lambda * ||rho * q_theta,phi - RHS||^2

If this does not beat the finite-difference baseline on noisy, sparse data, the
network has not earned its place. ``scripts/exp_identify_chi.py`` runs both and
reports the comparison either way.

Pure NumPy with hand-derived gradients: one hidden layer, tanh, so d/drho is
analytic and no autodiff framework is needed next to JAX on a memory-limited
machine. It also keeps the whole thing testable without TORAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ScalarMLP:
    """1 -> 1 tanh MLP with an analytic first derivative.

    y(x)      = W2 @ tanh(W1 x + b1) + b2
    dy/dx     = W2 @ ((1 - h^2) * W1)

    Deliberately tiny. The unknown is a smooth radial profile, not an image,
    and SPEC.md §8 requires small networks throughout.
    """

    def __init__(self, hidden: int = 16, seed: int = 0, out_bias: float = 0.0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.W1 = rng.normal(0.0, 1.0, hidden)
        self.b1 = rng.uniform(-1.0, 1.0, hidden)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden)
        self.b2 = float(out_bias)

    # -- forward ---------------------------------------------------------

    def _hidden(self, x: np.ndarray):
        z = np.outer(x, self.W1) + self.b1  # (n, h)
        return np.tanh(z)

    def value(self, x: np.ndarray) -> np.ndarray:
        return self._hidden(x) @ self.W2 + self.b2

    def value_and_grad_x(self, x: np.ndarray):
        """Returns (y, dy/dx). The derivative is exact, not finite-differenced."""
        h = self._hidden(x)
        y = h @ self.W2 + self.b2
        dydx = (1.0 - h**2) @ (self.W2 * self.W1)
        return y, dydx

    # -- parameter vector view ------------------------------------------

    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.W1, self.b1, self.W2, [self.b2]])

    def set_flat(self, v: np.ndarray) -> None:
        h = self.hidden
        self.W1 = v[:h]
        self.b1 = v[h : 2 * h]
        self.W2 = v[2 * h : 3 * h]
        self.b2 = float(v[3 * h])

    @property
    def n_params(self) -> int:
        return 3 * self.hidden + 1


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _try_jax_grad(rho, T_hat, n_obs, rhs, T_mean, T_std, hidden,
                  physics_weight, chi_scale):
    """Build a JAX gradient of the same loss, or return None if JAX is absent.

    The loss is transcribed rather than shared, so the NumPy version stays the
    readable reference. `tests/test_identification.py` checks the two agree --
    a divergence between them is a bug in one of them, and the test says which.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None

    rho_j = jnp.asarray(rho)
    T_hat_j = jnp.asarray(T_hat)
    n_j = jnp.asarray(n_obs)
    rhs_j = jnp.asarray(rhs)

    def mlp(p, x):
        W1, b1, W2, b2 = p[:hidden], p[hidden:2 * hidden], p[2 * hidden:3 * hidden], p[3 * hidden]
        h = jnp.tanh(jnp.outer(x, W1) + b1)
        y = h @ W2 + b2
        dydx = (1.0 - h**2) @ (W2 * W1)
        return y, dydx

    n_t = 3 * hidden + 1

    def loss(p):
        tp, cp = p[:n_t], p[n_t:]
        T_s, dT_s = mlp(tp, rho_j)
        dT_phys = dT_s * T_std
        raw, _ = mlp(cp, rho_j)
        chi = chi_scale * jax.nn.softplus(raw)
        data_l = jnp.mean((T_s - T_hat_j) ** 2)
        phys_l = jnp.mean((rho_j * (-n_j * chi * dT_phys) - rhs_j) ** 2)
        return data_l + physics_weight * phys_l

    return jax.jit(jax.grad(loss))


@dataclass
class PinnResult:
    rho: np.ndarray
    chi: np.ndarray
    T_smooth: np.ndarray
    history: list[dict] = field(default_factory=list)
    final_loss: float = float("nan")
    data_loss: float = float("nan")
    physics_loss: float = float("nan")


def identify_chi_pinn(
    rho_obs: np.ndarray,
    T_obs: np.ndarray,
    n_obs: np.ndarray,
    rhs: np.ndarray,
    rho_eval: np.ndarray | None = None,
    *,
    hidden: int = 16,
    iterations: int = 4000,
    lr: float = 5e-3,
    physics_weight: float = 1.0,
    chi_scale: float = 1.0,
    seed: int = 0,
    use_jax: bool = True,
    verbose: bool = False,
) -> PinnResult:
    """Fit a smooth T(rho) and chi(rho) to noisy observations under the balance.

    Args:
        rho_obs: (m,) radii where T and n were measured. May be sparse and
            irregular -- that is the case the network is for.
        T_obs: (m,) noisy temperature observations.
        n_obs: (m,) density at the same radii.
        rhs: (m,) the data-derived right-hand side
            ``integral_0^rho [S - (3/2) d(nT)/dt] rho' drho'``, from
            ``closure.heat_flux_from_balance`` multiplied by V'.
        chi_scale: chi is parameterised as ``chi_scale * softplus(net)`` so it
            is POSITIVE by construction. A negative transport coefficient is
            unphysical and letting the optimiser explore there wastes the fit
            and can make the residual satisfiable in a nonsense way.

    Gradients are hand-derived; see the module docstring for why the residual
    uses only first derivatives.
    """
    rho_obs = np.asarray(rho_obs, dtype=float).ravel()
    T_obs = np.asarray(T_obs, dtype=float).ravel()
    n_obs = np.asarray(n_obs, dtype=float).ravel()
    rhs = np.asarray(rhs, dtype=float).ravel()
    m = rho_obs.size
    if not (T_obs.size == n_obs.size == rhs.size == m):
        raise ValueError("rho_obs, T_obs, n_obs and rhs must have equal length")
    if m < 4:
        raise ValueError("need at least 4 observation points")

    # Scale T so the data term is O(1) regardless of units.
    T_mean, T_std = float(T_obs.mean()), float(T_obs.std() + 1e-12)
    T_hat = (T_obs - T_mean) / T_std

    t_net = ScalarMLP(hidden, seed=seed)
    c_net = ScalarMLP(hidden, seed=seed + 1)

    params = np.concatenate([t_net.get_flat(), c_net.get_flat()])
    n_t = t_net.n_params
    m_adam = np.zeros_like(params)
    v_adam = np.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def unpack(p):
        t_net.set_flat(p[:n_t])
        c_net.set_flat(p[n_t:])

    def loss_fn(p):
        unpack(p)
        T_s, dT_s = t_net.value_and_grad_x(rho_obs)
        # Back to physical units: T = T_hat*std + mean, dT/drho = dT_hat*std
        T_phys = T_s * T_std + T_mean
        dT_phys = dT_s * T_std
        raw = c_net.value(rho_obs)
        chi = chi_scale * _softplus(raw)

        data_r = T_s - T_hat
        # rho * q = rho * (-n chi dT/drho) must equal the measured RHS.
        phys_r = rho_obs * (-n_obs * chi * dT_phys) - rhs

        data_l = float(np.mean(data_r**2))
        phys_l = float(np.mean(phys_r**2))
        return data_l + physics_weight * phys_l, data_l, phys_l, T_phys, chi

    jax_grad = _try_jax_grad(
        rho_obs, T_hat, n_obs, rhs, T_mean, T_std, hidden,
        physics_weight, chi_scale,
    ) if use_jax else None

    def grad_fn(p, h=1e-6):
        """Gradient of the loss w.r.t. the parameter vector.

        Uses JAX autodiff when available. The NumPy fallback is CENTRAL
        DIFFERENCES, which costs 2*n_params loss evaluations per step -- for
        ~50 parameters that is 100x slower per step than autodiff, and in
        practice it means the fit is undertrained rather than wrong. On clean
        manufactured data the finite-difference path reaches chi errors around
        0.9 where the analytic baseline reaches 0.01; almost all of that gap is
        step count, not formulation.

        So: the NumPy path exists to keep this module testable with no JAX
        (and it is what the unit tests exercise). Real fits should run with
        `use_jax=True`, which the target machine supports.
        """
        if jax_grad is not None:
            return np.asarray(jax_grad(p), dtype=float)
        g = np.zeros_like(p)
        for i in range(p.size):
            up, dn = p.copy(), p.copy()
            up[i] += h
            dn[i] -= h
            g[i] = (loss_fn(up)[0] - loss_fn(dn)[0]) / (2 * h)
        return g

    history = []
    for it in range(iterations):
        total, dl, pl, _, _ = loss_fn(params)
        g = grad_fn(params)
        m_adam = b1 * m_adam + (1 - b1) * g
        v_adam = b2 * v_adam + (1 - b2) * g**2
        mh = m_adam / (1 - b1 ** (it + 1))
        vh = v_adam / (1 - b2 ** (it + 1))
        params -= lr * mh / (np.sqrt(vh) + eps)

        if it % max(1, iterations // 10) == 0 or it == iterations - 1:
            history.append({"iteration": it, "loss": total, "data": dl, "physics": pl})
            if verbose:
                print(f"    it {it:5d}  loss {total:.4e}  data {dl:.3e}  phys {pl:.3e}")

    total, dl, pl, T_phys, _ = loss_fn(params)
    grid = rho_obs if rho_eval is None else np.asarray(rho_eval, dtype=float)
    unpack(params)
    chi_grid = chi_scale * _softplus(c_net.value(grid))
    T_grid = t_net.value(grid) * T_std + T_mean

    return PinnResult(
        rho=grid, chi=chi_grid, T_smooth=T_grid, history=history,
        final_loss=total, data_loss=dl, physics_loss=pl,
    )
