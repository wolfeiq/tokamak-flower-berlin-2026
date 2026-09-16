"""Scalar parameter recovery by differentiating through TORAX.

UNVERIFIED AGAINST A LIVE TORAX -- like `envs/torax_env.py`, written from the
sources. `scripts/gate_identify.py` exists to find out what is wrong.

WHY NOT A PINN FOR THESE
------------------------
TORAX is differentiable, and these unknowns are traced leaves of its config:

    plasma_composition.Z_eff
    sources.ei_exchange.Qei_multiplier
    neoclassical.bootstrap_current.bootstrap_multiplier
    numerics.resistivity_multiplier
    transport.chi_i / chi_e / D_e        (under the `prescribed` model)

So the estimate can be made against the TRUE forward model:

    theta_hat = argmin || observed - TORAX(theta) ||^2       via jax.grad

A PINN would replace that forward model with a network that approximately
satisfies the residual, and inherit the approximation error for no benefit.
Use the simulator where the simulator is exact; use the PINN (identification/
pinn.py) for the field-valued closure, which the simulator cannot give you
because it bakes its closure in.

GRADIENT FLOW IS CHECKED, NOT ASSUMED
-------------------------------------
A zero gradient w.r.t. a parameter means either it does not affect the
observations (unidentifiable) or it is not being traced (a plumbing bug). Those
need different fixes and look identical from the outside, so
`check_gradient_flow` runs before any optimisation and reports per parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Config paths that are plausible per-device unknowns AND traced JAX leaves.
# Verified against TORAX 1.4.3 sources; `gate_identify.py` re-checks against the
# installed version rather than trusting this list.
CANDIDATE_UNKNOWNS: dict[str, tuple[float, float, float]] = {
    # path: (nominal, lo, hi)
    "plasma_composition.Z_eff": (1.6, 1.0, 4.0),
    "sources.ei_exchange.Qei_multiplier": (1.0, 0.3, 3.0),
    "neoclassical.bootstrap_current.bootstrap_multiplier": (1.0, 0.5, 1.5),
    "numerics.resistivity_multiplier": (1.0, 0.5, 2.0),
}


@dataclass
class RecoveryResult:
    names: list[str]
    truth: np.ndarray
    estimate: np.ndarray
    history: list[dict] = field(default_factory=list)
    gradient_flow: dict[str, float] = field(default_factory=dict)

    @property
    def relative_error(self) -> np.ndarray:
        t = np.abs(self.truth)
        out = np.full_like(self.estimate, np.inf)
        ok = t > 1e-30
        out[ok] = np.abs(self.estimate[ok] - self.truth[ok]) / t[ok]
        return out

    def summary(self, tolerance: float = 0.05) -> str:
        lines = [f"{'parameter':52s}{'truth':>10s}{'recovered':>12s}{'rel err':>10s}"]
        for k, n in enumerate(self.names):
            lines.append(f"{n:52s}{self.truth[k]:>10.4g}{self.estimate[k]:>12.4g}"
                         f"{self.relative_error[k]:>9.2%}")
        worst = float(np.max(self.relative_error))
        lines += ["", f"worst relative error: {worst:.2%} "
                      f"({'PASS' if worst < tolerance else 'FAIL'} at {tolerance:.0%})"]
        dead = [n for n, g in self.gradient_flow.items() if abs(g) < 1e-12]
        if dead:
            lines.append(f"  ZERO GRADIENT for {dead} -- these are either "
                         "unidentifiable from this observation set or not being "
                         "traced. Check CRLB before blaming the optimiser.")
        return "\n".join(lines)


def check_gradient_flow(loss_and_grad: Callable, theta: np.ndarray,
                        names: list[str]) -> dict[str, float]:
    """Per-parameter |dL/dtheta| at the starting point.

    Run this BEFORE optimising. A zero entry is the single most informative
    diagnostic available: it separates "the physics does not depend on this"
    from "the optimiser is struggling", which otherwise look the same.
    """
    _, g = loss_and_grad(theta)
    g = np.asarray(g, dtype=float).ravel()
    return {n: float(abs(g[k])) for k, n in enumerate(names)}


def adam(loss_and_grad: Callable, theta0: np.ndarray, *, iterations: int = 200,
         lr: float = 0.05, bounds: tuple[np.ndarray, np.ndarray] | None = None,
         verbose: bool = True, log_every: int = 10) -> tuple[np.ndarray, list[dict]]:
    """Plain Adam with box projection. No optimiser dependency.

    Bounds are projected rather than penalised: these are physical quantities
    with known ranges (Z_eff >= 1 by definition), and letting the optimiser
    wander outside them produces simulator failures that look like fit failures.
    """
    theta = np.asarray(theta0, dtype=float).ravel().copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[dict] = []

    for it in range(iterations):
        loss, g = loss_and_grad(theta)
        g = np.asarray(g, dtype=float).ravel()
        if not np.all(np.isfinite(g)):
            history.append({"iteration": it, "loss": float(loss), "note": "non-finite gradient"})
            break
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        theta = theta - lr * (m / (1 - b1 ** (it + 1))) / (
            np.sqrt(v / (1 - b2 ** (it + 1))) + eps
        )
        if bounds is not None:
            theta = np.clip(theta, bounds[0], bounds[1])
        if it % log_every == 0 or it == iterations - 1:
            history.append({"iteration": it, "loss": float(loss),
                            "theta": theta.copy().tolist()})
            if verbose:
                print(f"    it {it:4d}  loss {float(loss):.6e}  "
                      f"theta {np.array2string(theta, precision=4)}")
    return theta, history


def _updater_for_paths(provider, unknown_paths: list[str]):
    """One value-wrapper per unknown, chosen by what the config leaf actually is.

    THE BUG THIS FIXES. Every unknown was handed to
    ``update_provider_from_mapping`` as a bare float. TORAX accepts that only
    for leaves that are plain scalars; a ``TimeVaryingScalar`` needs a
    ``TimeVaryingScalarUpdate`` and a ``TimeVaryingArray`` a
    ``TimeVaryingArrayUpdate``, and it raises rather than coercing:

        ValueError: To replace a `TimeVaryingArray` use a
        `TimeVaryingArrayUpdate`, got <class 'numpy.float64'> instead.

    ``plasma_composition.Z_eff`` is a ``TimeVaryingArray`` (Z_eff may vary in
    both rho and time), so the very first candidate unknown tripped this and
    nothing in the inverse path could run at all.

    The wrapper is resolved ONCE, outside the traced function, by asking the
    provider what sits at each path -- guessing per-parameter would be the same
    mistake one level up. A scalar unknown broadcast into a
    ``TimeVaryingArray`` becomes a radially flat, time-constant profile, which
    is exactly what a scalar multiplier means.
    """
    from torax._src.torax_pydantic import interpolated_param_1d as _p1
    from torax._src.torax_pydantic import interpolated_param_2d as _p2

    import jax.numpy as jnp

    wrap: dict[str, Any] = {}
    for path in unknown_paths:
        leaf = provider.get_node_from_path(path)
        if isinstance(leaf, _p1.TimeVaryingScalar):
            def make(v, _u=_p1.TimeVaryingScalarUpdate):
                return _u(time=jnp.asarray([0.0]), value=jnp.atleast_1d(v))
        elif isinstance(leaf, _p2.TimeVaryingArray):
            def make(v, _u=_p2.TimeVaryingArrayUpdate):
                # `value` is (t, rhon); `rho_norm` is (rhon,) -- NOT (1, rhon).
                # Two rho points at the ends make the profile radially flat,
                # one time point makes it time-constant, which is what a
                # scalar multiplier means.
                return _u(
                    time=jnp.asarray([0.0]),
                    rho_norm=jnp.asarray([0.0, 1.0]),
                    value=jnp.atleast_1d(v)[..., None] * jnp.ones((1, 2)),
                )
        else:
            # Plain scalar/Array leaf: TORAX requires an exact shape and dtype
            # match, so mirror the leaf rather than passing a Python float.
            ref = jnp.asarray(leaf)

            def make(v, _ref=ref):
                return jnp.asarray(v, dtype=_ref.dtype).reshape(_ref.shape)

        wrap[path] = make
    return wrap


def build_loss(
    config_dict: dict[str, Any],
    unknown_paths: list[str],
    observations: dict[str, np.ndarray],
    delta_t: float,
    n_steps: int,
    weights: dict[str, float] | None = None,
    rho_indices: np.ndarray | None = None,
):
    """Build a differentiable ``theta -> (loss, dloss/dtheta)``.

    Runs TORAX forward with the candidate parameters patched in, compares the
    resulting profiles against `observations`, and differentiates the whole
    thing with `jax.grad`.

    Observations are keyed by core-profile name (``T_e``, ``T_i``, ``n_e``) and
    shaped (n_steps, n_rho). Each is normalised by its own scale before entering
    the loss, so a temperature in keV and a density in 1e20 m^-3 contribute
    comparably instead of the larger number deciding the fit.

    ``rho_indices`` gives the radial cells the observations were measured at,
    for sparse diagnostics. Without it the loss subtracted a (n_steps, 25)
    prediction from a (n_steps, 7) observation and raised

        TypeError: sub got incompatible shapes for broadcasting: (8, 25), (8, 7)

    so ``exp_recover_scalars.py --subsample`` -- the run RUNBOOK 5b calls the
    one that keeps every downstream claim honest -- could never execute. The
    slice belongs here rather than at the call site because the prediction is
    what has to be reduced, and only this function holds it.
    """
    import jax
    import jax.numpy as jnp
    from torax.experimental import (
        RuntimeParamsProvider,
        SimulationStepFn,
        get_initial_state_and_post_processed_outputs,
        make_step_fn,
    )
    import torax

    # ADAPTIVE dt MUST BE OFF ON THE DIFFERENTIABLE PATH.
    # With TORAX's default `numerics.adaptive_dt = True`, one solver step
    # branches on `solver_error_state` and retries at a halved dt. The primal
    # is fine, but the tangent of that branch is NaN, and the NaN propagates
    # into every parameter -- so `check_gradient_flow` reported all four
    # candidates DEAD and the CRLB screen called a perfectly identifiable set
    # unidentifiable. Measured on TORAX 1.4.3: the same one-step JVP gives
    # `nan` with adaptive_dt on and 0.765 with it off.
    #
    # `adaptive_dt` is JAX_STATIC, so this is a compile-time choice and costs
    # nothing at run time. It is set here rather than left to the caller
    # because a silent NaN gradient is indistinguishable from an
    # unidentifiable parameter, which is precisely the confusion
    # `gate_identify.py` exists to resolve. The trade is real and worth
    # stating: without adaptive retry a step that would have been rescued by
    # halving dt now simply fails, so use a `fixed_dt` the forward solve is
    # comfortable with.
    config_dict = dict(config_dict)
    config_dict["numerics"] = {**config_dict.get("numerics", {}), "adaptive_dt": False}

    base_cfg = torax.ToraxConfig.from_dict(config_dict)
    base_step_fn = make_step_fn(base_cfg)
    wrap = _updater_for_paths(base_step_fn.runtime_params_provider, unknown_paths)
    weights = weights or {k: 1.0 for k in observations}
    scales = {k: float(np.std(np.asarray(v)) + 1e-30) for k, v in observations.items()}
    obs_j = {k: jnp.asarray(v) for k, v in observations.items()}

    def simulate(theta):
        provider = base_step_fn.runtime_params_provider.update_provider_from_mapping(
            {p: wrap[p](theta[i]) for i, p in enumerate(unknown_paths)}
        )
        step_fn = SimulationStepFn(
            solver=base_step_fn.solver,
            time_step_calculator=base_step_fn.time_step_calculator,
            runtime_params_provider=provider,
            geometry_provider=base_step_fn.geometry_provider,
        )
        state, post = get_initial_state_and_post_processed_outputs(
            t=float(base_cfg.numerics.t_initial), step_fn=step_fn
        )
        traces = {k: [] for k in observations}
        for _ in range(n_steps):
            state, post = step_fn.jitted_fixed_time_step(delta_t, state, post)
            for k in observations:
                var = getattr(state.core_profiles, k)
                traces[k].append(getattr(var, "value", var))
        return {k: jnp.stack(v) for k, v in traces.items()}

    idx = None if rho_indices is None else jnp.asarray(np.asarray(rho_indices, int))

    def loss(theta):
        pred = simulate(theta)
        total = 0.0
        for k, ref in obs_j.items():
            p = pred[k] if idx is None else pred[k][:, idx]
            if p.shape != ref.shape:
                raise ValueError(
                    f"{k}: prediction {p.shape} does not match observation "
                    f"{ref.shape}. Pass `rho_indices` naming the radial cells "
                    "the observations were taken at."
                )
            total = total + weights[k] * jnp.mean(((p - ref) / scales[k]) ** 2)
        return total

    # FORWARD-mode, not reverse. `jax.value_and_grad` is reverse-mode, and
    # TORAX's `jitted_fixed_time_step` consumes the action window inside a
    # `jax.lax.while_loop` (step_function.py:336) whose trip count is dynamic.
    # JAX cannot reverse-differentiate that at all:
    #
    #     ValueError: Reverse-mode differentiation does not work for
    #     lax.while_loop or lax.fori_loop with dynamic start/stop values.
    #
    # so the whole inverse path raised before producing a single gradient.
    # Forward-mode (JVP) works through `while_loop` unchanged, and here it is
    # also the CHEAPER direction: its cost scales with the number of INPUTS,
    # and `CANDIDATE_UNKNOWNS` holds four scalars against a scalar loss. Reverse
    # mode would only win once the unknowns run to the hundreds -- at which
    # point the fix is a scan-based stepper, not a different `grad` call.
    loss_jit = jax.jit(loss)
    grad_fwd = jax.jit(jax.jacfwd(loss))

    def value_and_grad(theta):
        return loss_jit(theta), grad_fwd(theta)

    return value_and_grad, simulate
