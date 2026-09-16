"""Identifiability screen: which parameters are worth trying to identify?

WHY THIS RUNS FIRST
-------------------
Several of the physically interesting unknowns are degenerate with each other,
and fitting a degenerate set produces confident nonsense rather than an error:

  * confinement-deviation-from-scaling and a transport multiplier are close to
    the same parameter -- tau_E deviation IS an integrated measure of chi;
  * wall recycling may be structurally invisible in a 1-D core model with a
    prescribed edge density boundary condition, because recycling is absorbed
    into that boundary condition;
  * Z_eff enters resistivity, radiation and dilution, so it is only partially
    separable from each.

The Cramer-Rao lower bound turns "is this identifiable?" into a number: given
the sensitivity of the observations to each parameter and the noise level, no
unbiased estimator can do better than the CRLB. If the bound on a parameter is
already larger than the effect you care about, no amount of training will fix
it and the honest move is to drop the parameter or change the experiment.

JACOBIAN BY FINITE DIFFERENCES, DELIBERATELY
--------------------------------------------
The natural route is `jax.jacfwd` through the simulator. TORAX issue #2331
reports that `jacfwd` and `jacrev(grad(...))` **silently return NaN** -- and a
silent NaN in an identifiability screen is worse than no screen. Finite
differences cost n_params+1 forward solves, which is nothing at this scale, and
cannot fail quietly.

Pure NumPy: the bound is computed from a Jacobian the caller supplies, so this
is testable without TORAX and reusable for any forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IdentifiabilityReport:
    names: list[str]
    crlb_std: np.ndarray  # lower bound on the std of each estimate
    true_values: np.ndarray
    correlation: np.ndarray  # parameter correlation matrix
    condition_number: float
    rank_deficient: bool

    @property
    def relative_crlb(self) -> np.ndarray:
        """CRLB as a fraction of each parameter's value -- the usable number."""
        v = np.abs(self.true_values)
        out = np.full_like(self.crlb_std, np.inf)
        ok = v > 1e-30
        out[ok] = self.crlb_std[ok] / v[ok]
        return out

    def worst_pair(self) -> tuple[str, str, float]:
        """The most collinear pair -- the one most likely to be unidentifiable."""
        c = np.abs(self.correlation.copy())
        np.fill_diagonal(c, 0.0)
        i, j = np.unravel_index(np.argmax(c), c.shape)
        return self.names[i], self.names[j], float(c[i, j])

    def summary(self, tolerance: float = 0.1) -> str:
        lines = [f"{'parameter':28s}{'true':>12s}{'CRLB std':>12s}{'rel':>10s}  verdict"]
        for k, name in enumerate(self.names):
            rel = self.relative_crlb[k]
            verdict = ("identifiable" if rel < tolerance
                       else "marginal" if rel < 3 * tolerance
                       else "NOT identifiable")
            rel_s = f"{rel:.1%}" if np.isfinite(rel) else "inf"
            lines.append(f"{name:28s}{self.true_values[k]:>12.4g}"
                         f"{self.crlb_std[k]:>12.4g}{rel_s:>10s}  {verdict}")
        a, b, c = self.worst_pair()
        lines += ["", f"most collinear pair: {a} / {b}  (|r| = {c:.3f})"]
        if c > 0.95:
            lines.append("  WARNING: these two are nearly the same parameter. Fitting "
                         "both produces a confident answer along an arbitrary point "
                         "of the degenerate direction. Drop one, or design an "
                         "experiment that separates them.")
        lines.append(f"Fisher condition number: {self.condition_number:.3e}")
        if self.rank_deficient:
            lines.append("  WARNING: the Fisher matrix is rank-deficient -- at least "
                         "one direction in parameter space leaves the observations "
                         "unchanged. That parameter combination cannot be identified "
                         "from this data at any noise level.")
        return "\n".join(lines)


def fisher_from_jacobian(J: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    """Fisher information for Gaussian noise: F = J^T Sigma^-1 J."""
    J = np.atleast_2d(np.asarray(J, dtype=float))
    if np.isscalar(sigma):
        return (J.T @ J) / float(sigma) ** 2
    s = np.asarray(sigma, dtype=float).ravel()
    if s.size != J.shape[0]:
        raise ValueError("sigma must be scalar or one value per observation")
    return J.T @ (J / (s**2)[:, None])


def crlb(
    J: np.ndarray,
    sigma: float | np.ndarray,
    names: list[str],
    true_values: np.ndarray,
    rcond: float = 1e-12,
) -> IdentifiabilityReport:
    """Cramer-Rao bound from a sensitivity Jacobian.

    Args:
        J: (n_obs, n_params) d(observation)/d(parameter).
        sigma: observation noise std, scalar or per-observation.
        names, true_values: for reporting.

    A pseudo-inverse is used so a singular Fisher matrix reports as
    rank-deficient rather than raising -- an exactly unidentifiable direction
    is a finding, not an error.
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    names = list(names)
    true_values = np.asarray(true_values, dtype=float).ravel()
    if J.shape[1] != len(names) or len(names) != true_values.size:
        raise ValueError("J columns, names and true_values must agree")

    F = fisher_from_jacobian(J, sigma)
    eig = np.linalg.eigvalsh(F)
    eig_max = float(np.max(np.abs(eig))) if eig.size else 0.0
    rank_deficient = bool(eig_max <= 0 or np.min(eig) <= rcond * eig_max)
    cond = float(eig_max / max(np.min(np.abs(eig)), 1e-300)) if eig.size else np.inf

    cov = np.linalg.pinv(F, rcond=rcond)
    var = np.clip(np.diag(cov), 0.0, None)
    std = np.sqrt(var)

    d = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(d > 0, cov / d, 0.0)
    np.fill_diagonal(corr, 1.0)

    return IdentifiabilityReport(
        names=names, crlb_std=std, true_values=true_values,
        correlation=corr, condition_number=cond, rank_deficient=rank_deficient,
    )


def jacobian_by_finite_differences(
    forward, theta: np.ndarray, rel_step: float = 1e-3, abs_step: float = 1e-8
) -> np.ndarray:
    """d(observation)/d(parameter) by central differences.

    `forward(theta) -> flat observation vector`. Costs 2*n_params forward
    solves. See the module docstring for why this is preferred to `jax.jacfwd`
    here (TORAX #2331: silent NaN).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    base = np.asarray(forward(theta), dtype=float).ravel()
    J = np.zeros((base.size, theta.size))
    for k in range(theta.size):
        h = max(abs(theta[k]) * rel_step, abs_step)
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        J[:, k] = (np.asarray(forward(up), float).ravel()
                   - np.asarray(forward(dn), float).ravel()) / (2 * h)
    return J
