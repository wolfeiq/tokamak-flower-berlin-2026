"""Closure-free transport identification.

THE IDEA
--------
The transport equations split into two parts with very different epistemic
status:

    conservation   (3/2) d(n T)/dt + (1/rho) d(rho q)/drho = S      CERTAIN
    closure        q = -n chi dT/drho                               A MODEL

Fitting TORAX's own parameters assumes BOTH are right and only the parameters
are unknown. Here we impose conservation exactly and treat ``chi(rho)`` as a
free field, identified from data. That is something inverting the simulator
structurally cannot do, because TORAX bakes its closure in -- and it is what
experimental transport physicists do by hand ("power balance analysis").

The source S is KNOWN: it is the heating the controller commanded. That is
what makes the problem well-posed, and it is the same structural trick that
rescued the Cosserat stiffness identification -- expose the unknown against a
known or data-derived quantity via the integrated balance law, instead of
fighting under-resolved collocation derivatives.

TWO ESTIMATORS, DELIBERATELY
----------------------------
``chi_from_power_balance``  the classical integral inversion. No network, no
                            training. Exact in principle, noise-sensitive.
                            This is the BASELINE the PINN has to beat.
``identification.pinn``     a smooth chi(rho) fitted to the same balance.
                            Should win on sparse/noisy data and lose nothing
                            on clean data.

If the PINN does not beat the baseline, it has not earned its place. Report
that rather than tuning until it does.

GEOMETRY
--------
Cylindrical (V' proportional to rho), which is consistent with the circular
TORAX geometry every device in this repo uses. For a real equilibrium the
volume element changes and `volume_element` must be supplied.

Pure NumPy. No TORAX, no JAX -- so the identification maths is verified here
against a manufactured solution before it ever meets simulator output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _as_2d(a: np.ndarray, n_t: int, n_rho: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        if a.shape[0] != n_rho:
            raise ValueError(f"{name}: expected {n_rho} radial points, got {a.shape}")
        return np.broadcast_to(a, (n_t, n_rho)).copy()
    if a.shape != (n_t, n_rho):
        raise ValueError(f"{name}: expected {(n_t, n_rho)}, got {a.shape}")
    return a


def heat_flux_from_balance(
    rho: np.ndarray,
    n_e: np.ndarray,
    T_e: np.ndarray,
    source: np.ndarray,
    times: np.ndarray | None = None,
    volume_element: np.ndarray | None = None,
) -> np.ndarray:
    """Heat flux q(rho) from the INTEGRATED conservation law.

        rho * q(rho) = integral_0^rho [ S - (3/2) d(n T)/dt ] rho' drho'

    Integrating rather than differentiating is the whole point: the flux is
    recovered from an integral of measured quantities, which smooths noise
    instead of amplifying it, and q never touches the unknown chi. The closure
    is only invoked afterwards, in ``chi_from_flux``.

    Args:
        rho: (n_rho,) normalised radius, ascending, starting at 0.
        n_e, T_e: (n_rho,) or (n_t, n_rho). Units are the caller's; chi comes
            back in [T_e]*[rho]^2 / ([T_e]*[rho]) consistent units.
        source: volumetric heating, same shape convention.
        times: (n_t,) needed only when the inputs are time-dependent.
        volume_element: V'(rho); defaults to rho (cylindrical).

    Returns:
        q with the same (n_t, n_rho) shape. q[:, 0] = 0 by construction --
        no flux crosses the magnetic axis.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 3:
        raise ValueError("rho must be 1-D with at least 3 points")
    if not np.all(np.diff(rho) > 0):
        raise ValueError("rho must be strictly increasing")

    steady = times is None
    n_t = 1 if steady else len(np.atleast_1d(times))
    n_rho = rho.size

    n2 = _as_2d(n_e, n_t, n_rho, "n_e")
    T2 = _as_2d(T_e, n_t, n_rho, "T_e")
    S2 = _as_2d(source, n_t, n_rho, "source")
    Vp = rho if volume_element is None else np.asarray(volume_element, float)
    if Vp.shape != rho.shape:
        raise ValueError("volume_element must match rho")

    # (3/2) d(nT)/dt -- zero in steady state.
    if steady or n_t == 1:
        dW_dt = np.zeros_like(n2)
    else:
        t = np.asarray(times, dtype=float)
        dW_dt = 1.5 * np.gradient(n2 * T2, t, axis=0)

    integrand = (S2 - dW_dt) * Vp[None, :]
    # Cumulative trapezoid along rho, anchored at 0.
    dr = np.diff(rho)
    seg = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dr[None, :]
    cum = np.concatenate([np.zeros((n_t, 1)), np.cumsum(seg, axis=1)], axis=1)

    q = np.zeros_like(cum)
    nz = Vp > 0
    q[:, nz] = cum[:, nz] / Vp[None, nz]
    return q


# Points whose |n dT/drho| falls below this fraction of the profile's own
# median are discarded.
#
# THE GUARD HAS TO BE RELATIVE. It used to be a bare `min_gradient=1e-8`, an
# ABSOLUTE threshold five orders of magnitude below the gradient scale of any
# realistic profile -- so it never masked anything, despite masking being the
# documented purpose. chi is a ratio with a denominator that noise can drive
# arbitrarily close to zero, and one such point is enough: on the manufactured
# case at 3% noise a single denominator landed at 0.4% of the median and
# produced a chi 105x off, dragging the MEAN relative error to 4.49 while the
# median stayed at 0.28. It also made the error non-monotone in noise -- 3%
# scored worse than 5% -- because whether a near-zero denominator occurs at all
# is luck, which is how the problem hid.
#
# An absolute threshold is wrong for a second reason: it is dimensional. The
# same 1e-8 means completely different things for T in eV and T in keV, so the
# default could not be correct for both.
#
# 0.05 keeps everything within a factor of 20 of the typical gradient. On clean
# data it removes almost nothing (the profile is smooth and the flattest point
# already sits near 10% of the median); on noisy data it removes exactly the
# points that carry no transport information anyway.
MIN_GRADIENT_FRACTION: float = 0.05


def chi_from_flux(
    rho: np.ndarray,
    q: np.ndarray,
    n_e: np.ndarray,
    T_e: np.ndarray,
    min_gradient: float = 1e-8,
    min_gradient_fraction: float = MIN_GRADIENT_FRACTION,
) -> np.ndarray:
    """Apply the closure once: chi = -q / (n dT/drho).

    Returns NaN where the temperature gradient is too flat to divide by. That
    is deliberate: a flat gradient carries no information about chi, and
    returning a number there would invent one. Callers must mask, and
    ``chi_from_power_balance`` reports the masked fraction.

    Two thresholds, and the second is the one that does the work:
        `min_gradient`          absolute floor, guards true zeros only;
        `min_gradient_fraction` fraction of the profile's own median gradient.
    See ``MIN_GRADIENT_FRACTION``.
    """
    rho = np.asarray(rho, dtype=float)
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n_t, n_rho = q.shape
    n2 = _as_2d(n_e, n_t, n_rho, "n_e")
    T2 = _as_2d(T_e, n_t, n_rho, "T_e")

    dT = np.gradient(T2, rho, axis=1)
    denom = n2 * dT
    mag = np.abs(denom)

    # Scale set per time slice, so a profile that flattens over time is judged
    # against itself rather than against the hottest moment of the discharge.
    # An all-NaN row (a completely flat slice) is a routine condition here, not
    # a problem, so the "all-NaN slice" warning is suppressed rather than left
    # to look like one -- same treatment as the nanmean below.
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scale = np.nanmedian(np.where(mag > min_gradient, mag, np.nan), axis=1)
    scale = np.where(np.isfinite(scale), scale, 0.0)[:, None]

    chi = np.full_like(q, np.nan)
    ok = (mag > min_gradient) & (mag >= min_gradient_fraction * scale)
    chi[ok] = -q[ok] / denom[ok]
    return chi


@dataclass
class ChiEstimate:
    """A chi(rho) estimate plus the diagnostics needed to judge it."""

    rho: np.ndarray
    chi: np.ndarray  # (n_rho,) time-averaged over valid points
    chi_t: np.ndarray  # (n_t, n_rho) before averaging
    valid_fraction: float  # fraction of points with a usable gradient
    q: np.ndarray

    def masked_summary(self) -> str:
        if self.valid_fraction > 0.9:
            return f"{self.valid_fraction:.0%} of points usable"
        return (
            f"WARNING: only {self.valid_fraction:.0%} of points had a "
            "temperature gradient steep enough to identify chi. A flat profile "
            "carries no information about transport; identify over a region "
            "with a real gradient, or heat harder."
        )


def chi_from_power_balance(
    rho: np.ndarray,
    n_e: np.ndarray,
    T_e: np.ndarray,
    source: np.ndarray,
    times: np.ndarray | None = None,
    volume_element: np.ndarray | None = None,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
) -> ChiEstimate:
    """The classical estimator, end to end. This is the baseline to beat.

    Restricted to ``rho_min..rho_max`` by default: chi is 0/0 on the axis
    (both q and dT/drho vanish) and the edge is dominated by the boundary
    condition rather than by transport.
    """
    rho = np.asarray(rho, dtype=float)
    q = heat_flux_from_balance(rho, n_e, T_e, source, times, volume_element)
    chi_t = chi_from_flux(rho, q, n_e, T_e)

    window = (rho >= rho_min) & (rho <= rho_max)
    chi_masked = np.where(window[None, :], chi_t, np.nan)
    valid = np.isfinite(chi_masked)
    frac = float(valid.sum() / max(valid.size, 1))

    # All-NaN columns (outside the window, or flat gradient) are expected, so
    # suppress the "mean of empty slice" warning rather than letting a routine
    # condition look like a problem.
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chi = np.nanmean(chi_masked, axis=0)
    return ChiEstimate(rho=rho, chi=chi, chi_t=chi_t, valid_fraction=frac, q=q)


# ---------------------------------------------------------------------------
# Manufactured solution -- how the maths above is verified without TORAX
# ---------------------------------------------------------------------------


def manufactured_case(
    chi_true,
    n_rho: int = 101,
    T_axis: float = 8.0,
    T_edge: float = 0.2,
    n_axis: float = 1.0,
    peaking: float = 0.8,
) -> dict:
    """Build profiles and the source that makes them EXACTLY consistent.

    Method of manufactured solutions: choose chi(rho), choose n(rho) and
    T(rho), then compute the S that the conservation law demands. Feeding
    (n, T, S) back to the identifier must return the chi we started from.

    S is produced from the DIFFERENTIAL form and chi is recovered from the
    INTEGRAL form, so agreement is a real check rather than a tautology.

    Steady state, so d/dt vanishes and the test isolates the spatial operator.

    Args:
        chi_true: callable rho -> chi. Use a mid-radius bump to emulate the
            "anomalous transport at mid-radius" case the whole method exists
            to detect.
    """
    rho = np.linspace(0.0, 1.0, n_rho)
    T = T_edge + (T_axis - T_edge) * (1.0 - rho**2)
    n = n_axis * (1.0 - peaking * rho**2) + 0.05
    chi = np.asarray(chi_true(rho), dtype=float)

    dT = np.gradient(T, rho)
    q = -n * chi * dT  # the closure, used only to MAKE the case
    # S = (1/rho) d(rho q)/drho, the differential form of conservation.
    rq = rho * q
    S = np.gradient(rq, rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(rho > 0, S / np.where(rho > 0, rho, 1.0), 0.0)
    S[0] = S[1]  # l'Hopital on the axis

    return {"rho": rho, "n_e": n, "T_e": T, "source": S, "chi_true": chi, "q_true": q}


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean relative error over points where both are finite."""
    e, t = np.asarray(estimate, float), np.asarray(truth, float)
    ok = np.isfinite(e) & np.isfinite(t) & (np.abs(t) > 1e-12)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(e[ok] - t[ok]) / np.abs(t[ok])))


def relative_error_summary(estimate: np.ndarray, truth: np.ndarray) -> dict:
    """Mean, median and max relative error, plus the count behind them.

    The mean alone is a bad summary of a RATIO estimator: chi = -q/(n dT/drho)
    and a single near-zero denominator produces an error large enough to set
    the mean by itself. `hfmarl/metrics/curves.asymptotic_performance` already
    carries the same lesson for episode returns. Report all three so a lone
    blow-up is visible as a gap between mean and median rather than silently
    becoming "the error".
    """
    e, t = np.asarray(estimate, float), np.asarray(truth, float)
    ok = np.isfinite(e) & np.isfinite(t) & (np.abs(t) > 1e-12)
    if not ok.any():
        return {"mean": float("nan"), "median": float("nan"),
                "max": float("nan"), "n": 0}
    r = np.abs(e[ok] - t[ok]) / np.abs(t[ok])
    return {"mean": float(r.mean()), "median": float(np.median(r)),
            "max": float(r.max()), "n": int(ok.sum())}
