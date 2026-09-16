"""TORAX state -> dimensionless state. The module `dimensionless.py` promised.

Its docstring has said "the TORAX adapter lives in
hfmarl/physics/torax_adapter.py" since the beginning and the file did not
exist. That is why `OPERATING_POINTS` is a hand-written table of nominal
kinetics, and why every aggregation weight in the project has been computed
from coordinates no simulation produced. Measured against the plant, two of the
four devices were placed ~1.8x above any beta_N their plant can reach.

WHAT IS AVERAGED, AND WHY VOLUME
--------------------------------
`encode` requires VOLUME-AVERAGED kinetics, and says why: beta from core values
is a core beta, roughly 1.8x the volume-averaged beta_N TORAX reports and that
the limits are written against, and nu* here is paired with q95 and eps = a/R,
which are global quantities. Core kinetics with edge geometry is a mixture of
two radial locations and means nothing in particular.

The weights come from TORAX's own `geo.vpr`, the volume derivative on the
cell grid, rather than from a formula.

AUDIT #8: this used to assume dV proportional to rho, which is exact for a
cylinder and wrong here. TORAX 1.4.3's circular geometry carries a radially
varying elongation, kappa(rho) = 1 + rho (kappa_LCFS - 1) -- measured on the
installed geometry it runs 1.014 to 1.706 -- so vpr goes as
2 rho + 3 (kappa_LCFS - 1) rho^2. The rho-weighted mean was biased by +9% to
+13% depending on device, which lands directly on the temperature-derived
similarity coordinates. beta_N is taken from TORAX and was never affected.

A plain unweighted mean would be worse still -- it over-weights the core,
which is exactly the error `encode` warns about.

BETA COMES FROM TORAX, NOT FROM US
----------------------------------
`encode` can compute beta_N from the kinetics, but TORAX already publishes its
own `beta_N` from the full profiles and geometry. Recomputing it from two
volume averages would be a worse estimate of the same quantity, and worse: the
operating limits are written against TORAX's number, so a device would sit at
one beta_N for the limit checks and a different one in similarity space.
"""

from __future__ import annotations

import numpy as np

from hfmarl.physics.dimensionless import DimensionlessState, encode

# TORAX reports temperatures in keV and densities in m^-3. A density that
# arrives as ~1e1 rather than ~1e20 is in units of 1e20 m^-3, which some TORAX
# outputs use; rescaling on magnitude is crude but the two are twenty orders of
# magnitude apart, so it cannot misfire quietly.
_N_E_SCALE_THRESHOLD = 1e15


def _volume_average(profile: np.ndarray,
                    vpr: np.ndarray | None = None) -> float:
    """Volume average using TORAX's own volume derivative.

    `vpr` is dV/drho on the cell grid. Without it this falls back to the
    rho weighting, which AUDIT #8 showed is biased by ~9-13% here -- so the
    fallback is a degraded mode, not an equivalent one, and callers that can
    supply the geometry should.
    """
    p = np.asarray(profile, dtype=float).ravel()
    if p.size == 0:
        return float("nan")
    if vpr is not None:
        w = np.asarray(vpr, dtype=float).ravel()
        if w.size != p.size:
            raise ValueError(
                f"vpr has {w.size} cells but the profile has {p.size}; "
                "averaging one grid with another's weights is silent nonsense")
    else:
        w = (np.arange(p.size) + 0.5) / p.size
    good = np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not good.any():
        return float("nan")
    return float(np.sum(p[good] * w[good]) / np.sum(w[good]))


def _as_si_density(value: float) -> float:
    return float(value) * (1e20 if abs(value) < _N_E_SCALE_THRESHOLD else 1.0)


def state_from_torax(
    profiles: dict[str, np.ndarray],
    scalars: dict[str, float],
    device,
    vpr: np.ndarray | None = None,
    # AUDIT #8: 1.6 is what `torax_config.build_config` actually sets.
    # A divergent default here meant the collisionality used for
    # federation was computed from a different plasma than the one being
    # simulated.
    Z_eff: float = 1.6,
    A_i: float = 2.5,
) -> DimensionlessState:
    """Encode one TORAX state into the coordinates federation weights on.

    `profiles` and `scalars` are what `ToraxCore.read_profiles` and
    `read_scalars` return; `device` supplies the geometry, which TORAX is not
    asked for because the registry is already the single source of truth for it.
    """
    T_e = _volume_average(profiles.get("T_e", np.array([])), vpr)
    T_i = _volume_average(profiles.get("T_i", np.array([])), vpr)

    n_e = scalars.get("n_e_volume_avg")
    if n_e is None or not np.isfinite(n_e):
        n_e = _volume_average(profiles.get("n_e", np.array([])), vpr)
    n_e = _as_si_density(n_e)

    q95 = float(scalars.get("q95", float("nan")))
    beta_N = scalars.get("beta_N")
    beta_N = float(beta_N) if beta_N is not None and np.isfinite(beta_N) else None

    return encode(
        T_e_keV=T_e,
        T_i_keV=T_i,
        n_e=n_e,
        q95=q95,
        B_0=device.B_0,
        R_major=device.R_major,
        a_minor=device.a_minor,
        Ip=device.Ip_nominal,
        A_i=A_i,
        Z_eff=Z_eff,
        beta_N=beta_N,
    )


def state_from_env(env, Z_eff: float = 1.6,
                   A_i: float = 2.5) -> DimensionlessState:
    """Where this device is operating RIGHT NOW.

    This is what `ClientUpdate.state` should carry. A static nominal point
    cannot express that a machine drifted out of the regime where similarity
    holds, so `regime_valid` could never fire on one.
    """
    geo = getattr(env.core.state, "geometry", None)
    vpr = np.asarray(geo.vpr, dtype=float) if geo is not None else None
    return state_from_torax(
        env.core.read_profiles(), env.core.read_scalars(), env.device,
        vpr=vpr, Z_eff=Z_eff, A_i=A_i,
    )


def mean_state(states: list[DimensionlessState]) -> DimensionlessState:
    """Average of several measured states, for summarising a whole shot.

    rho* and nu* are averaged in the LOG, because both span decades across this
    device set and an arithmetic mean over a decade is dominated by its largest
    term. `similarity_distance` already log-scales exactly these two; averaging
    them linearly here and comparing them logarithmically there would be two
    different metrics wearing one name.
    """
    if not states:
        raise ValueError("no states to average")
    arr = {k: np.array([getattr(s, k) for s in states], dtype=float)
           for k in ("rho_star", "nu_star", "beta_N", "q95", "mach")}

    def lmean(x):
        good = x[np.isfinite(x) & (x > 0)]
        return float(np.exp(np.mean(np.log(good)))) if good.size else float("nan")

    def amean(x):
        good = x[np.isfinite(x)]
        return float(np.mean(good)) if good.size else float("nan")

    return DimensionlessState(
        rho_star=lmean(arr["rho_star"]),
        nu_star=lmean(arr["nu_star"]),
        beta_N=amean(arr["beta_N"]),
        q95=amean(arr["q95"]),
        mach=amean(arr["mach"]),
    )
