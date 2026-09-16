"""Forward transport solve producing the profiles the investigation analyses.

This replaces the method-of-manufactured-solutions fixture the demo used to
call. `closure.manufactured_case` picks T(rho) and back-computes the source
that makes it consistent: a unit-test device for the estimator, not physics,
and every facility looked alike because nothing device-specific entered it.

Here the causality runs the physical way round. A source profile and a
transport model are given, and T(rho) is SOLVED from steady-state power
balance on a real machine's geometry:

    (1/rho) d(rho q)/drho = S        conservation, exact
    q = -n chi dT/drho               closure, a model

chi is stiff: above a critical normalised gradient it rises sharply, which is
what makes tokamak profiles resilient and what makes "apparent transport"
a genuinely ambiguous measurement. The equation is therefore nonlinear and is
solved numerically, inward from the separatrix.

This is a REDUCED 1-D model, not TORAX. It is cylindrical (V' proportional to
rho, matching closure.py), electron-channel only, steady state, with no
equilibrium, no pedestal physics, no fusion or radiation terms, and a
hand-written chi rather than a gyrokinetic surrogate. When TORAX-generated
fixtures are present `core.py` uses those instead; see
scripts/generate_thermal_fixtures.py. Provenance is reported either way, and
must never be presented as a validated transport prediction.
"""
from __future__ import annotations

import numpy as np

# Practical-units ion sound gyroradius coefficient, matching
# hfmarl/physics/dimensionless.py so a device sits in the same place here as
# it does in the repo's similarity space.
_M_P = 1.67262192e-27
_E = 1.602176634e-19
_RHO_S_COEFF = float(np.sqrt(_M_P / _E))

# Stiff-transport parameters. Illustrative, NOT calibrated against experiment
# or against any gyrokinetic model. The threshold is on the normalised
# logarithmic gradient R/L_T, the quantity critical-gradient transport models
# are actually written in.
CRITICAL_R_OVER_LT = 4.0
STIFFNESS = 1.5
CHI_FLOOR_FRACTION = 0.2
KEV_TO_J = 1.0e3 * _E


def gyro_bohm_chi(T_keV: float, B_0: float, a_minor: float, A_i: float = 2.5) -> float:
    """Gyro-Bohm transport scale, chi_gB = rho_s^2 c_s / a, in m^2/s.

    This is where device identity enters the profile: a compact high-field
    machine and a large low-field one get different transport for the same
    heating, so their profiles differ for physical reasons rather than because
    a fixture was hand-tuned per site.
    """
    T_eV = max(float(T_keV) * 1e3, 1.0)
    rho_s = _RHO_S_COEFF * np.sqrt(A_i * T_eV) / float(B_0)
    c_s = np.sqrt(float(_E) * T_eV / (A_i * _M_P))
    return float(rho_s**2 * c_s / float(a_minor))


def chi_of_gradient(chi_gb: float, r_over_lt: np.ndarray) -> np.ndarray:
    """Stiff chi: a floor plus a sharp rise above the critical gradient."""
    excess = np.maximum(0.0, np.asarray(r_over_lt, dtype=float) - CRITICAL_R_OVER_LT)
    return chi_gb * (CHI_FLOOR_FRACTION + STIFFNESS * excess**1.5)


def source_profile(r: np.ndarray, total_power: float, a_minor: float,
                   elongation: float, R_major: float, width: float = 0.35) -> np.ndarray:
    """Gaussian on-axis absorbed heating in W/m^3, integrating to total_power."""
    shape = np.exp(-((r / (width * a_minor)) ** 2))
    # Cylindrical volume element dV = (2 pi R)(2 pi kappa r) dr.
    volume_element = 4.0 * np.pi**2 * R_major * elongation * r
    integral = float(np.trapezoid(shape * volume_element, r))
    return shape * (float(total_power) / max(integral, 1e-30))


def density_profile(r: np.ndarray, n_axis: float, a_minor: float, peaking: float = 0.8) -> np.ndarray:
    return n_axis * (1.0 - peaking * (r / a_minor) ** 2)


def solve_temperature(r, n_e, source, chi_gb, T_edge_J, R_major):
    """Integrate steady-state power balance inward from the separatrix.

    q(r) follows from conservation alone. The closure then gives dT/dr, but chi
    depends on that same gradient through the stiffness term, so each step
    solves a scalar fixed point rather than evaluating a formula.

    Everything here is SI: r in m, n_e in m^-3, source in W/m^3, T in joules,
    chi in m^2/s. Mixing normalised radius with an SI chi silently produces a
    flat profile, which is exactly as wrong as it looks.
    """
    r = np.asarray(r, dtype=float)
    integrand = source * r
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(r) * 0.5 * (integrand[1:] + integrand[:-1]))]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(r > 0, cumulative / np.where(r > 0, r, 1.0), 0.0)

    T = np.empty_like(r)
    chi_cell = np.zeros_like(r)
    T[-1] = float(T_edge_J)
    for i in range(len(r) - 2, -1, -1):
        step = r[i + 1] - r[i]
        q_mid = 0.5 * (q[i] + q[i + 1])
        n_mid = 0.5 * (n_e[i] + n_e[i + 1])
        T_out = T[i + 1]

        def flux(T_in):
            """Conducted flux across this cell for an inner temperature T_in.

            Monotonically increasing in T_in: a steeper gradient raises both the
            gradient itself and, through the stiffness term, chi. That
            monotonicity is what makes the bracket below safe.
            """
            grad = (T_in - T_out) / step
            T_mid = max(0.5 * (T_in + T_out), 1e-300)
            chi = max(float(chi_of_gradient(chi_gb, np.array(R_major * abs(grad) / T_mid))), 1e-12)
            return n_mid * chi * grad, chi

        # Bracket: zero gradient carries nothing; the chi floor carries the most
        # a cell can at a given gradient, so it bounds the root from above.
        lo = T_out
        chi_floor = max(chi_gb * CHI_FLOOR_FRACTION, 1e-12)
        hi = T_out + step * q_mid / (n_mid * chi_floor)
        if q_mid <= 0.0 or hi <= lo:
            T[i] = T_out
            chi_cell[i] = chi_floor
            continue
        # A damped fixed point stalls on the stiff branch; bisection cannot.
        chi_root = chi_floor
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            value, chi_root = flux(mid)
            if value < q_mid:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-14 * abs(hi):
                break
        T[i] = 0.5 * (lo + hi)
        chi_cell[i] = chi_root
    # chi the march converged to, on the same one-sided differences it used.
    chi_cell[-1] = chi_cell[-2] if len(chi_cell) > 1 else 0.0
    return T, chi_cell, q


def simulate(device: dict, power_fraction: float = 1.0) -> dict:
    """Solve one device's core transport and return SI profiles plus true chi.

    `device` carries machine parameters copied from hfmarl/devices/registry.py
    by scripts/prepare_thermal_investigation.py. `power_fraction` scales that
    device's own auxiliary heating, so a weakly heated case is weak relative to
    what the machine can actually do rather than to a shared constant.

    The domain stops at the pedestal top. Solving to the separatrix with a cold
    boundary makes the model carry the entire edge heat flux by conduction,
    which it answers with an unphysical chi of order 10^4 m^2/s; the H-mode
    pedestal is separate physics this model does not attempt.
    """
    n_rho = int(device.get("n_rho", 101))
    a_minor = float(device["a_minor"])
    R_major = float(device["R_major"])
    elongation = float(device["elongation"])
    rho_bc = float(device.get("rho_boundary", 0.85))
    r = np.linspace(0.0, rho_bc * a_minor, n_rho)

    T_ref = float(device["T_e_keV"])
    T_bc_J = float(device["T_pedestal_keV"]) * KEV_TO_J
    n_e = density_profile(r, float(device["n_e"]), a_minor)
    power = float(device["P_aux"]) * float(power_fraction)
    source = source_profile(r, power, a_minor, elongation, R_major)
    chi_gb = gyro_bohm_chi(T_ref, float(device["B_0"]), a_minor)
    T_e, chi_true, q = solve_temperature(r, n_e, source, chi_gb, T_bc_J, R_major)
    # chi_true is ground truth: never exported to a model context.
    return {
        "r": r,
        "rho": r / a_minor,
        "n_e": n_e,
        "T_e": T_e,
        "T_e_keV": T_e / KEV_TO_J,
        "source": source,
        "chi_true": chi_true,
        "q": q,
        "chi_gb": chi_gb,
        "total_power": power,
        "device": device["name"],
        "provenance": "reduced-1d-transport-v1",
    }
