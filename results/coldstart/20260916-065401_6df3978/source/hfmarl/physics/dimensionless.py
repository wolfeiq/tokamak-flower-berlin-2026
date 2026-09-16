"""The dimensionless encoder -- SPEC.md §4a, the core contribution.

Tokamak transport is scale-invariant in a small set of dimensionless
parameters. Connor-Taylor similarity (Connor & Taylor, Nucl. Fusion 17, 1047,
1977) holds that two plasmas matched in (rho*, nu*, beta, q) behave
identically regardless of absolute machine size. The entire field's
extrapolation to ITER rests on this, which is why it is a defensible basis for
sharing policy weights across devices rather than an invented trick.

DESIGN NOTE -- why this module is pure NumPy
--------------------------------------------
Nothing here imports TORAX or JAX. Every function takes plain arrays and
scalars. That is deliberate:

  * the formulas can be unit-tested against hand-computed values on any
    machine, with no GPU, no JIT and no 1-minute compile;
  * the same code encodes a live simulation state, a published device
    operating point, or a historical shot.

The TORAX adapter lives in ``hfmarl/physics/torax_adapter.py`` and is the only
place that knows about ``SimState``.

UNITS -- stated once, enforced everywhere
------------------------------------------
    T_e, T_i   keV
    n_e, n_i   m^-3
    B          T
    R, a       m
Everything else is dimensionless. Functions assert on this where cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# np.trapezoid replaced np.trapz in NumPy 2.0. The target environment is
# NumPy >= 2, but this module is deliberately runnable anywhere (it is the one
# piece of physics that must be testable without a full install), so bind
# whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Physical constants (SI), CODATA.
_M_P = 1.67262192e-27  # kg, proton mass
_E = 1.602176634e-19  # C, elementary charge
_MU_0 = 4.0e-7 * np.pi  # H/m

# Practical-units coefficient for the ion sound gyroradius,
#     rho_s = sqrt(m_i T_e)/(e B) = C * sqrt(A_i * T_e[eV]) / B[T],
#     C = sqrt(m_p * e) / e = sqrt(m_p / e).
# Derived rather than hardcoded so the constants above are the single source of
# truth; equals 1.0217e-4 to five figures.
_RHO_S_COEFF = np.sqrt(_M_P / _E)


# ---------------------------------------------------------------------------
# Coulomb logarithm
# ---------------------------------------------------------------------------


def coulomb_log_e(n_e: np.ndarray, T_e_keV: np.ndarray) -> np.ndarray:
    """Electron Coulomb logarithm, Wesson *Tokamaks* 4th ed. §2.15.

        ln A_e = 31.3 - ln( sqrt(n_e) / T_e )      [n_e in m^-3, T_e in eV]

    Typical tokamak core value is 15-18; we clip to [10, 25] because the
    formula misbehaves at the very low temperatures that can appear in a
    transient, and a wrong ln A of 2x is far less damaging than a NaN
    propagating into a policy gradient.
    """
    T_e_eV = np.asarray(T_e_keV, dtype=float) * 1e3
    n_e = np.asarray(n_e, dtype=float)
    safe_T = np.maximum(T_e_eV, 1.0)
    safe_n = np.maximum(n_e, 1e16)
    val = 31.3 - np.log(np.sqrt(safe_n) / safe_T)
    return np.clip(val, 10.0, 25.0)


# ---------------------------------------------------------------------------
# The four similarity parameters
# ---------------------------------------------------------------------------


def rho_star(
    T_e_keV: np.ndarray, B_0: float, a_minor: float, A_i: float = 2.5
) -> np.ndarray:
    """Normalised ion sound gyroradius, rho* = rho_s / a.

        rho_s = sqrt(m_i * T_e) / (e * B)

    In practical units this reduces to

        rho_s [m] = 1.0217e-4 * sqrt(A_i * T_e[eV]) / B[T]

    Sanity check that this file's tests pin: ITER-like, T_e = 10 keV, B = 5.3 T,
    A_i = 2.5 (D-T) gives rho_s = 3.0 mm and, with a = 2.0 m, rho* = 1.5e-3
    -- i.e. of order 1/650, the textbook ITER value.

    rho* is the parameter that most strongly separates machine sizes, and
    therefore does most of the work in the similarity distance of §4b.
    """
    if B_0 <= 0 or a_minor <= 0:
        raise ValueError("B_0 and a_minor must be positive")
    T_e_eV = np.maximum(np.asarray(T_e_keV, dtype=float) * 1e3, 1.0)
    rho_s = _RHO_S_COEFF * np.sqrt(A_i * T_e_eV) / B_0
    return rho_s / a_minor


def nu_star_e(
    n_e: np.ndarray,
    T_e_keV: np.ndarray,
    q: float,
    R_major: float,
    epsilon: float,
    Z_eff: float = 1.5,
) -> np.ndarray:
    """Electron collisionality, Wesson *Tokamaks* 4th ed. eq. (4.10.2).

        nu*_e = 6.921e-18 * q * R * n_e * Z_eff * ln A_e
                / ( eps^{3/2} * T_e[eV]^2 )

    nu* < 1 is the banana (collisionless) regime, nu* > 1 the
    Pfirsch-Schlueter (collisional) regime. Transport physics differs
    qualitatively across nu* = 1, which is precisely why averaging policy
    weights between devices on opposite sides of it is unsafe -- and why §4b
    downweights it.
    """
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon = a/R must be in (0,1), got {epsilon}")
    T_e_eV = np.maximum(np.asarray(T_e_keV, dtype=float) * 1e3, 1.0)
    n_e = np.asarray(n_e, dtype=float)
    ln_A = coulomb_log_e(n_e, T_e_keV)
    return (
        6.921e-18 * q * R_major * n_e * Z_eff * ln_A / (epsilon**1.5 * T_e_eV**2)
    )


def beta_toroidal(
    n_e: np.ndarray, T_e_keV: np.ndarray, T_i_keV: np.ndarray, B_0: float
) -> np.ndarray:
    """Toroidal beta, ratio of kinetic to magnetic pressure.

        beta_t = 2 * mu_0 * p / B_0^2,     p = n_e*T_e + n_i*T_i

    Quasi-neutrality with a single main ion species is assumed, n_i = n_e;
    with impurities this overestimates p slightly.

    **n_e and T must be volume-averaged.** Fed core values this returns a core
    beta, which is roughly 1.8x the volume-averaged beta TORAX reports -- a
    different physical quantity with the same name. TORAX's ``beta_N`` and
    ``beta_tor`` are always preferable once a simulation is running; this
    exists for characterising a device before one does.
    """
    n_e = np.asarray(n_e, dtype=float)
    p = n_e * np.asarray(T_e_keV, float) * 1e3 * _E + n_e * np.asarray(
        T_i_keV, float
    ) * 1e3 * _E
    return 2.0 * _MU_0 * p / B_0**2


def beta_normalised(beta_t: float, a_minor: float, B_0: float, Ip: float) -> float:
    """beta_N = beta_t[%] * a * B_0 / I_p[MA] -- the Troyon-normalised beta.

    The ideal-MHD no-wall limit sits near beta_N = 2.8-4; this is the
    quantity the stability limit in the reward is written against.
    """
    return float(beta_t * 100.0 * a_minor * B_0 / (Ip / 1e6))


# ---------------------------------------------------------------------------
# The encoded state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionlessState:
    """A plasma's location in similarity space.

    These five numbers are what every agent observes and what the federation
    distance metric is computed on. Two plasmas with the same values are, by
    Connor-Taylor, the same plasma.
    """

    rho_star: float
    nu_star: float
    beta_N: float
    q95: float
    mach: float = 0.0  # toroidal rotation / sound speed; 0 until rotation is modelled

    def as_vector(self) -> np.ndarray:
        """Order is fixed and load-bearing -- the distance metric indexes it."""
        return np.array(
            [self.rho_star, self.nu_star, self.beta_N, self.q95, self.mach],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @staticmethod
    def labels() -> tuple[str, ...]:
        return ("rho_star", "nu_star", "beta_N", "q95", "mach")


def encode(
    *,
    T_e_keV: float,
    T_i_keV: float,
    n_e: float,
    q95: float,
    B_0: float,
    R_major: float,
    a_minor: float,
    Ip: float,
    A_i: float = 2.5,
    Z_eff: float = 1.5,
    mach: float = 0.0,
    beta_N: float | None = None,
) -> DimensionlessState:
    """SI plasma state -> dimensionless state. The encoder of SPEC.md §4a.

    Scalars in, scalars out: profile shape is deliberately not encoded here --
    see ``profile_shape_features``, kept separate because shape similarity is a
    weaker and more contestable claim than four-parameter scale invariance.

    **Pass VOLUME-AVERAGED n_e and T.** Two reasons, both of which produced
    wrong numbers before this was enforced:

      * beta computed from core values is a CORE beta, about 1.8x the
        volume-averaged ``beta_N`` that TORAX reports and that the operating
        limits are written against. Same name, different quantity -- and a
        device would then sit somewhere different in similarity space before a
        run than during one.
      * nu* here is paired with q95 and eps = a/R, which are global/edge
        quantities. Volume-averaged kinetics make that a coherent *global*
        collisionality; core kinetics with edge geometry is a mixture of two
        radial locations and means nothing in particular.

    Args:
        beta_N: pass TORAX's own ``beta_N`` when a simulation is running. It is
            computed from the real pressure profile and flux surfaces, so it is
            strictly better than anything derivable from two scalars. The
            n_e/T-based estimate is only for placing a device before any run
            exists.
    """
    epsilon = a_minor / R_major
    rs = float(rho_star(T_e_keV, B_0, a_minor, A_i))
    ns = float(nu_star_e(n_e, T_e_keV, q95, R_major, epsilon, Z_eff))
    if beta_N is None:
        bt = float(beta_toroidal(n_e, T_e_keV, T_i_keV, B_0))
        bn = beta_normalised(bt, a_minor, B_0, Ip)
    else:
        bn = float(beta_N)
    return DimensionlessState(
        rho_star=rs, nu_star=ns, beta_N=bn, q95=float(q95), mach=float(mach)
    )


def profile_shape_features(profile: np.ndarray, rho_norm: np.ndarray) -> np.ndarray:
    """Scale-free descriptors of a radial profile.

    Normalising a profile by its own volume average removes the units and the
    machine size, leaving shape. Returned features are peaking factor
    (centre / average), the normalised gradient at mid-radius, and the edge
    fraction.

    This is a WEAKER similarity claim than the four-parameter one above and is
    kept out of ``encode`` for that reason -- it is a convenience for the
    encoder network's input, not something Connor-Taylor licenses.
    """
    profile = np.asarray(profile, dtype=float)
    rho_norm = np.asarray(rho_norm, dtype=float)
    if profile.shape != rho_norm.shape:
        raise ValueError("profile and rho_norm must have the same shape")
    avg = _trapz(profile * rho_norm, rho_norm) * 2.0
    avg = avg if abs(avg) > 1e-30 else 1e-30
    p = profile / avg
    mid = len(p) // 2
    grad_mid = float(np.gradient(p, rho_norm)[mid])
    return np.array([float(p[0]), grad_mid, float(p[-1])], dtype=float)
