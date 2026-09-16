"""Dimensionless encoder against hand-computed and textbook values."""

import numpy as np
import pytest

from hfmarl.devices.registry import get
from hfmarl.physics.dimensionless import (
    DimensionlessState,
    beta_normalised,
    beta_toroidal,
    coulomb_log_e,
    encode,
    nu_star_e,
    profile_shape_features,
    rho_star,
)


def test_rho_star_iter_matches_textbook():
    """ITER: T_e = 10 keV, B = 5.3 T, a = 2.0 m, D-T.

    rho_s = 1.0217e-4 * sqrt(2.5 * 1e4) / 5.3 = 3.05e-3 m
    rho*   = 3.05e-3 / 2.0 = 1.5e-3, i.e. ~1/650 -- the textbook ITER value.
    """
    rs = float(rho_star(10.0, B_0=5.3, a_minor=2.0, A_i=2.5))
    assert rs == pytest.approx(1.52e-3, rel=0.02)
    assert 1.0 / 800 < rs < 1.0 / 500


def test_rho_star_scales_as_sqrt_T_over_B():
    base = float(rho_star(10.0, 5.0, 1.0))
    assert float(rho_star(40.0, 5.0, 1.0)) == pytest.approx(2 * base, rel=1e-9)
    assert float(rho_star(10.0, 10.0, 1.0)) == pytest.approx(base / 2, rel=1e-9)


def test_rho_star_ordering_across_devices():
    """Small machines have large rho*. This ordering is the whole point."""
    pts = {"iter_like": 10.0, "sparc_like": 10.0, "diiid_like": 3.0, "tcv_like": 1.5}
    vals = {
        n: float(rho_star(T, get(n).B_0, get(n).a_minor)) for n, T in pts.items()
    }
    assert vals["iter_like"] < vals["diiid_like"] < vals["tcv_like"]


def test_sparc_reaches_iter_like_rho_star_despite_being_small():
    """The similarity overlap Phase 6 depends on: high field compensates size."""
    i = float(rho_star(10.0, get("iter_like").B_0, get("iter_like").a_minor))
    s = float(rho_star(10.0, get("sparc_like").B_0, get("sparc_like").a_minor))
    assert s / i < 2.0, "SPARC-like must stay within a factor 2 of ITER-like in rho*"


def test_rho_star_rejects_bad_geometry():
    with pytest.raises(ValueError):
        rho_star(10.0, B_0=0.0, a_minor=2.0)
    with pytest.raises(ValueError):
        rho_star(10.0, B_0=5.0, a_minor=-1.0)


def test_coulomb_log_in_tokamak_range():
    assert 15.0 < float(coulomb_log_e(1e20, 10.0)) < 20.0


def test_coulomb_log_is_clipped_not_nan_at_extremes():
    for n, T in [(0.0, 0.0), (1e30, 1e-6), (1e10, 1e6)]:
        v = float(coulomb_log_e(n, T))
        assert np.isfinite(v) and 10.0 <= v <= 25.0


def test_nu_star_hot_plasma_is_collisionless():
    """Core ITER should sit deep in the banana regime, nu* << 1."""
    d = get("iter_like")
    v = float(nu_star_e(1e20, 10.0, q=3.0, R_major=d.R_major,
                        epsilon=d.inverse_aspect_ratio))
    assert 0.0 < v < 0.1


def test_nu_star_rises_steeply_as_temperature_falls():
    """nu* ~ T^-2, so a cold edge is far more collisional than the core."""
    d = get("iter_like")
    kw = dict(q=3.0, R_major=d.R_major, epsilon=d.inverse_aspect_ratio)
    hot = float(nu_star_e(1e20, 10.0, **kw))
    cold = float(nu_star_e(1e20, 1.0, **kw))
    assert cold > 50 * hot


def test_nu_star_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        nu_star_e(1e20, 10.0, q=3.0, R_major=6.2, epsilon=1.5)


def test_beta_toroidal_hand_computed():
    """beta = 2*mu_0*p/B^2, p = n(T_e + T_i)e."""
    n, T, B = 1e20, 10.0, 5.0
    p = 2 * n * T * 1e3 * 1.602176634e-19
    expected = 2 * (4e-7 * np.pi) * p / B**2
    assert float(beta_toroidal(n, T, T, B)) == pytest.approx(expected, rel=1e-9)


def test_beta_normalised_definition():
    assert beta_normalised(0.02, a_minor=2.0, B_0=5.3, Ip=10.5e6) == pytest.approx(
        2.0 * 2.0 * 5.3 / 10.5, rel=1e-9
    )


def test_encode_produces_physical_iter_point():
    d = get("iter_like")
    s = encode(T_e_keV=10.0, T_i_keV=10.0, n_e=1e20, q95=3.0, B_0=d.B_0,
               R_major=d.R_major, a_minor=d.a_minor, Ip=d.Ip_nominal)
    assert isinstance(s, DimensionlessState)
    assert 1e-3 < s.rho_star < 3e-3
    assert 0 < s.nu_star < 1
    assert 1.0 < s.beta_N < 5.0
    assert s.q95 == 3.0


def test_vector_order_matches_labels():
    s = DimensionlessState(1e-3, 0.1, 2.0, 3.0, 0.2)
    assert DimensionlessState.labels() == ("rho_star", "nu_star", "beta_N", "q95", "mach")
    assert np.allclose(s.as_vector(), [1e-3, 0.1, 2.0, 3.0, 0.2])


def test_profile_shape_is_scale_invariant():
    """Doubling a profile must not change its shape descriptors."""
    rho = np.linspace(0, 1, 25)
    prof = 1.0 - 0.8 * rho**2
    a = profile_shape_features(prof, rho)
    b = profile_shape_features(2.0 * prof, rho)
    assert np.allclose(a, b, rtol=1e-9)


def test_profile_shape_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        profile_shape_features(np.ones(10), np.linspace(0, 1, 11))
