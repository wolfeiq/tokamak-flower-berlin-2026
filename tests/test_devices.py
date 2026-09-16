"""Device registry: derived quantities against hand-computed values."""

import math

import pytest

from hfmarl.devices.registry import DEVICES, all_devices, get


def test_all_devices_present():
    assert set(DEVICES) == {"iter_like", "sparc_like", "diiid_like", "tcv_like"}


def test_aspect_ratios_are_physical():
    for d in all_devices():
        assert 1.0 < d.aspect_ratio < 10.0, d.name
        assert 0.0 < d.inverse_aspect_ratio < 1.0, d.name


def test_greenwald_iter_hand_computed():
    """n_GW = Ip[MA] / (pi a^2) x 1e20. ITER at 10.5 MA, a = 2.0 m."""
    d = get("iter_like")
    expected = (10.5 / (math.pi * 4.0)) * 1e20
    assert d.greenwald_density == pytest.approx(expected, rel=1e-12)
    # Sanity against the published ITER figure of ~1.2e20 at 15 MA.
    assert 0.8e20 < d.greenwald_density < 0.9e20


def test_sparc_is_high_density_and_high_field():
    """SPARC's compactness makes its Greenwald limit an order above ITER's."""
    sparc, iter_ = get("sparc_like"), get("iter_like")
    assert sparc.B_0 > 2 * iter_.B_0
    assert sparc.greenwald_density > 5 * iter_.greenwald_density


def test_q_cylindrical_in_operational_range():
    """Every device should sit near q ~ 2-5; outside that the config is wrong."""
    for d in all_devices():
        assert 1.5 < d.q_cylindrical < 6.0, f"{d.name}: q={d.q_cylindrical}"


def test_icrh_excluded_by_default():
    """ICRH has no independent TORAX 1.4.3 source and must not silently alias."""
    d = get("iter_like")
    default = [a.name for a in d.actuators_for("thermal")]
    full = [a.name for a in d.actuators_for("thermal", include_unavailable=True)]
    assert "icrh" not in default
    assert "icrh" in full


def test_aux_heat_and_icrh_share_a_path_which_is_why_icrh_is_disabled():
    d = get("iter_like")
    aux = d.actuator("aux_heat")
    icrh = d.actuator("icrh")
    assert aux.torax_path == icrh.torax_path
    assert icrh.available is False


def test_actuator_clip():
    a = get("iter_like").actuator("aux_heat")
    assert a.clip(-5.0) == a.lo
    assert a.clip(1e12) == a.hi


def test_unknown_device_raises():
    with pytest.raises(KeyError):
        get("jet_like")


def test_every_device_has_an_operating_point():
    """The similarity figure and the Phase 5 bandwidth check both need these."""
    from hfmarl.devices.registry import OPERATING_POINTS, encoded_states

    assert set(OPERATING_POINTS) == set(DEVICES)
    states = encoded_states()
    assert set(states) == set(DEVICES)
    for name, s in states.items():
        assert 1e-4 < s.rho_star < 1e-1, name
        assert 0.0 < s.nu_star < 10.0, name
        assert 0.5 < s.beta_N < 6.0, name


def test_operating_point_returns_a_copy():
    from hfmarl.devices.registry import operating_point

    op = operating_point("iter_like")
    op["T_e_keV"] = 999.0
    assert operating_point("iter_like")["T_e_keV"] != 999.0
