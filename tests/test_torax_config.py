"""Config construction. Schema validation itself happens in gate0_env.py."""

import pytest

from hfmarl.devices.registry import all_devices, get
from hfmarl.envs.torax_config import build_config


def test_every_device_builds_a_config():
    for d in all_devices():
        cfg = build_config(d)
        assert cfg["geometry"]["geometry_type"] == "circular"
        assert cfg["geometry"]["R_major"] == d.R_major
        assert cfg["geometry"]["B_0"] == d.B_0


def test_uses_flat_transport_form_for_1_4_3():
    """main's core_transport_models registry would break here -- that's intended."""
    cfg = build_config(get("iter_like"))
    assert "model_name" in cfg["transport"]
    assert "core_transport_models" not in cfg["transport"]


def test_actuated_sources_start_at_zero():
    """Otherwise step 1 applies a command nobody asked for."""
    s = build_config(get("iter_like"))["sources"]
    assert s["generic_heat"]["P_total"] == 0.0
    assert s["ecrh"]["P_total"] == 0.0
    assert s["gas_puff"]["S_total"] == 0.0


def test_ecrh_is_an_independent_source():
    """The second heat source the thermal cluster needs."""
    s = build_config(get("iter_like"))["sources"]
    assert "ecrh" in s and "generic_heat" in s


def test_density_is_specified_in_greenwald_fractions():
    """Makes devices physically comparable despite order-of-magnitude SI gaps."""
    pc = build_config(get("iter_like"))["profile_conditions"]
    assert pc["n_e_nbar_is_fGW"] is True
    assert pc["n_e_right_bc_is_fGW"] is True


def test_fusion_source_can_be_disabled():
    assert "fusion" in build_config(get("iter_like"))["sources"]
    assert "fusion" not in build_config(get("iter_like"), enable_fusion=False)["sources"]


def test_rejects_too_few_radial_cells():
    with pytest.raises(ValueError):
        build_config(get("iter_like"), n_rho=2)


def test_rejects_nonpositive_times():
    with pytest.raises(ValueError):
        build_config(get("iter_like"), t_final=0.0)
    with pytest.raises(ValueError):
        build_config(get("iter_like"), fixed_dt=-1.0)


def test_all_evolved_equations_are_on():
    n = build_config(get("iter_like"))["numerics"]
    for k in ("evolve_ion_heat", "evolve_electron_heat", "evolve_current",
              "evolve_density"):
        assert n[k] is True
