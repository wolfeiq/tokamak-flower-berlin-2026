"""Actuators and the no-recompilation contract.

The two-breakpoint property tested here is what keeps TORAX JIT-compiled
across steps. If these tests fail, training will be unusably slow.
"""

import numpy as np
import pytest

from hfmarl.devices.registry import get
from hfmarl.envs.actuators import (
    STATIC_EXACT,
    ActuatorBank,
    ActuatorSpec,
    StaticParameterError,
    assert_not_static,
    bank_from_device,
)


# -- the JAX_STATIC guard ------------------------------------------------


@pytest.mark.parametrize("path", sorted(STATIC_EXACT))
def test_every_known_static_path_is_rejected(path):
    with pytest.raises(StaticParameterError):
        assert_not_static(path)


@pytest.mark.parametrize(
    "path",
    ["sources.ecrh.mode", "sources.generic_heat.is_explicit",
     "profile_conditions.Ip.interpolation_mode", "numerics.is_bool_param"],
)
def test_static_suffixes_are_rejected_anywhere(path):
    with pytest.raises(StaticParameterError):
        assert_not_static(path)


@pytest.mark.parametrize(
    "path",
    ["sources.generic_heat.P_total", "sources.ecrh.P_total",
     "sources.gas_puff.S_total", "profile_conditions.Ip", "numerics.t_final"],
)
def test_real_actuator_paths_are_allowed(path):
    assert_not_static(path)


def test_guard_fires_at_construction_not_at_runtime():
    with pytest.raises(StaticParameterError):
        ActuatorSpec("bad", "solver.solver_type", 0.0, 1.0, "thermal")


def test_spec_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        ActuatorSpec("x", "sources.ecrh.P_total", 10.0, 1.0, "thermal")


# -- unit mapping --------------------------------------------------------


def test_unit_mapping_spans_the_envelope():
    s = ActuatorSpec("p", "sources.ecrh.P_total", 0.0, 20e6, "thermal")
    assert s.from_unit(-1.0) == pytest.approx(0.0)
    assert s.from_unit(0.0) == pytest.approx(10e6)
    assert s.from_unit(1.0) == pytest.approx(20e6)


def test_unit_mapping_clamps_out_of_range_actions():
    s = ActuatorSpec("p", "sources.ecrh.P_total", 0.0, 20e6, "thermal")
    assert s.from_unit(-5.0) == pytest.approx(0.0)
    assert s.from_unit(5.0) == pytest.approx(20e6)


def test_unit_roundtrip():
    s = ActuatorSpec("p", "sources.ecrh.P_total", 1e6, 20e6, "thermal")
    for v in (1e6, 5e6, 20e6):
        assert s.from_unit(s.to_unit(v)) == pytest.approx(v, rel=1e-9)


# -- bank ----------------------------------------------------------------


def test_bank_rejects_duplicate_names():
    s = ActuatorSpec("p", "sources.ecrh.P_total", 0.0, 1.0, "thermal")
    with pytest.raises(ValueError, match="duplicate"):
        ActuatorBank([s, s])


def test_bank_rejects_empty():
    with pytest.raises(ValueError):
        ActuatorBank([])


def test_bank_from_device_filters_by_cluster():
    d = get("iter_like")
    assert bank_from_device(d, ("thermal",)).names == ("aux_heat", "ecrh")
    assert bank_from_device(d, ("particle",)).names == ("gas_puff",)
    assert bank_from_device(d, ("current",)).names == ("ip",)


def test_wrong_action_count_raises():
    b = bank_from_device(get("iter_like"), ("thermal",))
    with pytest.raises(ValueError):
        b.from_unit_actions(np.array([0.0]))


# -- THE no-recompile contract -------------------------------------------


def test_updates_always_have_exactly_two_breakpoints(fake_torax):
    """The single property that keeps TORAX compiled across steps."""
    b = bank_from_device(get("iter_like"), ("thermal",))
    t = 0.0
    for i in range(8):
        vals = b.from_unit_actions(np.full(len(b), (i % 5) / 4.0 * 2 - 1))
        updates = b.build_updates(vals, t, 0.5)
        for path, u in updates.items():
            assert u.time.shape == (2,), f"{path} has {u.time.shape}, must be (2,)"
            assert u.value.shape == (2,), f"{path} has {u.value.shape}, must be (2,)"
        t += 0.5


def test_update_shapes_are_identical_across_steps(fake_torax):
    """Constant shape is what JAX keys its compilation cache on."""
    b = bank_from_device(get("iter_like"), ("thermal",))
    shapes = []
    t = 0.0
    for i in range(5):
        vals = b.from_unit_actions(np.full(len(b), 0.2 * i))
        u = b.build_updates(vals, t, 0.5)
        shapes.append({k: (v.time.shape, v.value.shape) for k, v in u.items()})
        t += 0.5
    assert all(s == shapes[0] for s in shapes)


def test_ramp_starts_from_the_previous_command(fake_torax):
    """value[0] must be where the actuator was, not zero, or the plasma sees a jump."""
    b = bank_from_device(get("iter_like"), ("thermal",))
    first = b.build_updates(b.from_unit_actions(np.array([0.0, 0.0])), 0.0, 0.5)
    prev_end = {k: v.value[1] for k, v in first.items()}
    second = b.build_updates(b.from_unit_actions(np.array([1.0, 1.0])), 0.5, 0.5)
    for k, u in second.items():
        assert u.value[0] == pytest.approx(prev_end[k])


def test_ramp_times_span_the_action_window(fake_torax):
    b = bank_from_device(get("iter_like"), ("thermal",))
    u = b.build_updates(b.from_unit_actions(np.array([0.5, 0.5])), 3.0, 0.25)
    for x in u.values():
        assert x.time[0] == pytest.approx(3.0)
        assert x.time[1] == pytest.approx(3.25)


def test_updates_are_float64(fake_torax):
    """TORAX runs in double precision; a float32 leaf changes the dtype and recompiles."""
    b = bank_from_device(get("iter_like"), ("thermal",))
    u = b.build_updates(b.from_unit_actions(np.array([0.5, 0.5])), 0.0, 0.5)
    for x in u.values():
        assert x.time.dtype == np.float64
        assert x.value.dtype == np.float64


def test_colliding_paths_are_summed_not_silently_dropped(fake_torax):
    """aux_heat and icrh share generic_heat in 1.4.3; losing one is invisible."""
    d = get("iter_like")
    b = bank_from_device(d, ("thermal",), include_unavailable=True)
    assert b.names == ("aux_heat", "ecrh", "icrh")
    vals = np.array([10e6, 5e6, 4e6])
    u = b.build_updates(vals, 0.0, 0.5)
    # One entry for the shared path, carrying aux_heat + icrh.
    assert u["sources.generic_heat.P_total"].value[1] == pytest.approx(14e6)
    assert u["sources.ecrh.P_total"].value[1] == pytest.approx(5e6)


def test_reset_returns_actuators_to_the_bottom_of_the_envelope(fake_torax):
    b = bank_from_device(get("iter_like"), ("thermal",))
    b.build_updates(b.from_unit_actions(np.array([1.0, 1.0])), 0.0, 0.5)
    assert b.prev_values.max() > 0
    b.reset()
    assert np.allclose(b.prev_values, [s.lo for s in b.specs])


def test_build_updates_rejects_nonpositive_dt(fake_torax):
    b = bank_from_device(get("iter_like"), ("thermal",))
    with pytest.raises(ValueError):
        b.build_updates(np.array([1e6, 1e6]), 0.0, 0.0)
