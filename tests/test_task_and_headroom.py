"""Task difficulty and the headroom gate -- 'what kills the claim'."""

import numpy as np
import pytest

from hfmarl.envs.limits import LimitSet
from hfmarl.envs.task import (
    DEFAULT_TASK,
    DIFFICULTY_ORDER,
    PRESETS,
    SetpointSchedule,
    TaskSpec,
    get,
    validate_against_limits,
)
from hfmarl.experiments.headroom import (
    MAX_MATRIX_HOURS,
    MIN_ISOLATED_SHOTS,
    assess,
)
from hfmarl.metrics.curves import ThresholdResult


# -- setpoint schedules --------------------------------------------------


def test_constant_setpoint_does_not_move():
    s = SetpointSchedule("constant", base=2.0)
    assert all(s.target(t, 10.0) == 2.0 for t in (0.0, 5.0, 10.0))


def test_ramp_spans_base_to_base_plus_amplitude():
    s = SetpointSchedule("ramp", base=1.5, amplitude=1.0)
    assert s.target(0.0, 10.0) == pytest.approx(1.5)
    assert s.target(10.0, 10.0) == pytest.approx(2.5)


def test_steps_alternate_so_a_monotone_drift_does_not_solve_it():
    s = SetpointSchedule("steps", base=2.0, amplitude=0.8, n_steps=4)
    vals = [s.target(t, 10.0) for t in np.linspace(0, 9.99, 4)]
    diffs = np.diff(vals)
    assert np.any(diffs > 0) and np.any(diffs < 0)


def test_target_is_clamped_outside_the_episode():
    s = SetpointSchedule("ramp", base=1.0, amplitude=1.0)
    assert s.target(-5.0, 10.0) == pytest.approx(1.0)
    assert s.target(50.0, 10.0) == pytest.approx(2.0)


def test_unknown_schedule_kind_rejected():
    with pytest.raises(ValueError):
        SetpointSchedule("chaos")


def test_nonpositive_base_rejected():
    with pytest.raises(ValueError):
        SetpointSchedule("constant", base=0.0)


# -- presets -------------------------------------------------------------


def test_presets_are_ordered_by_declared_difficulty():
    diffs = [get(n).expected_difficulty for n in DIFFICULTY_ORDER]
    assert diffs == sorted(diffs)


def test_harder_presets_cost_more_steps_per_shot():
    steps = [get(n).steps_per_shot for n in DIFFICULTY_ORDER]
    assert steps == sorted(steps)


def test_every_preset_is_satisfiable():
    """A setpoint above the beta limit is unsatisfiable, not hard.

    Presets are fractions of each device band, so they must be resolved against
    a device before `validate_against_limits` (which works in beta_N) can say
    anything about them.
    """
    from hfmarl.devices.registry import all_devices

    for name in DIFFICULTY_ORDER:
        for d in all_devices():
            assert validate_against_limits(get(name).resolve_for(d)) == [], (
                name, d.name)


def test_setpoint_above_hard_limit_is_caught():
    bad = TaskSpec("bad", SetpointSchedule("constant", base=3.5), 0.1,
                   "constant", True, 10.0, 0.5, setpoint_mode="absolute")
    problems = validate_against_limits(bad)
    assert problems and "HARD limit" in problems[0]


def test_setpoint_past_soft_edge_is_flagged_separately():
    warn = TaskSpec("warn", SetpointSchedule("constant", base=2.7), 0.05,
                    "constant", True, 10.0, 0.5, setpoint_mode="absolute")
    problems = validate_against_limits(warn)
    assert problems and "soft edge" in problems[0]


def test_tolerance_band_straddling_the_limit_is_caught():
    t = TaskSpec("t", SetpointSchedule("constant", base=2.95), 0.2,
                 "constant", True, 10.0, 0.5, setpoint_mode="absolute")
    assert any("tolerance" in p for p in validate_against_limits(t))


def test_escalation_walks_the_order_and_saturates():
    assert get("trivial").harder().name == "easy"
    assert get("brutal").harder().name == "brutal"
    assert get("easy").easier().name == "trivial"
    assert get("trivial").easier().name == "trivial"


def test_default_task_is_not_the_easiest():
    """Starting a headroom sweep at 'trivial' wastes a run."""
    assert DIFFICULTY_ORDER.index(DEFAULT_TASK) >= 2


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        get("impossible")


def test_steps_per_shot_is_consistent():
    for name in DIFFICULTY_ORDER:
        t = get(name)
        assert t.steps_per_shot == pytest.approx(t.episode_length / t.delta_t_a)


# -- the headroom verdict ------------------------------------------------


def _verdict(median, reach=1.0, sec=0.4, budget=3000, seeds=5):
    n = seeds
    n_reach = int(round(reach * n))
    return assess(
        ThresholdResult([float(median)] * n_reach, [budget] * (n - n_reach), -2.0),
        seconds_per_shot=sec, shot_budget=budget, n_seeds=seeds,
    )


def test_fast_isolated_convergence_fails_headroom():
    """The claim-killer: a few hundred shots means no room to show anything."""
    v = _verdict(200)
    assert not v.has_headroom and not v.passes
    assert v.binding_constraint == "headroom"
    assert "KILLS THE CLAIM" in "\n".join(v.advice())


def test_good_task_passes_all_three():
    v = _verdict(2200, sec=0.3, budget=4000)
    assert v.has_headroom and v.is_measurable and v.is_affordable and v.passes
    assert "PASS" in v.summary()


def test_unreliable_convergence_fails_measurability():
    v = _verdict(3000, reach=0.4)
    assert not v.is_measurable and v.binding_constraint == "measurability"
    assert "TOO HARD" in "\n".join(v.advice())


def test_expensive_matrix_fails_affordability():
    v = _verdict(3000, sec=20.0, budget=5000)
    assert v.has_headroom and not v.is_affordable
    assert v.binding_constraint == "cost"
    assert any("PARALLEL" in a for a in v.advice())


def test_cost_advice_protects_the_device_set():
    """Dropping the small machine removes the case federation should help most."""
    v = _verdict(3000, sec=20.0, budget=5000)
    assert any("last resort" in a for a in v.advice())


def test_matrix_hours_scale_with_the_matrix():
    small = assess(ThresholdResult([2000.0] * 3, [], -2.0), 0.5, 3000, n_seeds=3)
    big = assess(ThresholdResult([2000.0] * 3, [], -2.0), 0.5, 3000, n_seeds=9)
    assert big.matrix_hours == pytest.approx(3 * small.matrix_hours)


def test_measurability_is_checked_before_headroom():
    """A task that never converges must not be reported as merely 'too easy'."""
    v = _verdict(500, reach=0.2)
    assert v.binding_constraint == "measurability"


def test_threshold_constants_are_sane():
    assert MIN_ISOLATED_SHOTS >= 500
    assert MAX_MATRIX_HOURS > 0


def test_a_new_task_can_be_band_measured_without_a_band():
    """The circular dependency that made `brink` unmeasurable.

    `resolve_for` needs the device's measured beta_N band; the band is
    produced by `scripts/gate_authority.py`; that script built an env, which
    resolves the task. So a task with no band raised a KeyError telling you to
    run the tool that could not run.

    `for_band_measurement` is the way out: a command sweep never reads the
    setpoint, so an absolute placeholder makes `resolve_for` a no-op while
    leaving transport, episode length and cluster -- everything that actually
    moves the band -- untouched.
    """
    spec = PRESETS["brink"]
    placeholder = spec.for_band_measurement()

    assert placeholder.resolve_for("any_device_at_all") is placeholder
    for attr in ("transport_model", "enable_fusion", "episode_length",
                 "delta_t_a", "clusters", "name"):
        assert getattr(placeholder, attr) == getattr(spec, attr), attr


def test_the_placeholder_target_is_not_usable_for_scoring():
    """Guard against someone reaching for it as a general 'unresolved' spec."""
    p = PRESETS["moderate"].for_band_measurement()
    assert p.setpoint.amplitude == 0.0
    assert p.tolerance == 1.0  # nonsense on purpose, not a real tolerance


def test_brink_sits_below_the_soft_limit_after_resolution():
    """The preset exists to sit NEAR the limit, not past it.

    If the top of the ramp resolved above the beta_N soft edge, tracking and
    containment would be contradictory rather than in tension, and the task
    would be unsatisfiable rather than hard -- which is how `hard` and
    `brutal` went wrong.
    """
    import pytest

    from hfmarl.devices.registry import BETA_N_BANDS, get as get_device
    from hfmarl.envs.limits import DEFAULT_LIMITS
    from hfmarl.envs.task import validate_against_limits

    limits = LimitSet(DEFAULT_LIMITS)
    soft = next(lim.soft for lim in DEFAULT_LIMITS if lim.name == "beta_N")
    checked = 0
    for name in ("diiid_like", "tcv_like"):
        if "brink" not in BETA_N_BANDS.get(name, {}):
            continue  # band not measured yet
        r = PRESETS["brink"].resolve_for(get_device(name), limits)
        assert r.setpoint.base + r.setpoint.amplitude <= soft
        assert validate_against_limits(r, limits) == []
        checked += 1
    if checked == 0:
        pytest.skip("no brink band recorded yet")
