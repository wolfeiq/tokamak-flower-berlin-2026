"""Cold-start endpoint and small-seed uncertainty without simulator imports."""

import math

import pytest

from hfmarl.metrics.curves import (
    ThresholdResult,
    binomial_exact_interval,
    joint_success_rate,
    paired_joint_reach,
    shots_to_competence_evaluated,
    shots_to_joint_competence_evaluated,
    speedup_from_results,
)
from hfmarl.metrics.log import RunLog, ShotRecord


def run(seed=0, success=True, n=6):
    log = RunLog("test", "test_device", seed)
    for i in range(n):
        log.add(ShotRecord(shot=i, reward=-0.01, steps=20,
                           beta_error=0.01 if success else 1.0,
                           is_evaluation=True))
    return log


@pytest.mark.parametrize("field,value", [
    ("violations", ("beta_N",)),
    ("terminated_early", True),
    ("steps", 0),
    ("steps", 19),
    ("beta_error", float("nan")),
    ("beta_error", float("inf")),
    ("beta_error", 0.2),
])
def test_joint_competence_rejects_every_failed_endpoint(field, value):
    log = run()
    for shot in log.shots:
        setattr(shot, field, value)
    result = shots_to_joint_competence_evaluated(logs := [log], 0.1,
                                               expected_steps=20)
    assert result.n_reached == 0
    assert result.censored_at == [6]
    assert joint_success_rate(logs, 0.1, expected_steps=20) == 0.0


def test_early_termination_is_failure_without_a_step_count_argument():
    log = run()
    for shot in log.shots:
        shot.terminated_early = True
    assert joint_success_rate([log], 0.1) == 0.0
    assert shots_to_joint_competence_evaluated([log], 0.1).n_reached == 0


def test_missing_error_cannot_be_smoothed_out_of_the_joint_endpoint():
    log = run(n=4)
    for i in (1, 3):
        log.shots[i].beta_error = float("nan")
    assert shots_to_competence_evaluated([log], 0.1).n_reached == 1
    assert shots_to_joint_competence_evaluated([log], 0.1).n_reached == 0


def test_failed_evaluation_resets_streak_and_records_confirming_shot():
    log = run(n=11)
    for shot in log.shots:
        shot.is_evaluation = shot.shot in (0, 3, 6, 10)
    log.shots[3].violations = ("solver",)
    result = shots_to_joint_competence_evaluated([log], 0.1)
    assert result.values == [11.0]
    assert (result.window, result.persistence) == (1, 2)


def test_rejected_handover_cannot_certify_installed_controller():
    log = run(n=2)
    log.shots[0].evaluation_eligible = False
    assert shots_to_joint_competence_evaluated([log], 0.1).n_reached == 0
    assert shots_to_competence_evaluated([log], 0.1).n_reached == 1


@pytest.mark.parametrize("kwargs", [
    {"tolerance": float("nan")}, {"tolerance": 0}, {"tolerance": -1},
    {"tolerance": 0.1, "persistence": 0},
    {"tolerance": 0.1, "expected_steps": 0},
])
def test_invalid_endpoint_parameters_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        shots_to_joint_competence_evaluated([run()], **kwargs)


def test_exact_binomial_boundary_intervals_do_not_collapse():
    lo, hi = binomial_exact_interval(0, 6)
    assert lo == 0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 6))
    lo2, hi2 = binomial_exact_interval(6, 6)
    assert lo2 == pytest.approx(1 - hi)
    assert hi2 == 1
    assert all(math.isnan(x) for x in binomial_exact_interval(0, 0))


def test_exact_binomial_interval_matches_known_symmetric_example():
    assert binomial_exact_interval(3, 6) == pytest.approx(
        (0.1181172487570252, 0.8818827512429748))


@pytest.mark.parametrize("wins,p_value", [(0, 1), (5, 0.0625), (6, 0.03125)])
def test_paired_reach_uses_discordant_seeds_not_independent_fisher(wins, p_value):
    baseline = [run(seed=i, success=False) for i in range(6)]
    method = [run(seed=i, success=i < wins) for i in reversed(range(6))]
    result = paired_joint_reach(baseline, method, 0.1)
    assert result.n_pairs == 6
    assert result.method_only == wins
    assert result.baseline_only == 0
    assert result.difference == pytest.approx(wins / 6)
    assert result.p_value == pytest.approx(p_value)
    assert result.ci_low < result.ci_high
    assert result.ci_low <= result.difference <= result.ci_high


def test_equal_observed_reach_does_not_establish_equivalence():
    baseline = [run(seed=i, success=True) for i in range(6)]
    method = [run(seed=i, success=True) for i in range(6)]
    result = paired_joint_reach(baseline, method, 0.1)
    assert result.difference == 0
    assert result.p_value == 1
    assert result.ci_low < -0.5 and result.ci_high > 0.5


def test_paired_result_reverses_sign_when_arms_are_swapped():
    baseline = [run(seed=i, success=i < 2) for i in range(6)]
    method = [run(seed=i, success=i < 4) for i in range(6)]
    a = paired_joint_reach(baseline, method, 0.1)
    b = paired_joint_reach(method, baseline, 0.1)
    assert a.difference == -b.difference
    assert a.p_value == b.p_value
    assert a.ci_low == pytest.approx(-b.ci_high)
    assert a.ci_high == pytest.approx(-b.ci_low)


def test_incomplete_seed_pairs_are_not_silently_dropped():
    with pytest.raises(ValueError, match="same nonempty seed set"):
        paired_joint_reach([run(seed=0)], [run(seed=1)], 0.1)
    with pytest.raises(ValueError, match="unique seed"):
        paired_joint_reach([run(seed=0), run(seed=0)], [run(seed=0)], 0.1)
    with pytest.raises(ValueError, match="fixed shot budget"):
        paired_joint_reach([run(n=6)], [run(n=5)], 0.1)


def test_equal_censoring_still_makes_survivor_ratio_conditional():
    baseline = ThresholdResult([100, 110, 120], [150] * 3, 0.1)
    method = ThresholdResult([10, 11, 12], [150] * 3, 0.1)
    result = speedup_from_results(baseline, method)
    assert not result.trustworthy
    assert any("conditions on reaching seeds" in w for w in result.warnings)
