"""Regressions for the external audit of 2026-09-16.

Each test is the audit's own counterexample, reproduced against this checkout
before the fix and pinned after it. Where an existing test asserted the defect,
it has been corrected rather than deleted -- a test can encode a bug as firmly
as the code does, and `test_two_points_fall_back_to_the_weighted_mean` did.

Audit: C:/Users/mothe/hfmarl-fusion-codex/AUDIT.md, against commit 967d120.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.devices.registry import encoded_states, get as get_device
from hfmarl.envs.task import get as get_task
from hfmarl.federation.robust import centered_clip
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import ClientUpdate
from hfmarl.metrics.curves import (
    ThresholdResult,
    shots_to_competence_evaluated,
    speedup_from_results,
    trailing_mean,
)
from hfmarl.metrics.log import RunLog, ShotRecord

STATES = encoded_states()


def upd(device, value, violation_rate=0.0):
    return ClientUpdate(device=device, cluster="thermal",
                        weights={"flat": np.full(3, float(value))},
                        state=STATES[device], n_samples=5, round_produced=1,
                        violation_rate=violation_rate)


# -- #1 per-device tolerance ---------------------------------------------


@pytest.mark.parametrize("task_name", ["easy", "moderate"])
def test_tolerance_is_per_device_and_wildly_so(task_name):
    """Not a fix so much as the fact that makes #1 severe: the two default
    joiners differ by a factor of 24, so scoring both with one of them is not
    an approximation, it is a different experiment."""
    task = get_task(task_name)
    tol = {d: float(task.resolve_for(get_device(d)).tolerance)
           for d in ("sparc_like", "tcv_like")}
    assert tol["tcv_like"] / tol["sparc_like"] > 20


# -- #3 rejected clients must not reach the merge -------------------------


def test_a_rejected_peer_cannot_decide_the_coordinate_median():
    """The audit's sharpest one. Weights were [1, 0, 0] and the coordinate
    median returned the REJECTED peer's value, because a rank statistic does
    not read weights at all. Zeroing a weight is not removing a client."""
    s = FedBuffServer(bandwidth=1.0, align=False, clip_factor=None,
                      rule="median")
    s.publish(upd("diiid_like", 0.0))
    s.publish(upd("iter_like", 100.0, violation_rate=0.9))
    s.publish(upd("sparc_like", 200.0, violation_rate=0.9))
    agg = s.aggregate_for("tcv_like", STATES["tcv_like"], "thermal", 1)
    assert np.allclose(agg, 0.0), "only the admissible peer may contribute"


def test_a_rejected_outlier_cannot_change_the_clipping_radius():
    """The radius is a median over the rows it is given, so an inadmissible
    outlier could relax the bound protecting the admissible ones."""
    common = dict(align=False, rule="mean", clip_factor=2.0)
    ref = np.zeros(3)

    clean = FedBuffServer(bandwidth=1.0, **common)
    for d, v in (("diiid_like", 100.0), ("iter_like", 1.0), ("sparc_like", 1.0)):
        clean.publish(upd(d, v))
    a = clean.aggregate_for("tcv_like", STATES["tcv_like"], "thermal", 1,
                            reference=ref)

    polluted = FedBuffServer(bandwidth=1.0, **common)
    for d, v in (("diiid_like", 100.0), ("iter_like", 1.0), ("sparc_like", 1.0)):
        polluted.publish(upd(d, v))
    polluted.publish(upd("tcv_like", 1e6, violation_rate=0.99))
    b = polluted.aggregate_for("tcv_like", STATES["tcv_like"], "thermal", 1,
                               reference=ref)

    assert np.allclose(a, b), "a rejected peer changed the admissible result"


def test_zero_median_deviation_still_bounds_an_outlier():
    """#9. Over half the rows sitting exactly on the reference is routine with
    a common initialisation, and it used to switch clipping off entirely."""
    clipped, n = centered_clip(np.array([[0.0], [0.0], [1e9]]), np.zeros(1),
                               factor=2.0)
    assert n == 1
    assert clipped[2, 0] < 1e9


# -- #5 competence is confirmed, not backdated ---------------------------


def test_competence_is_reported_at_the_shot_that_confirms_it():
    """Two good evaluations at shots 1 and 11 with persistence 2 reported 1 --
    backdating the answer past the ten shots that earned it, which inflates
    every early ratio."""
    r = RunLog("x", "iter_like", 0)
    for i in range(30):
        is_eval = i % 10 == 0
        r.add(ShotRecord(shot=i, reward=-1.0, steps=20,
                         beta_error=0.001 if is_eval else 9.0,
                         is_evaluation=is_eval))
    res = shots_to_competence_evaluated([r], 0.05, window=1, persistence=2)
    assert res.values == [11.0]


# -- #6 a failed shot must leave the window ------------------------------


def test_one_nonfinite_shot_does_not_censor_the_rest_of_the_run():
    """`fire_shot` records NaN when a solver-failed shot has no usable steps.
    The cumulative-sum form propagated that to the end of the series, so one
    failed evaluation made a subsequently working controller look permanently
    censored."""
    got = trailing_mean(np.array([np.nan, 1.0, 1.0, 1.0]), 2)
    assert np.isnan(got[0])
    assert np.allclose(got[1:], 1.0)


def test_trailing_mean_is_unchanged_on_finite_input():
    """The repair must not move any number that was already right."""
    assert np.allclose(trailing_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2),
                       [1.0, 1.5, 2.5, 3.5])


def test_a_window_of_only_failures_is_nan_not_zero():
    got = trailing_mean(np.array([np.nan, np.nan, 2.0]), 2)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2] == pytest.approx(2.0)


# -- #7 a ratio from one seed per arm is not trustworthy -----------------


def test_severe_equal_censoring_is_not_reported_as_trustworthy():
    """Equal censoring passes the reach-rate check, and the seed check counted
    seeds that RAN rather than seeds that contributed. One success against four
    censored runs per arm returned 10.00x [10.00, 10.00] trustworthy=True."""
    base = ThresholdResult(values=[100.0], censored_at=[200] * 4, threshold=0.0)
    meth = ThresholdResult(values=[10.0], censored_at=[200] * 4, threshold=0.0)
    sp = speedup_from_results(base, meth)
    assert not sp.trustworthy
    assert any("reaching seed" in w for w in sp.warnings)


def test_a_well_populated_comparison_is_still_trustworthy():
    """The guard must not fire on the case it is meant to permit."""
    base = ThresholdResult(values=[100.0, 110.0, 90.0, 105.0],
                           censored_at=[], threshold=0.0)
    meth = ThresholdResult(values=[50.0, 55.0, 45.0, 52.0],
                           censored_at=[], threshold=0.0)
    assert speedup_from_results(base, meth).trustworthy


# -- self-audit of the post-audit code -----------------------------------
#
# The external audit reviewed commit 967d120. Everything below was written
# after it, mostly in a hurry, and these are the defects a self-audit found.
# The external audit's own base rate was the warning: nearly every bug it
# reported was mine, written that same day.


def test_a_failed_solve_does_not_poison_the_operating_region():
    """SELF-AUDIT. `state_from_env` encodes a failed solve to NaN. One of those
    made the region mean and variance NaN, `overlap_weight` return NaN, every
    weight NaN, and the server -- reading `any(w > 0)` as False -- return no
    aggregate. Federation switched itself off for the round and said nothing,
    so the experiment would report that federating did not help when
    federating had not happened. Same class as AUDIT #6."""
    from hfmarl.federation.similarity import overlap_weight, region_from_states
    from hfmarl.physics.dimensionless import DimensionlessState

    good = DimensionlessState(3e-3, 0.05, 1.5, 4.0, 0.0)
    bad = DimensionlessState(float("nan"), 0.05, 1.5, 4.0, 0.0)

    region = region_from_states([good, good, bad])
    assert np.isfinite(region.mean).all()
    assert np.isfinite(region.var).all()
    assert region.n == 2, "the unusable state must not be counted either"
    assert np.isfinite(overlap_weight(region, region_from_states([good]), 1.0))


def test_a_device_with_no_usable_solve_raises_rather_than_guessing():
    """The caller must choose between a nominal fallback and skipping the
    round, and that choice must not be made silently inside the summary."""
    from hfmarl.federation.similarity import region_from_states
    from hfmarl.physics.dimensionless import DimensionlessState

    bad = DimensionlessState(float("nan"), float("nan"), 1.5, 4.0, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        region_from_states([bad, bad])


def test_non_finite_weights_are_distinguished_from_rejection():
    """A numerical fault used to be indistinguishable from 'every peer was
    rejected': both produced no aggregate, and only one of them is a result."""
    from hfmarl.physics.dimensionless import DimensionlessState

    s = FedBuffServer(bandwidth=1.0, align=False, clip_factor=None)
    nan_state = DimensionlessState(float("nan"), 0.05, 1.5, 4.0, 0.0)
    for d in ("diiid_like", "iter_like"):
        s.publish(ClientUpdate(device=d, cluster="thermal",
                               weights={"flat": np.zeros(3)}, state=nan_state,
                               n_samples=5, round_produced=1))
    assert s.aggregate_for("tcv_like", STATES["tcv_like"], "thermal", 1) is None
    assert s.last_rejected_all is False, "not a rejection -- a fault"
    assert "non-finite" in s.last_error


def test_the_handover_shot_is_marked_as_an_evaluation():
    """AUDIT #2, and a regression of it.

    The shot that fires the inherited controller is the only observation of it
    before any local learning -- the zero-adaptation assessment. Unmarked, it
    is excluded from every evaluation-based metric, so the handover gets
    measured only after the joiner has started changing it.

    The external audit caught this once and I fixed it. My own commissioning-
    shots patch then rewrote the same block and dropped the flag again, and
    nothing noticed, because #2 was the one finding I never wrote a test for.
    This is that test: it reads the source rather than running TORAX, because
    the defect is a missing argument at a specific call site and a behavioural
    test would need a full cold-start run to reach it.
    """
    import inspect

    from hfmarl.experiments import runner

    src = inspect.getsource(runner.run_cold_start)
    marker = "AUDIT #2"
    assert marker in src, "the reasoning for this call site was removed"

    after = src.split(marker, 1)[1]
    call = after.split("fire_shot(", 1)[1].split(")", 1)[0]
    assert "evaluation=True" in call, (
        "the handover shot is no longer marked as an evaluation; it will be "
        "excluded from every evaluation-based metric again")


def test_no_experiment_script_scores_every_device_with_one_tolerance():
    """AUDIT #1, generalised, after it recurred.

    The audit found `exp_coldstart.py` resolving the tolerance once from the
    first joiner and scoring every joiner against it. I fixed that file and
    stopped. `exp_federation.py` had the identical line, was never reviewed,
    and produced a 300-shot two-seed result scoring sparc_like against
    diiid_like's tolerance -- 26x looser than the task it trained on.

    Tolerance is a fraction of each device's own measured band, so resolving it
    from `device_names[0]` or `joiners[0]` is always wrong when more than one
    device is scored. This greps for the shape rather than the file, so a third
    script cannot reintroduce it quietly.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    pattern = re.compile(
        r"resolve_for\(\s*get_device\(\s*\w+\[0\]\s*\)\s*\)", re.S)
    for script in sorted((root / "scripts").glob("*.py")):
        if pattern.search(script.read_text(encoding="utf-8")):
            offenders.append(script.name)
    assert not offenders, (
        f"{offenders} resolve one device's tolerance and apply it to all: "
        "score each device against its own criterion")
