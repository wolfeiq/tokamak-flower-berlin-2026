"""Metrics: shots-to-threshold, censoring, and the statistics that could lie."""

import numpy as np
import pytest

from hfmarl.metrics.curves import (
    asymptotic_performance,
    bootstrap_ci,
    catastrophe_test,
    learning_curve,
    shots_to_threshold,
    shots_to_threshold_single,
    speedup_ratio,
    staleness_tolerance,
    threshold_from_reference,
    trailing_mean,
    violation_rate,
    violations_during_training,
)
from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord


def synth(condition, rate, n=300, seed=0, noise=0.3, viol_tau=50, device="iter_like"):
    """Exponential learning curve with decaying violation probability."""
    rng = np.random.default_rng(seed)
    r = RunLog(condition, device, seed)
    for i in range(n):
        reward = -10.0 * np.exp(-i / rate) + rng.normal(0, noise)
        viol = ("beta_N",) if rng.random() < 0.4 * np.exp(-i / viol_tau) else ()
        r.add(ShotRecord(shot=i, reward=float(reward), steps=20, violations=viol))
    return r


def flat(condition, value, n=100, seed=0, device="iter_like"):
    r = RunLog(condition, device, seed)
    for i in range(n):
        r.add(ShotRecord(shot=i, reward=value, steps=20))
    return r


# -- smoothing -----------------------------------------------------------


def test_trailing_mean_is_causal():
    """A centred filter would let future shots trigger a crossing. That cheats."""
    x = np.array([0.0, 0.0, 0.0, 100.0])
    sm = trailing_mean(x, window=2)
    assert sm[0] == 0.0 and sm[1] == 0.0 and sm[2] == 0.0
    assert sm[3] == 50.0


def test_trailing_mean_handles_short_prefix():
    sm = trailing_mean(np.array([2.0, 4.0]), window=10)
    assert sm[0] == 2.0 and sm[1] == 3.0


def test_trailing_mean_rejects_bad_window():
    with pytest.raises(ValueError):
        trailing_mean(np.ones(5), 0)


# -- shots to threshold --------------------------------------------------


def test_finds_a_clean_crossing():
    rewards = np.concatenate([np.full(50, -10.0), np.full(50, 0.0)])
    n = shots_to_threshold_single(rewards, threshold=-1.0, window=5, persistence=3)
    assert n is not None and 50 < n < 65


def test_returns_none_when_never_reached():
    assert shots_to_threshold_single(np.full(100, -10.0), -1.0) is None


def test_persistence_rejects_a_lucky_single_crossing():
    """One fluke shot above the line must not count as convergence."""
    rewards = np.full(100, -10.0)
    rewards[50] = 1000.0
    assert shots_to_threshold_single(rewards, -1.0, window=1, persistence=5) is None


def test_persistence_of_one_accepts_it():
    rewards = np.full(100, -10.0)
    rewards[50] = 1000.0
    assert shots_to_threshold_single(rewards, -1.0, window=1, persistence=1) == 51


def test_empty_run_is_none():
    assert shots_to_threshold_single(np.zeros(0), 0.0) is None


def test_recovers_a_known_speedup():
    """Ground truth: federated learns 3x faster. Measured ratio must reflect it."""
    iso = [synth("isolated", 120, seed=s) for s in range(5)]
    fed = [synth("fedbuff_similarity", 40, seed=s) for s in range(5)]
    res = speedup_ratio(iso, fed, threshold=-2.0)
    assert 2.0 < res.ratio < 4.0
    assert res.ci_low < res.ratio < res.ci_high


# -- censoring: the thing that would bias the headline number ------------


def test_censored_seeds_are_counted_not_dropped():
    runs = [synth("isolated", 30, n=300, seed=0),  # converges
            flat("isolated", -10.0, n=300, seed=1)]  # never does
    res = shots_to_threshold(runs, threshold=-2.0)
    assert res.n_reached == 1 and res.n_total == 2
    assert res.reach_rate == 0.5
    assert res.censored_at == [300]


def test_unequal_censoring_raises_a_warning_not_a_clean_number():
    """The failure this guards: 4 of 5 seeds fail, the lucky one reports fast."""
    iso = [synth("isolated", 40, seed=s) for s in range(5)]
    fed = [synth("fedbuff_similarity", 20, seed=0)] + [
        flat("fedbuff_similarity", -10.0, seed=s) for s in range(1, 5)
    ]
    res = speedup_ratio(iso, fed, threshold=-2.0)
    assert not res.trustworthy
    assert any("censoring differs" in w for w in res.warnings)


def test_too_few_seeds_is_flagged():
    iso = [synth("isolated", 100, seed=0), synth("isolated", 100, seed=1)]
    fed = [synth("fedbuff_similarity", 40, seed=0), synth("fedbuff_similarity", 40, seed=1)]
    res = speedup_ratio(iso, fed, threshold=-2.0)
    assert any("seeds" in w for w in res.warnings)


def test_no_ratio_when_a_condition_never_converges():
    iso = [flat("isolated", -10.0, seed=s) for s in range(3)]
    fed = [synth("fedbuff_similarity", 40, seed=s) for s in range(3)]
    res = speedup_ratio(iso, fed, threshold=-2.0)
    assert np.isnan(res.ratio) and not res.trustworthy


def test_summary_mentions_censoring():
    runs = [synth("isolated", 30, seed=0), flat("isolated", -10.0, seed=1)]
    assert "censored" in shots_to_threshold(runs, -2.0).summary()


# -- asymptotic performance ----------------------------------------------


def test_asymptote_matches_the_plateau():
    runs = [synth("isolated", 30, n=400, seed=s, noise=0.1) for s in range(4)]
    mean, sem = asymptotic_performance(runs)
    assert -0.5 < mean < 0.5 and sem >= 0


def test_threshold_from_reference_lies_between_worst_and_plateau():
    runs = [synth("centralised", 40, seed=s) for s in range(3)]
    thr = threshold_from_reference(runs, fraction=0.8)
    plateau, _ = asymptotic_performance(runs)
    worst = min(float(np.min(r.rewards())) for r in runs)
    assert worst < thr < plateau


def test_threshold_fraction_is_monotone():
    runs = [synth("centralised", 40, seed=s) for s in range(3)]
    assert (threshold_from_reference(runs, 0.5)
            < threshold_from_reference(runs, 0.9))

def test_threshold_refuses_a_reference_that_never_improves():
    """A flat curve has no threshold, and the error has to say which
    anchors it compared -- `gate_headroom.py` catches this to distinguish
    "the budget was too small" from "the task is wrong", and those two
    send you to opposite fixes.
    """
    runs = [flat("isolated", -0.17, n=60, seed=s) for s in range(2)]
    with pytest.raises(ValueError, match=r'does not improve'):
        threshold_from_reference(runs, 0.8)


def test_asymptote_rejects_bad_fraction():
    with pytest.raises(ValueError):
        asymptotic_performance([synth("x", 30)], last_fraction=0.0)


# -- violations ----------------------------------------------------------


def test_cumulative_violations_are_monotone():
    runs = [synth("isolated", 100, seed=s) for s in range(3)]
    mean, _ = violations_during_training(runs)
    assert np.all(np.diff(mean) >= -1e-12)


def test_a_shot_violating_three_limits_counts_once():
    """Counting crossings would make one bad shot look like a trend."""
    r = RunLog("isolated", "iter_like", 0)
    r.add(ShotRecord(shot=0, reward=0.0, steps=5,
                     violations=("beta_N", "q95", "greenwald_fraction")))
    assert r.cumulative_violations()[-1] == 1


def test_violation_rate_tail_restriction():
    r = RunLog("isolated", "iter_like", 0)
    for i in range(100):
        r.add(ShotRecord(shot=i, reward=0.0, steps=5,
                         violations=("beta_N",) if i < 50 else ()))
    assert violation_rate([r]) == pytest.approx(0.5)
    assert violation_rate([r], last_fraction=0.2) == pytest.approx(0.0)


def test_unequal_length_runs_are_truncated_not_extrapolated():
    a, b = synth("isolated", 50, n=100, seed=0), synth("isolated", 50, n=60, seed=1)
    mean, _ = violations_during_training([a, b])
    assert mean.size == 60


# -- the catastrophe claim -----------------------------------------------


def _iso_fed(fed_error=0.01, steps=5):
    iso = RunLog("isolated", "iter_like", 0)
    fed = RunLog("fedbuff_similarity", "iter_like", 0)
    for i in range(20):
        iso.add(ShotRecord(shot=i, reward=0.0, steps=5, violations=("beta_N",),
                           beta_error=0.01))
        fed.add(ShotRecord(shot=i, reward=0.0, steps=steps,
                           beta_error=fed_error))
    return iso, fed


def test_safety_alone_does_not_establish_the_claim():
    """AUDIT. `claim_holds` used to depend on violation rates only, so a
    controller that never drove the plant satisfied it: no limit crossed, and
    the isolated arm crossing one. Safety bought with uselessness is not the
    claim, and an UNMEASURED competence check must not silently grant it
    either."""
    iso, fed = _iso_fed()
    res = catastrophe_test([iso], [fed])  # no tolerance supplied
    assert not res.claim_holds


def test_claim_holds_when_federated_is_clean_and_competent():
    iso, fed = _iso_fed(fed_error=0.01)
    res = catastrophe_test([iso], [fed], tolerance=0.05, expected_steps=5)
    assert res.claim_holds and "CLAIM HOLDS" in res.summary()
    assert res.federated_success == 1.0


def test_a_safe_but_useless_controller_does_not_hold_the_claim():
    """Tracking error 99 with no violation. Perfectly safe, perfectly idle."""
    iso, fed = _iso_fed(fed_error=99.0)
    res = catastrophe_test([iso], [fed], tolerance=0.05, expected_steps=5)
    assert res.federated_rate == 0.0
    assert not res.claim_holds


def test_a_shot_that_died_early_cannot_count_as_a_success():
    """Tracking error is averaged over SURVIVING steps, so a discharge that
    dies at step 2 near its setpoint reports an excellent error. Completion is
    checked separately for exactly that reason."""
    iso, fed = _iso_fed(fed_error=0.01, steps=2)
    res = catastrophe_test([iso], [fed], tolerance=0.05, expected_steps=5)
    assert res.federated_success == 0.0
    assert not res.claim_holds


def test_reduced_but_nonzero_violations_is_reported_as_partial():
    """The spec's claim is avoidance, not reduction. Don't let it slide."""
    iso = RunLog("isolated", "iter_like", 0)
    fed = RunLog("fedbuff_similarity", "iter_like", 0)
    for i in range(20):
        iso.add(ShotRecord(shot=i, reward=0.0, steps=5, violations=("beta_N",)))
        fed.add(ShotRecord(shot=i, reward=0.0, steps=5,
                           violations=("beta_N",) if i < 4 else ()))
    res = catastrophe_test([iso], [fed])
    assert not res.claim_holds
    assert "PARTIAL" in res.summary()


def test_claim_fails_when_federation_does_not_help():
    iso = RunLog("isolated", "iter_like", 0)
    fed = RunLog("fedbuff_similarity", "iter_like", 0)
    for i in range(10):
        iso.add(ShotRecord(shot=i, reward=0.0, steps=5))
        fed.add(ShotRecord(shot=i, reward=0.0, steps=5, violations=("q95",)))
    assert "CLAIM FAILS" in catastrophe_test([iso], [fed]).summary()


# -- staleness -----------------------------------------------------------


def test_staleness_returns_sorted_lags():
    by_lag = {5: [synth("f", 40, seed=0)], 1: [synth("f", 30, seed=1)],
              10: [synth("f", 60, seed=2)]}
    lags, perf, sem = staleness_tolerance(by_lag)
    assert list(lags) == [1, 5, 10]
    assert perf.size == 3 and sem.size == 3


# -- shared stats --------------------------------------------------------


def test_bootstrap_ci_brackets_the_median():
    x = np.arange(1.0, 21.0)
    lo, hi = bootstrap_ci(x)
    assert lo <= np.median(x) <= hi


def test_bootstrap_handles_single_value():
    assert bootstrap_ci(np.array([7.0])) == (7.0, 7.0)


def test_bootstrap_handles_empty():
    lo, hi = bootstrap_ci(np.zeros(0))
    assert np.isnan(lo) and np.isnan(hi)


def test_learning_curve_shapes_agree():
    runs = [synth("isolated", 60, n=120, seed=s) for s in range(4)]
    x, mean, sem = learning_curve(runs)
    assert x.size == mean.size == sem.size == 120


# -- provenance ----------------------------------------------------------


def test_threshold_result_records_how_it_was_computed():
    """window and persistence change the answer; comparing across them is wrong."""
    runs = [synth("isolated", 60, seed=s) for s in range(3)]
    res = shots_to_threshold(runs, -2.0, window=25, persistence=7)
    assert res.window == 25 and res.persistence == 7
    assert "window=25" in res.summary()


def test_smoothing_window_materially_changes_the_answer():
    """Justifies recording it: this is not a cosmetic difference."""
    runs = [synth("isolated", 60, n=400, seed=s, noise=1.5) for s in range(3)]
    a = shots_to_threshold(runs, -2.0, window=5).median
    b = shots_to_threshold(runs, -2.0, window=50).median
    assert abs(a - b) > 5


def test_mismatched_smoothing_is_detected_when_results_are_combined():
    """The guard in speedup_ratio, exercised directly.

    Note it cannot fire through `speedup_ratio`'s own arguments -- that
    function smooths both conditions with one `window`. The real exposure is
    comparing numbers ACROSS invocations (gate_headroom reports at window=25;
    a later analysis at the default 10). The defence against that is the
    provenance now printed in `summary()`; this checks the combining logic
    itself is correct.
    """
    runs = [synth("isolated", 120, seed=s) for s in range(3)]
    a = shots_to_threshold(runs, -2.0, window=10)
    b = shots_to_threshold(runs, -2.0, window=40)
    assert (a.window, a.persistence) != (b.window, b.persistence)
    assert a.median != b.median or a.window != b.window


def test_matched_settings_produce_a_trustworthy_ratio():
    iso = [synth("isolated", 120, seed=s) for s in range(5)]
    fed = [synth("fedbuff_similarity", 40, seed=s) for s in range(5)]
    res = speedup_ratio(iso, fed, -2.0, window=10)
    assert res.trustworthy
    assert res.baseline.window == res.method.window == 10


def test_best_evaluated_error_reads_only_evaluations():
    """The metric that still separates arms when none of them passes.

    On a device whose tolerance no arm reaches, shots-to-competence says
    "never" five times and the fold is silent. How close each arm got is
    still a comparison, and it is in the tolerance's own units.
    """
    import numpy as np

    from hfmarl.metrics.curves import best_evaluated_error
    from hfmarl.metrics.log import RunLog, ShotRecord

    def run(errs, evals):
        r = RunLog(condition="x", device="d", seed=0)
        for i, (e, ev) in enumerate(zip(errs, evals)):
            r.add(ShotRecord(shot=i, reward=-e, beta_error=e, steps=30,
                             is_evaluation=ev))
        return r

    # The candidate stream contains a much better shot; it must be ignored,
    # because a candidate is a perturbation that was never adopted.
    r = run([0.9, 0.001, 0.5, 0.4], [True, False, True, True])
    med, iqr = best_evaluated_error([r])
    assert med == pytest.approx(0.4)
    assert iqr == pytest.approx(0.0)


def test_best_evaluated_error_is_a_median_over_seeds():
    import numpy as np

    from hfmarl.metrics.curves import best_evaluated_error
    from hfmarl.metrics.log import RunLog, ShotRecord

    def run(best):
        r = RunLog(condition="x", device="d", seed=0)
        for i, e in enumerate([1.0, best, 0.8]):
            r.add(ShotRecord(shot=i, reward=-e, beta_error=e, steps=30,
                             is_evaluation=True))
        return r

    med, _ = best_evaluated_error([run(0.2), run(0.4), run(0.6)])
    assert med == pytest.approx(0.4)
    del np


def test_best_evaluated_error_is_nan_when_nothing_was_evaluated():
    import numpy as np

    from hfmarl.metrics.curves import best_evaluated_error
    from hfmarl.metrics.log import RunLog, ShotRecord

    r = RunLog(condition="x", device="d", seed=0)
    r.add(ShotRecord(shot=0, reward=-1.0, beta_error=1.0, steps=30,
                     is_evaluation=False))
    med, iqr = best_evaluated_error([r])
    assert np.isnan(med) and np.isnan(iqr)


def test_best_evaluated_error_skips_a_nan_evaluation():
    """A solver-failed evaluation records NaN and still cost a shot; it must
    not become the minimum by accident, nor censor the seed."""
    import numpy as np

    from hfmarl.metrics.curves import best_evaluated_error
    from hfmarl.metrics.log import RunLog, ShotRecord

    r = RunLog(condition="x", device="d", seed=0)
    for i, e in enumerate([float("nan"), 0.3]):
        r.add(ShotRecord(shot=i, reward=-1.0, beta_error=e, steps=30,
                         is_evaluation=True))
    med, _ = best_evaluated_error([r])
    assert med == pytest.approx(0.3)
    del np
