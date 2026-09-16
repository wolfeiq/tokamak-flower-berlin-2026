"""Reject-if-worse must compare one draw against one draw.

`HillClimber.best_return` is a running MAXIMUM over noisy returns, so it is an
order statistic that drifts upward with the shot count whatever the controller
does. Using it as the acceptance bar means an incoming aggregate has to beat
the luckiest draw the incumbent ever had, on a single draw of its own.

Simulated with pure noise and two equally good models, adoption falls from 50%
after one shot to 1% after a hundred; with an aggregate genuinely one noise-sd
better, from 76% to 8%. So `--select` switched federation off as training
proceeded, for a reason with nothing to do with the models -- which is the
shape of the null result the folds kept producing.
"""
import numpy as np
import pytest

from hfmarl.agents.search import HillClimber


def climber(n=4):
    c = HillClimber(n_params=n, seed=0)
    c.seed_from(np.zeros(n))
    return c


def test_a_fresh_incumbent_has_no_bar():
    """And the caller must fire one rather than reach for best_return."""
    c = climber()
    c.observe(np.ones(4), 10.0)
    bar, source = c.acceptance_bar()
    assert source == "none"
    assert np.isnan(bar)


def test_an_evaluation_becomes_the_bar():
    c = climber()
    c.observe(np.ones(4), 10.0)
    c.note_evaluation(3.0)
    bar, source = c.acceptance_bar()
    assert source == "evaluation"
    assert bar == 3.0


def test_the_bar_is_the_evaluation_not_the_running_maximum():
    """The whole point. best_return is 10 because one lucky candidate scored
    10; the incumbent's unperturbed performance is 3, and 3 is what an
    aggregate has to beat."""
    c = climber()
    for r in (1.0, 10.0, 2.0):
        c.observe(np.ones(4) * r, r)
    c.note_evaluation(3.0)
    assert c.best_return == 10.0
    assert c.acceptance_bar()[0] == 3.0


def test_a_new_incumbent_invalidates_the_bar():
    """An evaluation measures the controller that was installed at the time;
    carrying it across a change would judge a new model by an old draw."""
    c = climber()
    c.observe(np.ones(4), 1.0)
    c.note_evaluation(3.0)
    c.observe(np.full(4, 2.0), 5.0)          # a better candidate takes over
    assert c.acceptance_bar()[1] == "none"


def test_adopting_sets_the_bar_from_the_adoption_shot():
    """The adoption shot IS an unperturbed evaluation of what was installed,
    so the next decision has an unbiased bar without firing anything."""
    c = climber()
    c.observe(np.ones(4), 1.0)
    c.adopt(np.full(4, 7.0), 4.5)
    assert c.acceptance_bar() == (4.5, "evaluation")


def test_the_max_of_n_bar_drifts_and_a_single_draw_does_not():
    """The bias, as arithmetic rather than assertion.

    Two equally good models. Against a best-of-N bar, adoption collapses as N
    grows; against one fresh draw it stays at a half.
    """
    rng = np.random.default_rng(0)
    trials = 20000
    biased = {}
    for n in (1, 5, 100):
        bar = rng.normal(0.0, 1.0, size=(trials, n)).max(axis=1)
        agg = rng.normal(0.0, 1.0, size=trials)
        biased[n] = float((agg > bar).mean())
    assert biased[1] == pytest.approx(0.5, abs=0.02)
    assert biased[5] < 0.25
    assert biased[100] < 0.03

    fair = float((rng.normal(0.0, 1.0, trials)
                  > rng.normal(0.0, 1.0, trials)).mean())
    assert fair == pytest.approx(0.5, abs=0.02)


def test_the_runner_no_longer_compares_against_best_return():
    """Source-read, because the call sites are what actually decide."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "hfmarl" / "experiments"
           / "runner.py").read_text(encoding="utf-8")
    assert "total < climbers[name].best_return" not in src
    assert "total < climber.best_return" not in src
    # run_condition asks twice (before and after buying a bar), the
    # handover twice for the same reason, run_catastrophe once.
    assert src.count("acceptance_bar()") == 5
    # five sites record an evaluation into the bar: the routine evaluation
    # cadence in run_condition and in the joiner loop, and the three places
    # that buy one outright when no current estimate exists.
    assert src.count("note_evaluation(") == 5
    assert "climber.note_evaluation(" in src


def test_the_shots_selection_costs_are_counted_and_reported():
    """Selection is not free, and it is not paid evenly across the ladder.

    An arm that exchanges must measure its incumbent before it can refuse an
    aggregate; an arm that never exchanges spends nothing. `scratch` has no
    handover, so it gets one more training shot out of the joiner's 120 than
    every inheriting arm -- under a percent, and exactly the sort of
    asymmetry that becomes a mystery if it is never written down.
    """
    from pathlib import Path

    runner = (Path(__file__).resolve().parents[1] / "hfmarl" / "experiments"
              / "runner.py").read_text(encoding="utf-8")
    assert '"acceptance_shots": acceptance_shots' in runner
    assert '"joiner_acceptance_shots": int(joiner_acceptance_shots)' in runner
    # every bought shot is counted, and every counted shot is logged
    assert runner.count("acceptance_shots += 1") == 2

    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "exp_coldstart.py").read_text(encoding="utf-8")
    assert "measuring the bar for reject-if-worse" in script
