"""Competence measured on evaluation shots, not on the exploration schedule.

The bug these pin: a controller that already tracks perfectly keeps firing
deliberate perturbations, so the candidate stream still logs errors and the run
reports NEVER REACHED while its incumbent is essentially exact. That happened
in a real cold-start run -- an arm plateaued at -0.001 reward and was recorded
as never having reached competence in the same table.
"""

from __future__ import annotations

import numpy as np

from hfmarl.metrics.curves import (
    shots_to_competence,
    shots_to_competence_evaluated,
)
from hfmarl.metrics.log import RunLog, ShotRecord

TOL = 0.05


def run_with(errors, eval_every=10, eval_error=None):
    """A run whose candidates carry `errors` and whose evaluations are clean."""
    r = RunLog("test", "iter_like", 0)
    for i, e in enumerate(errors):
        is_eval = eval_every and i % eval_every == 0
        r.add(ShotRecord(
            shot=i, reward=-float(e), steps=20,
            beta_error=float(eval_error if (is_eval and eval_error is not None)
                             else e),
            is_evaluation=bool(is_eval),
        ))
    return r


def test_a_working_controller_is_not_hidden_by_its_own_exploration():
    """The exact failure. Candidates sit outside tolerance because the search
    keeps probing; the incumbent is inside it the whole time."""
    n = 120
    candidates = np.full(n, TOL * 3)      # probing, always out of band
    run = run_with(candidates, eval_every=10, eval_error=TOL * 0.2)

    on_candidates = shots_to_competence([run], TOL, window=25)
    on_evaluations = shots_to_competence_evaluated([run], TOL)

    assert on_candidates.n_reached == 0, "candidate stream should look like failure"
    assert on_evaluations.n_reached == 1, "the incumbent was always competent"


def test_the_answer_is_in_total_shots_not_in_evaluations_performed():
    """A machine evaluating every tenth shot has still fired all ten. Reporting
    the count of evaluations would divide every number by ten."""
    n = 100
    run = run_with(np.full(n, TOL * 3), eval_every=10, eval_error=TOL * 0.1)
    res = shots_to_competence_evaluated([run], TOL, window=1, persistence=1)
    # First evaluation is shot index 0, so competence is confirmed at shot 1.
    assert res.values == [1.0]

    late = RunLog("late", "iter_like", 0)
    for i in range(n):
        is_eval = i % 10 == 0
        err = TOL * 0.1 if (is_eval and i >= 50) else TOL * 3
        late.add(ShotRecord(shot=i, reward=-1.0, steps=20, beta_error=err,
                            is_evaluation=is_eval))
    res = shots_to_competence_evaluated([late], TOL, window=1, persistence=1)
    assert res.values == [51.0], "shot 50, reported 1-based as 51"


def test_a_run_with_no_evaluation_shots_is_censored_not_crashed():
    """Old logs have none. They must read as 'not measured', never as zero."""
    run = run_with(np.full(30, TOL * 0.1), eval_every=0)
    res = shots_to_competence_evaluated([run], TOL)
    assert res.n_reached == 0 and res.censored_at == [30]


def test_persistence_rejects_a_single_lucky_evaluation():
    n = 60
    run = RunLog("lucky", "iter_like", 0)
    for i in range(n):
        is_eval = i % 10 == 0
        # One good evaluation at shot 20, everything else out of band.
        err = TOL * 0.1 if i == 20 else TOL * 3
        run.add(ShotRecord(shot=i, reward=-1.0, steps=20, beta_error=err,
                           is_evaluation=is_eval))
    res = shots_to_competence_evaluated([run], TOL, window=1, persistence=2)
    assert res.n_reached == 0, "one good shot is not competence"


def test_evaluation_errors_returns_aligned_values_and_indices():
    run = run_with(np.arange(20) * 0.01, eval_every=5, eval_error=0.001)
    errs, idx = run.evaluation_errors()
    assert list(idx) == [0, 5, 10, 15]
    assert np.allclose(errs, 0.001)
