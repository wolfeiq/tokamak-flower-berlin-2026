"""One shot, and the local search that spends them.

Both the headroom gate and the federation runner need the same two primitives:
fire exactly one shot and log it, and take one hill-climbing step. They were
written twice once already; a federated run whose local search anneals
differently from the isolated baseline is not a comparison, it is two
experiments with one label.

WHY HILL CLIMBING AND NOT CEM
-----------------------------
CEM evaluates a whole population per iteration, so "shots consumed" advances in
blocks. Shots are the currency of every claim in METRICS.md, and a currency
that jumps in blocks of `population` cannot express "this condition needed 240
and that one needed 180". Hill climbing spends exactly one shot per step, so
the axis is exact.

It is a weak optimiser, and that is stated rather than hidden: see the bias
discussion in `hfmarl/experiments/headroom.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hfmarl.agents.cem import rollout
from hfmarl.metrics.log import ShotRecord


def fire_shot(env, policy, vector: np.ndarray | None, shot: int,
              max_steps: int | None = None,
              evaluation: bool = False) -> tuple[ShotRecord, float]:
    """Run one episode with `vector` loaded into `policy`. Returns the record
    and the episode return.

    Every shot fired anywhere in this project goes through here, so that "a
    shot" means one thing: one episode, one row in the log, whatever the reason
    it was fired. That matters for the federated conditions, where adopting an
    aggregate costs a real evaluation -- counting it is the difference between
    a speedup and an accounting trick.
    """
    if max_steps is None:
        max_steps = env.task.steps_per_shot + 2
    if vector is None:
        # A controller with no learned parameters -- the conventional
        # baseline of PROTOCOL.md 3. It still gets a per-shot reset, because
        # an integral term carried between discharges would let it learn
        # across shots, which is exactly what this arm must not do.
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
    else:
        policy.set_flat(vector)
    t0 = time.perf_counter()
    total, steps, viol = rollout(env, policy, max_steps)
    wall = time.perf_counter() - t0

    errs = [
        abs(s.scalars.get("beta_N", np.nan) - s.target)
        for s in env.trajectory
        if s.ok and "beta_N" in s.scalars
    ]
    record = ShotRecord(
        shot=shot,
        reward=float(total),
        steps=steps,
        violations=viol,
        terminated_early=steps < max_steps - 2,
        beta_error=float(np.nanmean(errs)) if errs else float("nan"),
        wall_seconds=wall,
        is_evaluation=evaluation,
    )
    return record, float(total)


@dataclass
class HillClimber:
    """(1+1) search with a self-annealing step size.

    The anneal is the whole algorithm: grow on success so a good direction is
    pursued, shrink on failure so the search settles. The floor matters --
    without it sigma collapses and the run stops exploring while still
    consuming shots; the ceiling stops a lucky streak from throwing the
    incumbent away.
    """

    n_params: int
    seed: int = 0
    sigma: float = 0.30
    sigma_min: float = 0.02
    sigma_max: float = 0.50
    grow: float = 1.05
    shrink: float = 0.99

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.best: np.ndarray | None = None
        self.best_return: float = -np.inf
        # THE ACCEPTANCE BAR, which is not `best_return`.
        #
        # `best_return` is a running MAXIMUM over noisy draws, so it is an
        # order statistic: it drifts upward with the number of shots fired
        # even when the controller has stopped improving. Comparing an
        # aggregate's single evaluation against it means the aggregate must
        # beat the luckiest draw the incumbent ever had.
        #
        # Simulated with pure noise and two equally good models, adoption
        # falls from 50% after one shot to 1% after a hundred; with an
        # aggregate genuinely one noise-sd better, from 76% to 8%. So
        # reject-if-worse switches federation off as training proceeds, for
        # a reason that has nothing to do with the models.
        #
        # This is the most recent UNPERTURBED evaluation of the incumbent --
        # one draw against one draw, no drift. NaN when the current
        # incumbent has never been evaluated, which the caller must handle
        # rather than falling back to the biased bar.
        self.last_eval_return: float = float("nan")

    def seed_from(self, vector: np.ndarray) -> None:
        """Set the starting point without claiming to know its return.

        Used for the common initialisation every federated client shares.
        """
        self.best = np.asarray(vector, float).copy()

    def propose(self) -> np.ndarray:
        """The next point to evaluate. The first call returns the incumbent
        itself, so its return is measured rather than assumed."""
        if self.best is None:
            raise RuntimeError("seed_from() must be called before propose()")
        if not np.isfinite(self.best_return):
            return self.best.copy()
        return self.best + self.rng.normal(0, self.sigma, self.best.size)

    def note_evaluation(self, total: float) -> None:
        """Record an unperturbed evaluation of the CURRENT incumbent.

        Only valid until the incumbent changes, which is why `observe` and
        `adopt` clear it.
        """
        self.last_eval_return = float(total)

    def acceptance_bar(self) -> tuple[float, str]:
        """What an incoming aggregate has to beat, and where that came from.

        Returns (bar, source). `source` is "evaluation" for an unbiased
        recent draw and "none" when there is no unbiased estimate -- in which
        case the caller must fire one rather than reach for `best_return`.
        """
        if np.isfinite(self.last_eval_return):
            return float(self.last_eval_return), "evaluation"
        return float("nan"), "none"

    def observe(self, candidate: np.ndarray, total: float) -> bool:
        """Fold one evaluated candidate in. Returns whether it was adopted."""
        if total > self.best_return:
            self.best_return = float(total)
            self.best = np.asarray(candidate, float).copy()
            # A new incumbent has never been evaluated unperturbed.
            self.last_eval_return = float("nan")
            self.sigma = min(self.sigma * self.grow, self.sigma_max)
            return True
        self.sigma = max(self.sigma * self.shrink, self.sigma_min)
        return False

    def adopt(self, vector: np.ndarray, total: float) -> None:
        """Install an externally supplied point as the incumbent, whatever its
        return.

        Unconditional on purpose. Adopting only when the aggregate happens to
        beat the local incumbent would make every federated condition a local
        search with an occasional free hint, and the comparison would no longer
        be about aggregation at all. The cost of that honesty is that a bad
        aggregate really does set a client back, which is a result rather than
        a bug.
        """
        self.best = np.asarray(vector, float).copy()
        self.best_return = float(total)
        # `total` here IS an unperturbed evaluation of the newly installed
        # incumbent -- the adoption shot -- so it is the bar for next time.
        self.last_eval_return = float(total)
