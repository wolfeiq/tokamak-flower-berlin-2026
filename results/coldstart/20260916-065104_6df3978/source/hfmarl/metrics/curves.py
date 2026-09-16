"""The six metrics, denominated in shots.

    1. shots to threshold          (primary -- reported as a ratio vs isolated)
    2. asymptotic performance      (federation must not cost final quality)
    3. limit violations in training(safe behaviour transfers before good does)
    4. cold-start shots            (strongest practical number)
    5. unseen-regime violation rate(the catastrophe claim)
    6. staleness tolerance         (justifies FedBuff)

CENSORING -- the thing that would quietly break the headline number
-------------------------------------------------------------------
A run that never reaches the threshold has no shots-to-threshold. It is
right-censored at however many shots it ran. The tempting fixes are both wrong:

  * dropping non-reachers biases the comparison toward whichever condition
    failed more often -- a condition where 4 of 5 seeds never converged would
    report the single lucky seed's fast time and look excellent;
  * substituting the max shot count invents data and compresses the ratio
    toward 1.

So `shots_to_threshold` returns a `ThresholdResult` that carries the censoring
explicitly, and `speedup_ratio` REFUSES to produce a clean number when the two
conditions are censored at materially different rates. It says so instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from hfmarl.metrics.log import RunLog, ShotRecord


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving mean that RECOVERS after a non-finite observation.

    AUDIT #6. The cumulative-sum form propagated a single NaN to the end of
    the series: `trailing_mean([nan,1,1,1], 2)` returned all-NaN, including
    windows that no longer contained the bad point. That is reachable --
    `fire_shot` records a NaN tracking error when a solver-failed shot has no
    usable steps -- so one failed evaluation could make a subsequently
    working controller look permanently censored.

    Non-finite entries are excluded from the window they fall in rather than
    dropped from the series: the shot still happened, still cost budget, and
    still occupies its slot. A window containing only non-finite values is
    NaN, because there is nothing to average, and that NaN clears as soon as
    the window moves past it.

    Causal by construction: window i covers [i-window+1, i]. A centred filter
    would let shots that have not happened yet trigger a crossing, which for
    a sample-efficiency claim is cheating.
    """
    a = np.asarray(x, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    good = np.isfinite(a)
    vals = np.where(good, a, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    ccnt = np.concatenate([[0.0], np.cumsum(good.astype(float))])
    idx = np.arange(a.size)
    lo = np.maximum(0, idx - window + 1)
    total = csum[idx + 1] - csum[lo]
    count = ccnt[idx + 1] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    return out


# ---------------------------------------------------------------------------
# 1. Shots to threshold
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Shots-to-threshold across seeds, with censoring kept visible."""

    values: list[float]  # shots for seeds that reached it
    censored_at: list[int]  # shot counts for seeds that did not
    threshold: float
    # Provenance. Smoothing window and persistence change the answer by tens of
    # percent, and the call sites differ (gate_headroom uses window=25, the
    # default is 10). Without recording them, two shots-to-threshold numbers
    # computed differently get compared as if they were the same quantity.
    window: int = 10
    persistence: int = 5

    @property
    def n_reached(self) -> int:
        return len(self.values)

    @property
    def n_total(self) -> int:
        return len(self.values) + len(self.censored_at)

    @property
    def reach_rate(self) -> float:
        return self.n_reached / self.n_total if self.n_total else 0.0

    @property
    def median(self) -> float:
        """Median among seeds that reached it. NaN if none did."""
        return float(np.median(self.values)) if self.values else float("nan")

    def ci(self, level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple:
        """Bootstrap CI of the median, resampling seeds."""
        return bootstrap_ci(np.array(self.values), np.median, level, n_boot, seed)

    def summary(self) -> str:
        if not self.values:
            return (f"never reached threshold {self.threshold:.3g} "
                    f"in {self.n_total} seeds (censored at "
                    f"{max(self.censored_at) if self.censored_at else 0} shots)")
        lo, hi = self.ci()
        s = (f"{self.median:.0f} shots [95% CI {lo:.0f}-{hi:.0f}] "
             f"(window={self.window}, persistence={self.persistence})")
        if self.censored_at:
            s += (f"  -- among {self.n_reached}/{self.n_total} seeds that "
                  f"reached it; {len(self.censored_at)} censored")
        return s


def shots_to_threshold_single(
    rewards: np.ndarray,
    threshold: float,
    window: int = 10,
    persistence: int = 5,
) -> int | None:
    """Shots until the smoothed curve reaches and HOLDS the threshold.

    `persistence` guards against a noisy curve tipping over the line for one
    shot by luck. The crossing counts only if the smoothed reward stays at or
    above the threshold for `persistence` consecutive shots.

    Returns the number of shots consumed (1-based), or None if never reached.
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        return None
    if persistence < 1:
        raise ValueError("persistence must be >= 1")
    sm = trailing_mean(rewards, window)
    above = sm >= threshold
    if above.size < persistence:
        return None
    for i in range(above.size - persistence + 1):
        if above[i : i + persistence].all():
            # AUDIT #5: the shot that CONFIRMS the hold, not the one that
            # started it. See shots_to_competence_evaluated.
            return i + persistence
    return None


def shots_to_competence(
    runs: list[RunLog],
    tolerance: float,
    window: int = 25,
    persistence: int = 5,
) -> ThresholdResult:
    """Shots until the controller WORKS: |beta_N - target| inside the task
    tolerance, smoothed and held.

    WHEN TO USE THIS INSTEAD OF `threshold_from_reference`
    ------------------------------------------------------
    The relative threshold sits a fraction of the way from a run's start to
    its own plateau. Across CONDITIONS on one task that is right: every
    condition is measured against one shared reference, so the anchor
    cancels. Across TASKS it is wrong, because each task supplies its own
    anchor.

    Measured, and the reason this function exists: `easy` crossed its own
    80% line at 240 shots and `moderate` at 146. Moderate is not easier --
    its curve traverses a smaller range, so 80% of that range arrives
    sooner, while the search was still improving until shot ~400. A
    self-normalising threshold rewards a curve that barely moves.

    Tracking error inside tolerance is absolute, comparable across presets
    and conditions, and is what a fusion audience means by a controller
    that works.

    Implemented on the negated error against the negated tolerance so the
    smoothing and persistence rules are literally the same code as the
    relative metric, not a second convention that could drift from it.
    """
    values: list[float] = []
    censored: list[int] = []
    for r in runs:
        n = shots_to_threshold_single(
            -r.beta_errors(), -abs(tolerance), window, persistence)
        if n is None:
            censored.append(len(r))
        else:
            values.append(float(n))
    return ThresholdResult(values=values, censored_at=censored,
                           threshold=abs(tolerance), window=window,
                           persistence=persistence)


def shots_to_competence_evaluated(
    runs: list[RunLog],
    tolerance: float,
    window: int = 3,
    persistence: int = 2,
) -> ThresholdResult:
    """Shots to competence, measured only on EVALUATION shots.

    WHY THE CANDIDATE STREAM CANNOT ANSWER THIS
    -------------------------------------------
    Every ordinary shot fires a candidate -- a deliberate perturbation of
    the incumbent. A controller that already tracks perfectly still logs
    errors, because it is still probing. So `shots_to_competence` over all
    shots measures the exploration schedule at least as much as the
    controller.

    It bit twice before this existed. A cold-start arm plateaued at -0.001
    reward -- essentially perfect control -- and was reported as NEVER
    reaching competence in the same run.

    Evaluation shots fire the incumbent unperturbed, so the only noise left
    is the plant's. They cost budget like any other shot, which is why the
    answer is still reported in TOTAL shots consumed: a machine evaluating
    every tenth shot has still fired all ten.

    The window is small by default because these points are sparse and
    clean, where the candidate stream needed 25 to see through the noise.
    """
    values: list[float] = []
    censored: list[int] = []
    for r in runs:
        errs, idx = r.evaluation_errors()
        if errs.size < persistence:
            censored.append(len(r))
            continue
        sm = trailing_mean(errs, window)
        inside = sm <= abs(tolerance)
        hit = None
        for i in range(inside.size - persistence + 1):
            if inside[i : i + persistence].all():
                # AUDIT #5: the CONFIRMING evaluation, not the first of the
                # holding window. Persistence exists because one good point
                # is not competence, so the budget that establishes
                # competence is the budget spent reaching the last point of
                # the run -- reporting the first backdates the answer past
                # the shots that earned it and inflates early ratios.
                hit = int(idx[i + persistence - 1]) + 1
                break
        if hit is None:
            censored.append(len(r))
        else:
            values.append(float(hit))
    return ThresholdResult(values=values, censored_at=censored,
                           threshold=abs(tolerance), window=window,
                           persistence=persistence)


def shots_to_threshold(
    runs: list[RunLog],
    threshold: float,
    window: int = 10,
    persistence: int = 5,
) -> ThresholdResult:
    """Aggregate shots-to-threshold over the seeds of one condition/device."""
    values: list[float] = []
    censored: list[int] = []
    for r in runs:
        n = shots_to_threshold_single(r.rewards(), threshold, window, persistence)
        if n is None:
            censored.append(len(r))
        else:
            values.append(float(n))
    return ThresholdResult(values=values, censored_at=censored, threshold=threshold,
                           window=window, persistence=persistence)


def _joint_success(shot: ShotRecord, tolerance: float,
                   expected_steps: int | None = None) -> bool:
    """The protocol's per-shot endpoint, including explicit early termination."""
    return bool(
        not shot.terminated_early
        and not shot.violated
        and shot.steps > 0
        and (expected_steps is None or shot.steps >= expected_steps)
        and np.isfinite(shot.beta_error)
        and shot.beta_error <= abs(tolerance)
    )


def shots_to_joint_competence_evaluated(
    runs: list[RunLog],
    tolerance: float,
    expected_steps: int | None = None,
    persistence: int = 2,
) -> ThresholdResult:
    """Shots until consecutive evaluations satisfy the joint endpoint.

    PROTOCOL.md 4 requires every successful shot to complete, remain within
    limits, and track inside tolerance. The older tracking-only metric remains
    available as a diagnostic, but smoothing its errors can hide violations,
    premature termination, and even a failed solve with a non-finite error.
    Here each evaluation passes independently; a failed evaluation resets the
    streak. Candidate shots count toward cost but cannot certify competence.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if persistence < 1:
        raise ValueError("persistence must be >= 1")
    if expected_steps is not None and expected_steps < 1:
        raise ValueError("expected_steps must be >= 1")
    values: list[float] = []
    censored: list[int] = []
    for run in runs:
        streak = 0
        for shot in run.shots:
            if not shot.is_evaluation or not shot.evaluation_eligible:
                continue
            streak = streak + 1 if _joint_success(shot, tolerance, expected_steps) else 0
            if streak >= persistence:
                values.append(float(shot.shot + 1))
                break
        else:
            censored.append(len(run))
    return ThresholdResult(values=values, censored_at=censored,
                           threshold=float(tolerance), window=1,
                           persistence=persistence)


def binomial_exact_interval(successes: int, total: int,
                            level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval, including uncertainty at 0/n and n/n.

    Inverting binomial tails keeps this small-seed calculation independent
    of SciPy and, unlike a bootstrap at a boundary, gives a nonzero width.
    """
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total")
    if not 0 < level < 1:
        raise ValueError("level must lie between zero and one")
    if total == 0:
        return float("nan"), float("nan")

    def inverse_tail(k: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            p = (lo + hi) / 2
            tail = sum(comb(total, x) * p ** x * (1 - p) ** (total - x)
                       for x in range(k, total + 1))
            if tail < target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    alpha = 1 - level
    lower = inverse_tail(successes, alpha / 2) if successes else 0.0
    upper = (inverse_tail(successes + 1, 1 - alpha / 2)
             if successes < total else 1.0)
    return lower, upper


@dataclass
class PairedReachResult:
    """Matched-seed effect at a common, fixed target-shot budget."""

    n_pairs: int
    baseline_only: int
    method_only: int
    difference: float  # method reach rate minus baseline reach rate
    ci_low: float
    ci_high: float
    p_value: float

    def summary(self) -> str:
        return (
            f"paired reach difference {self.difference:+.0%} "
            f"[conservative 95% CI {self.ci_low:+.0%}, {self.ci_high:+.0%}], "
            f"exact paired p={self.p_value:.4f}; "
            f"discordant seeds {self.method_only} method-only / "
            f"{self.baseline_only} baseline-only ({self.n_pairs} pairs)"
        )


def paired_joint_reach(
    baseline_runs: list[RunLog],
    method_runs: list[RunLog],
    tolerance: float,
    expected_steps: int | None = None,
    persistence: int = 2,
) -> PairedReachResult:
    """Compare joint reach with an exact test on discordant matched seeds.

    Pair by seed identifiers, never list position, and require every seed on
    both sides. The confidence interval bounds the two discordant-category
    probabilities simultaneously with exact binomial intervals (Bonferroni),
    then subtracts them. It is conservative but retains finite-sample coverage
    and does not collapse to zero width when all six seeds agree. The p-value
    is the two-sided exact McNemar test, unadjusted for multiple comparisons.
    A nonsignificant result or a minimum detectable effect is not an
    equivalence bound.
    """
    def indexed(runs: list[RunLog]) -> dict[int, RunLog]:
        by_seed = {r.seed: r for r in runs}
        if len(by_seed) != len(runs):
            raise ValueError("paired reach requires unique seed identifiers")
        return by_seed

    baseline, method = indexed(baseline_runs), indexed(method_runs)
    if not baseline or baseline.keys() != method.keys():
        raise ValueError("paired reach requires the same nonempty seed set")
    if len({len(r) for r in baseline_runs + method_runs}) != 1:
        raise ValueError("paired reach requires a common fixed shot budget")
    baseline_only = method_only = 0
    for seed, base in baseline.items():
        other = method[seed]
        if base.device != other.device or len(base) != len(other):
            raise ValueError("paired reach requires the same device and shot budget")
        b = shots_to_joint_competence_evaluated(
            [base], tolerance, expected_steps, persistence).n_reached
        m = shots_to_joint_competence_evaluated(
            [other], tolerance, expected_steps, persistence).n_reached
        baseline_only += int(b and not m)
        method_only += int(m and not b)

    n = len(baseline)
    discordant = baseline_only + method_only
    p_value = (min(1.0, 2 * sum(comb(discordant, k)
                              for k in range(min(baseline_only, method_only) + 1))
                   / 2 ** discordant)
               if discordant else 1.0)
    # Each 97.5% interval fails with probability <= 2.5%; together their
    # coverage is at least 95%, regardless of dependence between categories.
    plus_lo, plus_hi = binomial_exact_interval(method_only, n, level=0.975)
    minus_lo, minus_hi = binomial_exact_interval(baseline_only, n, level=0.975)
    return PairedReachResult(
        n_pairs=n, baseline_only=baseline_only, method_only=method_only,
        difference=(method_only - baseline_only) / n,
        ci_low=plus_lo - minus_hi, ci_high=plus_hi - minus_lo,
        p_value=p_value,
    )


@dataclass
class SpeedupResult:
    ratio: float
    ci_low: float
    ci_high: float
    baseline: ThresholdResult
    method: ThresholdResult
    warnings: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        s = (f"{self.ratio:.2f}x fewer shots "
             f"[95% CI {self.ci_low:.2f}-{self.ci_high:.2f}]")
        for w in self.warnings:
            s += f"\n  WARNING: {w}"
        return s


def speedup_ratio(
    baseline_runs: list[RunLog],
    method_runs: list[RunLog],
    threshold: float,
    window: int = 10,
    persistence: int = 5,
    max_censoring_gap: float = 0.2,
    n_boot: int = 2000,
    seed: int = 0,
) -> SpeedupResult:
    """Shots-to-threshold ratio, baseline / method. >1 means the method is faster.

    This is the primary headline number. It is deliberately hard to report
    dishonestly: if the two conditions reach the threshold at materially
    different rates, the ratio compares different populations and a warning is
    attached that `trustworthy` exposes.
    """
    base = shots_to_threshold(baseline_runs, threshold, window, persistence)
    meth = shots_to_threshold(method_runs, threshold, window, persistence)
    return speedup_from_results(base, meth, max_censoring_gap=max_censoring_gap,
                                n_boot=n_boot, seed=seed)


def speedup_from_results(
    base: ThresholdResult,
    meth: ThresholdResult,
    max_censoring_gap: float = 0.2,
    n_boot: int = 2000,
    seed: int = 0,
) -> SpeedupResult:
    """The ratio and every guard on it, given two already-computed results.

    Split out so a speedup can be taken on ANY shots-to-X result -- notably
    `shots_to_competence`, whose criterion is absolute -- without the
    censoring, smoothing and sample-size guards being reimplemented
    somewhere else and drifting. Those guards are the reason this number is
    hard to report dishonestly; a second copy of them would not stay that
    way.
    """
    warnings: list[str] = []
    if not base.values or not meth.values:
        which = "baseline" if not base.values else "method"
        return SpeedupResult(float("nan"), float("nan"), float("nan"), base, meth,
                             [f"{which} never reached the threshold; no ratio exists"])

    if (base.window, base.persistence) != (meth.window, meth.persistence):
        warnings.append(
            "the two conditions were smoothed differently "
            f"({base.window}/{base.persistence} vs {meth.window}/{meth.persistence}); "
            "the ratio is not meaningful."
        )
    gap = abs(base.reach_rate - meth.reach_rate)
    if base.censored_at or meth.censored_at:
        warnings.append(
            "the ratio conditions on reaching seeds and excludes censored runs; "
            "even equal reach rates do not make it a population speedup. "
            "Report reach rates at a fixed budget as the primary endpoint."
        )
    if gap > max_censoring_gap:
        warnings.append(
            f"censoring differs: baseline reached {base.reach_rate:.0%} of seeds, "
            f"method {meth.reach_rate:.0%}. The ratio compares different "
            "populations and overstates whichever condition failed more often. "
            "Report the reach rates alongside it, or run longer."
        )
    if base.n_total < 3 or meth.n_total < 3:
        warnings.append(
            f"only {min(base.n_total, meth.n_total)} seeds; a sample-efficiency "
            "claim is a trend claim and will not survive this (SPEC.md §8)."
        )
    # AUDIT #7. The censoring check above compares reach RATES, so two arms
    # censored equally badly pass it, and the seed check counts seeds that
    # RAN rather than seeds that contributed. One success against four
    # censored runs on each side returned 10.00x [10.00, 10.00] with
    # trustworthy=True -- a bootstrap over a single value has no width, and
    # 80% of both arms was silently excluded from the estimate.
    n_est = min(base.n_reached, meth.n_reached)
    if n_est < 3:
        warnings.append(
            f"the ratio is estimated from {n_est} reaching seed(s) per arm "
            f"({base.n_reached}/{base.n_total} baseline, "
            f"{meth.n_reached}/{meth.n_total} method). A bootstrap over so "
            "few values has almost no width and the interval will look "
            "precise for a reason that has nothing to do with the effect. "
            "Report reach rates at a fixed budget instead."
        )

    rng = np.random.default_rng(seed)
    b, m = np.array(base.values), np.array(meth.values)
    draws = np.array([
        np.median(rng.choice(b, b.size, replace=True))
        / max(np.median(rng.choice(m, m.size, replace=True)), 1e-9)
        for _ in range(n_boot)
    ])
    return SpeedupResult(
        ratio=float(np.median(b) / max(np.median(m), 1e-9)),
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
        baseline=base, method=meth, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 2. Asymptotic performance
# ---------------------------------------------------------------------------


def best_evaluated_error(runs: list[RunLog]) -> tuple[float, float]:
    """Smallest tracking error any evaluation reached, median over seeds.

    Shots-to-competence answers "did it get inside the tolerance, and when",
    and on a device with a tight tolerance the answer can be "no" for every
    arm -- which makes the fold silent rather than negative. sparc_like on
    `moderate` is that case: its tolerance is 0.0091, twenty-six times
    tighter than DIII-D's, and the conventional controller needs its best
    gain to reach 0.0066.

    This is the number that still separates arms when none of them passes:
    how close did each get. In the same units as the tolerance, so it can be
    read against it directly.

    IT IS A MINIMUM, AND MINIMA DO NOT COMPARE TO MEDIANS. Use it to rank
    arms against each other, never against the conventional controller's
    reported error -- that one is a median over twenty evaluation shots, and
    setting a best-of-N against a median flatters the best-of-N by however
    much variance it had. I made exactly that comparison on iter_like, read
    0.0360 against the classical 0.0383 and briefly had the learned arms
    ahead. The comparable statement between a learned arm and the classical
    one is the joint success rate, which both report on the same footing.

    Returns (median over seeds, spread as the interquartile range). NaN if no
    seed produced a finite evaluation.
    """
    bests = []
    for r in runs:
        errs, _ = r.evaluation_errors()
        errs = errs[np.isfinite(errs)]
        if errs.size:
            bests.append(float(errs.min()))
    if not bests:
        return float("nan"), float("nan")
    a = np.array(bests, dtype=float)
    return float(np.median(a)), float(np.subtract(*np.percentile(a, [75, 25])))


def asymptotic_performance_evaluated(
    runs: list[RunLog], last_fraction: float = 0.5
) -> tuple[float, float]:
    """Final performance measured on EVALUATION shots only.

    `asymptotic_performance` averages `r.rewards()`, which is every shot --
    and most shots are candidates, deliberate perturbations of the
    incumbent. So the plateau it reports mixes how good the controller is
    with how hard the search was still probing, and two arms whose search
    had annealed differently would differ in plateau with identical
    controllers.

    This is the same defect that made shots-to-competence unreadable, in a
    metric I had already started quoting as evidence: `federated_similarity`
    was reported as having the best plateau (-0.985 against -3.347) and that
    claim rested on the contaminated number.

    `last_fraction` defaults to 0.5 rather than 0.2 because evaluation shots
    are sparse -- one in five at the usual cadence -- so the last fifth of
    them can be two points.
    """
    if not 0 < last_fraction <= 1:
        raise ValueError("last_fraction must be in (0, 1]")
    finals = []
    for r in runs:
        vals = np.array([s.reward for s in r.shots if s.is_evaluation],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        k = max(1, int(round(vals.size * last_fraction)))
        finals.append(float(vals[-k:].mean()))
    if not finals:
        return float("nan"), float("nan")
    a = np.array(finals, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def asymptotic_performance(runs: list[RunLog], last_fraction: float = 0.2
                           ) -> tuple[float, float]:
    """Mean final-plateau reward across seeds, and its standard error.

    Federation must not cost final quality. A method that reaches threshold
    faster but plateaus lower has traded the thing the device actually cares
    about for the thing the paper measures.
    """
    if not 0 < last_fraction <= 1:
        raise ValueError("last_fraction must be in (0, 1]")
    finals = []
    for r in runs:
        rw = r.rewards()
        rw = rw[np.isfinite(rw)]
        if rw.size == 0:
            continue
        k = max(1, int(round(rw.size * last_fraction)))
        # MEDIAN, not mean, within a seed. A plateau is a level, and one
        # blown-up shot in the tail (beta_N excursions make per-step costs
        # quadratic, so a single bad step can be many orders of magnitude out)
        # dragged the mean from -0.03 to -12500. Across seeds the mean is kept,
        # because seed-to-seed spread is signal rather than contamination.
        finals.append(float(np.median(rw[-k:])))
    if not finals:
        return float("nan"), float("nan")
    a = np.array(finals)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def threshold_from_reference(
    reference_runs: list[RunLog],
    fraction: float = 0.8,
    last_fraction: float = 0.2,
    start_window: int = 25,
) -> float:
    """Threshold at `fraction` of the way from STARTING to PLATEAU performance.

    Convention in transfer-RL work, and it avoids picking an absolute number
    out of the air. The natural reference is the centralised upper bound, so
    the threshold reads "80% of the way from untrained to what you could get if
    you pooled all the data".

    ROBUSTNESS -- this is why the anchor is the start, not the worst shot.
    Rewards here are negative costs, so the interpolation needs a low anchor.
    Using ``min()`` over raw rewards made the threshold hostage to a single
    shot: per-step cost is quadratic in tracking error, so one excursion can be
    orders of magnitude out, and a lone -1e5 shot dragged the threshold from
    -2.0 to -20000 -- at which point EVERY condition cleared it on shot 1 and
    shots-to-threshold collapsed from ~104 to 1. The primary metric would have
    read as an instant win for everything.

    The starting level is both robust (a median over the first `start_window`
    shots, pooled across seeds) and more meaningful: progress is measured from
    where an untrained policy actually begins.
    """
    plateau, _ = asymptotic_performance(reference_runs, last_fraction)
    if not np.isfinite(plateau):
        raise ValueError("reference condition has no usable runs")

    heads = []
    for r in reference_runs:
        rw = r.rewards()
        rw = rw[np.isfinite(rw)]
        if rw.size:
            heads.append(np.median(rw[: max(1, min(start_window, rw.size))]))
    if not heads:
        raise ValueError("reference condition has no usable runs")
    start = float(np.median(heads))

    if not np.isfinite(start) or plateau <= start:
        raise ValueError(
            f"reference condition does not improve (start {start:.3g}, "
            f"plateau {plateau:.3g}); no threshold can be defined from it"
        )
    return start + fraction * (plateau - start)


# ---------------------------------------------------------------------------
# 3 & 5. Violations
# ---------------------------------------------------------------------------


def violations_during_training(runs: list[RunLog]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the cumulative violation count vs shot.

    Runs of unequal length are truncated to the shortest, because extending the
    mean over a shrinking sample makes the curve bend for bookkeeping reasons
    rather than physical ones.
    """
    curves = [r.cumulative_violations() for r in runs if len(r)]
    if not curves:
        return np.zeros(0), np.zeros(0)
    n = min(len(c) for c in curves)
    stack = np.stack([c[:n] for c in curves]).astype(float)
    sem = (stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
           if stack.shape[0] > 1 else np.zeros(n))
    return stack.mean(axis=0), sem


def violation_rate(runs: list[RunLog], last_fraction: float = 1.0) -> float:
    """Fraction of shots that crossed any limit.

    With `last_fraction < 1`, restricted to the tail -- which is what the
    unseen-regime test needs, since early exploratory shots are not the claim.
    """
    total = unsafe = 0
    for r in runs:
        flags = r.violation_flags()
        if flags.size == 0:
            continue
        k = max(1, int(round(flags.size * last_fraction)))
        tail = flags[-k:]
        total += tail.size
        unsafe += int(tail.sum())
    return unsafe / total if total else float("nan")


def joint_success_rate(runs: list[RunLog], tolerance: float,
                       expected_steps: int | None = None) -> float:
    """Fraction of shots that COMPLETED, stayed within limits, AND tracked.

    AUDIT. A violation rate alone can be satisfied by a controller that does
    nothing: a shot with tracking error 99 and no limit crossing counts as a
    success against an isolated arm that crossed one, and the safety claim
    reads as held when the plant was simply never driven. Safety that is
    purchased with uselessness is not the claim.

    Three conditions, all required per shot:

      * completion   -- the discharge ran to the end of the episode;
      * containment  -- no limit crossed, solver failures included;
      * competence   -- mean |beta_N - target| inside the task tolerance.

    Completion is checked explicitly because tracking error is averaged over
    SURVIVING steps: a shot that dies at step 2 near its setpoint would
    otherwise report an excellent error, and the failure would be invisible
    in exactly the number meant to detect it.
    """
    total = good = 0
    for r in runs:
        for sh in r.shots:
            total += 1
            good += int(_joint_success(sh, tolerance, expected_steps))
    return good / total if total else float("nan")


def completion_rate(runs: list[RunLog], expected_steps: int) -> float:
    """Fraction of shots that ran to the end of the episode."""
    total = done = 0
    for r in runs:
        for sh in r.shots:
            total += 1
            done += int(sh.steps >= expected_steps)
    return done / total if total else float("nan")


@dataclass
class CatastropheResult:
    """The headline experiment (SPEC.md Phase 6)."""

    isolated_rate: float
    federated_rate: float
    n_isolated: int
    n_federated: int
    # AUDIT: safety alone cannot carry the claim. A controller that never
    # drives the plant is perfectly safe and perfectly useless.
    isolated_success: float = float("nan")
    federated_success: float = float("nan")

    @property
    def claim_holds(self) -> bool:
        """Federated A avoids the violation and isolated A does not.

        Deliberately strict: the spec's claim is not "fewer violations", it is
        that the federated device AVOIDS the regime. A federated rate that is
        merely lower is a weaker result and should be reported as such.
        """
        safe = self.federated_rate == 0.0 and self.isolated_rate > 0.0
        # If the joint endpoint was supplied, the federated arm must also
        # have done the job. NaN means it was not measured, and an
        # unmeasured competence check must not silently grant the claim --
        # so it blocks it.
        if not np.isfinite(self.federated_success):
            return False
        return safe and self.federated_success > 0.0

    def summary(self) -> str:
        s = (f"unseen-regime violation rate: "
             f"isolated {self.isolated_rate:.1%} (n={self.n_isolated}), "
             f"federated {self.federated_rate:.1%} (n={self.n_federated})")
        s += (f"\n  joint success (completed + contained + tracking): "
              f"isolated {self.isolated_success:.1%}, "
              f"federated {self.federated_success:.1%}")
        if self.claim_holds:
            return s + "\n  CLAIM HOLDS: federated avoided the regime entirely."
        if self.federated_rate < self.isolated_rate:
            return s + ("\n  PARTIAL: federation reduced but did not eliminate "
                        "violations. Weaker than the spec's claim -- say so.")
        return s + "\n  CLAIM FAILS: federation did not reduce violations."


def catastrophe_test(isolated_runs: list[RunLog], federated_runs: list[RunLog],
                     last_fraction: float = 1.0,
                     tolerance: float | None = None,
                     expected_steps: int | None = None) -> CatastropheResult:
    def joint(runs):
        if tolerance is None:
            return float("nan")
        return joint_success_rate(runs, tolerance, expected_steps)

    return CatastropheResult(
        isolated_rate=violation_rate(isolated_runs, last_fraction),
        federated_rate=violation_rate(federated_runs, last_fraction),
        n_isolated=sum(len(r) for r in isolated_runs),
        n_federated=sum(len(r) for r in federated_runs),
        isolated_success=joint(isolated_runs),
        federated_success=joint(federated_runs),
    )


# ---------------------------------------------------------------------------
# 4. Cold start
# ---------------------------------------------------------------------------


def cold_start_shots(runs: list[RunLog], threshold: float, window: int = 10,
                     persistence: int = 5) -> ThresholdResult:
    """Shots to competence for a device joining an existing federation.

    Mechanically the same as shots-to-threshold; separated because it answers a
    different question and is the strongest practical number in the set -- how
    long before a brand-new machine is useful.
    """
    return shots_to_threshold(runs, threshold, window, persistence)


# ---------------------------------------------------------------------------
# 6. Staleness tolerance
# ---------------------------------------------------------------------------


def staleness_tolerance(runs_by_lag: dict[int, list[RunLog]],
                        last_fraction: float = 0.2
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final performance as a function of update lag, in rounds.

    Justifies FedBuff: if performance is flat in lag, asynchrony costs nothing
    and synchronous rounds -- which tokamak campaigns cannot provide anyway --
    are not worth wanting.
    """
    lags = np.array(sorted(runs_by_lag))
    perf, sem = [], []
    for lag in lags:
        m, s = asymptotic_performance(runs_by_lag[int(lag)], last_fraction)
        perf.append(m)
        sem.append(s)
    return lags, np.array(perf), np.array(sem)


# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(x: np.ndarray, stat=np.median, level: float = 0.95,
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling the seed axis."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.choice(x, x.size, replace=True)) for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    return float(np.percentile(draws, 100 * a)), float(np.percentile(draws, 100 * (1 - a)))


def learning_curve(runs: list[RunLog], window: int = 10
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean smoothed reward vs shot, with standard error across seeds."""
    curves = [trailing_mean(r.rewards(), window) for r in runs if len(r)]
    if not curves:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    n = min(len(c) for c in curves)
    stack = np.stack([c[:n] for c in curves])
    sem = (stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
           if stack.shape[0] > 1 else np.zeros(n))
    return np.arange(1, n + 1), stack.mean(axis=0), sem
