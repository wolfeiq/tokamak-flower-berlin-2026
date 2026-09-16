# Deliverables: the statement, the metrics, the figures

This is what the project has to produce. SPEC.md is the argument; PLAN.md is
the build order; this file is the output.

---

## The statement

Research tokamaks each learn their control policies alone, from a few thousand
shots on one machine. Shots are expensive, campaigns are short, and no device
can afford to explore the regimes where it would fail. We build a federated
multi-agent system where control agents are grouped by function within a
device and federated only with their counterparts on other devices,
asynchronously, in a physics-normalised space. The claim is that a device
reaches competent control in substantially fewer shots, and learns to avoid
regimes it never entered, while exchanging only model weights — never discharge
data, never the controller itself.

---

## The currency is shots

Not wall-clock, not gradient steps, not environment transitions. A shot is one
discharge: one episode, one TORAX simulation. It is what costs money and
machine time, and it is the number a fusion audience reads.

Enforced structurally: `hfmarl/metrics/log.py` records one `ShotRecord` per
episode and every metric is denominated in shots. There is no code path that
reports a learning curve against gradient steps.

---

## What kills the claim

> If isolated training reaches threshold in a few hundred shots, the gain is
> too small to matter. Design the task hard enough that isolated learning is
> genuinely slow — otherwise there's no headroom for federation to show
> anything.

**This is a blocking gate, not a caveat.** `scripts/gate_headroom.py` trains
isolated-only at increasing difficulty and refuses to green-light Phase 5 until
three things hold at once:

| check | criterion | failure mode if ignored |
|---|---|---|
| **headroom** | isolated median ≥ 1000 shots | any speedup is within campaign noise |
| **measurable** | ≥80% of seeds converge, median ≤ 20000 shots | no denominator; the "ratio" compares failure rates |
| **affordable** | full matrix ≤ 72 h serial | headroom fine, experiment never finishes |

These pull against each other — harder task means more headroom *and* more
compute — so the gate names the binding constraint and gives specific advice
for that one. Difficulty is a first-class object (`hfmarl/envs/task.py`): five
presets from `trivial` to `brutal`, varying setpoint schedule, tolerance,
transport model, episode length and disturbances.

A setpoint above the beta limit makes tracking and safety strictly
contradictory — that is *unsatisfiable*, not *hard*, and it is an easy mistake
to make while turning the difficulty up. `validate_against_limits` catches it
at construction.

### The gate's own weakness, stated plainly

Shots-to-threshold is a property of the task **and the optimiser**. The gate
measures with simple hill climbing (one shot per step, so the shot axis is
exact). A weak optimiser needs more shots, so **the gate is biased toward
passing**: a task hill climbing takes 3000 shots to crack might take PPO 300,
and the certified headroom would not exist.

So a PASS is *provisional* — re-measure the isolated baseline with whatever
algorithm the matrix actually uses. A FAIL is trustworthy: if even hill
climbing converges in a few hundred shots, no stronger learner will make the
task harder.

### Reward shaping: no free exit

A violation ends the episode. Without care, a policy that cannot track pays a
per-step cost for a whole episode but only a one-off penalty for crashing out
on step one — and at the β_N error an untrained policy actually starts from
(~1.9), crashing scores *better* on every task preset. The agent learns to
drive into the limit on purpose, which would corrupt the
violations-during-training curve specifically.

`_early_exit_charge` bills the escaped remainder at the rate already being
incurred, so violating costs exactly `violation_penalty` more than finishing,
independent of task length and of when it happens. Regression-tested across
every preset in `tests/test_env_contract.py`.

---

## The six metrics

| # | metric | why | implementation |
|---|---|---|---|
| 1 | **Shots to threshold** | primary; reported as a ratio vs isolated | `shots_to_threshold`, `speedup_ratio` |
| 2 | **Asymptotic performance** | federation must not cost final quality | `asymptotic_performance` |
| 3 | **Limit violations during training** | safe behaviour transfers before good behaviour does | `violations_during_training` |
| 4 | **Cold-start shots** | strongest practical number: a new machine joining | `cold_start_shots` |
| 5 | **Unseen-regime violation rate** | the catastrophe claim | `catastrophe_test` |
| 6 | **Staleness tolerance** | justifies FedBuff | `staleness_tolerance` |

### The statistics that could lie, and what stops them

**Censoring.** A run that never reaches threshold is right-censored. Dropping
non-reachers biases the comparison toward whichever condition failed more often
— four seeds fail, the one lucky seed reports a fast time, and the method looks
excellent. Substituting the budget invents data. So `ThresholdResult` carries
`values` and `censored_at` separately, and **`speedup_ratio` refuses to give a
clean number** when the two conditions are censored at materially different
rates: it attaches a warning and `trustworthy` goes False.

**Lucky crossings.** A noisy curve tips over a threshold by chance.
Convergence requires the smoothed reward to *hold* above the line for
`persistence` consecutive shots.

**Look-ahead.** Smoothing is a strictly causal trailing mean. A centred filter
would let shots that had not happened yet trigger a crossing, which for a
sample-efficiency claim is cheating.

**Ragged matrices.** Conditions trained on different seed counts are not fairly
comparable, and an empty cell vanishes silently from a grouped bar chart.
`ExperimentLog.coverage()` checks the whole matrix and warns.

**Violation double-counting.** A shot that crosses three limits is one unsafe
discharge, not three. Counting crossings would make a single bad shot look like
a trend.

**Weak claims dressed as strong ones.** The spec's claim is that federated A
*avoids* the regime, not that it violates less often. `CatastropheResult.claim_holds`
requires a federated rate of exactly zero; a merely-lower rate reports as
`PARTIAL` and says so.

---

## The six figures

Generated by `hfmarl/metrics/plots.py`. See them now, on synthetic data:

    uv run python scripts/demo_figures.py

Those outputs are stamped **SYNTHETIC** in the corner — they are a rendering
test, never a result.

**1. Learning curves.** Reward vs shots, five conditions on one axis. The
load-bearing comparison is **naive FedAvg vs isolated**: if role matching
matters, ignoring roles should sit at or *below* training alone. The figure
annotates whether that holds, because it is the part a reader most easily
misses.

**2. Shots-to-threshold, per device.** Grouped bars. Prediction: the smallest,
data-poorest device gains most — the headline for anyone running a small
machine. Censored seeds are hatched bars at the budget with an explicit
`n/N censored` label, never averaged in.

**3. Violations during training.** Cumulative count vs shots. The expected
signature is that the federated curve **flattens earlier** — final height
matters less than where each curve stops rising.

**4. The catastrophe plot.** Device A in the unseen regime, individual
trajectories approaching the limit; isolated crosses, federated turns away.
Takes raw trajectories rather than means, because a mean would smear exactly
the behaviour being claimed. Crossing counts are printed on the figure.

**5. Similarity-weighting ablation.** Benefit vs dimensionless distance. If the
physics weighting means anything, benefit should **decay** with distance in
(ρ*, ν*, β, q) space. The figure fits the slope and states its own verdict — a
flat relationship says the metric is not capturing transferability, which is a
real result, not a tuning failure.

**6. Cold start.** A new device joining an existing federation, with and
without. The gap between the two markers is the answer.

**S1 (supplementary). Staleness tolerance.** Performance vs update lag. Flat in
lag means asynchrony costs nothing, which is the FedBuff argument — tokamak
campaigns cannot supply synchronous rounds regardless.

### Figure conventions

One condition is one colour in every figure (Okabe-Ito, colour-blind safe). The
method is drawn thickest. The centralised upper bound is **dashed**, because it
is not an achievable condition and conflating it with the others is the easiest
way to overclaim. Missing conditions are annotated on the figure in red, never
silently dropped.

---

## The five conditions

All on identical seeds. Defined once in `hfmarl/experiments/conditions.py` so
no figure, metric or runner can disagree about names or order.

1. **isolated** — baseline, and the denominator of every ratio.
2. **FedBuff, role-matched, uniform** — isolates the value of federating at all.
3. **FedBuff, role-matched, similarity** ← **the method**. The gap over (2) is
   the entire contribution of SPEC.md §4b; if it is zero, the physics weighting
   earns nothing.
4. **naive FedAvg, role-blind** — negative control. Should land at or below
   isolated. If it matches the others, role matching is unsupported.
5. **centralised** — upper bound, and *not achievable in practice*: private
   fusion will not pool discharge data (SPEC.md §9).
