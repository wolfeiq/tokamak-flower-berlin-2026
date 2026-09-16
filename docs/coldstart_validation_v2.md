# Proposed cold-start validation protocol

Status: design proposal, not implemented or launched. This is a follow-up to
the exploratory run `20260916-065401_6df3978`; it does not redefine that run's
registered endpoint after seeing its outcomes.

## Why a different test is needed

At the 07:50 checkpoint on 16 September 2026, all six SPARC-like scratch runs
reached the current criterion in 21, 11, 26, 36, 26, and 11 target shots.
Each also passed its last five eligible evaluations. Thus the observed ceiling
at 120 shots is not solely a single lucky early crossing. There is little
headroom for the primary reach-rate comparison on this fold.

The current criterion nevertheless asks whether two consecutive evaluations
ever succeed during adaptation. It does not establish the reliability of a
fixed controller. Selection trials can contribute to that criterion; policy
parameters can change between evaluations; final policy weights are not saved.

Other attribution problems need controls: adoption resets the optimizer's
historical reward threshold, and the single-source arm receives 80 historical
shots whereas the three-source arms receive 240. The latter is a valid practical
comparison but cannot isolate source diversity from total source experience.

## Question and primary comparison

Question: given the same allowed design/calibration information and the same
target-shot budget, does transferring a federated policy improve reliable
control during early commissioning?

Use a small target adaptation budget, provisionally 20 shots, with zero-, 5-,
10-, and 40-shot checkpoints as secondary descriptive measurements. These
choices are informed by the pilot and must be frozen before fresh evaluation
seeds are examined. Do not reinterpret the running study at 20 shots as a new
confirmatory result.

At each checkpoint, save one controller and measure its joint success fraction
over independent, previously unused evaluation discharges: 30 at the primary
budget and 10 at each secondary budget for an initial affordable design. Use
30 at every budget if uniform precision justifies the extra compute. Success means
completed, no limit/solver failure, and tracking inside the fixed task tolerance.
The primary outcome is the mean held-out success fraction across independently
trained controllers at the chosen adaptation budget. Preserve tracking error,
violation rate, and per-seed outcomes as secondary measurements.

Use a single declared primary contrast. The mechanistic claim that repeated
federation matters requires a comparison to the one-time-merge control with
matching optimizer refresh; beating scratch alone establishes a warm-start
benefit, not its cause. Remaining contrasts and devices are secondary unless a
multiple-comparison procedure and its sample-size implications are specified.

## Separate learning, selection, and testing

1. Generate distinct, recorded random streams for source training, target
   adaptation, acceptance decisions, and held-out evaluation. Use the same
   held-out scenario list across arms for paired comparisons, never for tuning.
2. Save a copy of the incumbent weights and a policy hash at each predeclared
   budget. Freeze the final incumbent after the last adaptation shot as well.
3. Evaluate through a separate environment and RNG. Evaluation must not call
   `observe`, update an acceptance bar, change exploration scale, pick a
   checkpoint, or determine which arm receives more training.
4. A handover-selection shot is training/selection evidence, never independent
   confirmation. Log its role explicitly even if the handover is accepted.
5. Report the complete predeclared grid. Do not choose the best checkpoint or
   stop evaluation when enough successes have appeared.

For zero-adaptation transfer, the initial policy must be selected without any
target reward observations. State clearly whether paid calibration is allowed.
The current five locally trained probe shots cannot be described as zero
target training. A zero-adaptation test with five calibration shots is not a
zero-target-discharge test.

## Controls that identify the benefit

| Control | Purpose |
|---|---|
| Scratch | Establish practical benefit over local learning. |
| Conventional controller | Compare against a useful non-learning alternative under the same information and evaluation scenarios. Choose gains on development/source data. |
| One source, 240 historical shots | Match the total historical budget of three sources with 80 shots each. The existing 80-shot variant can remain a secondary practical baseline. |
| Three sources trained separately, merged once | Isolate combining final policies from repeated exchange during source training. |
| One-time merge with sham source exchanges | Reevaluate/reinstall each source's own policy on the exchange schedule, matching optimizer resets, selection logic, and charged shot costs without importing peer knowledge. |
| Uniform federation | Test repeated exchange against the matched sham/merge control. |
| Similarity federation | Secondary test of the physics weighting once the basic transfer comparison is interpretable. |

All source conditions must fit within the same declared per-device/total budget,
including selection and exchange trials. Target acceptance rules and optimizer
bookkeeping must also match between relevant controls. A sham control must
reproduce these mechanics; merely adding an extra evaluation is insufficient.

Cache unchanged pretrained sources across compatible comparison arms to save
compute, while retaining their historical shot cost. Never reuse a source
federation containing the held-out target. Controller replications that share
a source realization must remain grouped in statistical analysis.

## Tasks and generalization

Start with the existing moderate task at early budgets; a ceiling at 120 shots
does not by itself justify making the plant harder. Retain the complete
baseline-success distribution rather than choosing the budget that maximizes a
federated gap.

Evaluate all four held-out device folds for scope, then a small predeclared set
of physically valid disturbances and operating conditions. Screen task
reachability and conventional-controller performance on development cases.
Freeze difficulties and tolerances before confirmation. Do not search for a
task on which federation wins and call that a held-out result.

Initial-condition shifts are already supported. Actuator degradation, delay,
and sensor noise require explicit, validated implementations before they can
be claimed as tested. Safety conclusions need devices/tasks where the modeled
limits can actually be crossed, such as the existing DIII-D-like/TCV-like cases.
Generalization remains limited to the specified TORAX plant family.

## Cost and uncertainty

Report historical source shots, target calibration, adaptation/selection, and
held-out evaluation separately. Every target discharge still counts toward the
total. For the proposed primary checkpoint, 5 cached calibration + 20 adaptation
+ 30 validation shots is 55 target shots, not a 20-shot certification claim.
The smaller five-checkpoint schedule adds 70 validation discharges per trained
trajectory: 30 at budget 20 and 10 at each other budget. With 40 adaptation and
5 calibration shots, the full trajectory costs 115 target discharges. Using
30 evaluations at every checkpoint instead costs 195. Use the primary
checkpoint first in a small development pilot to check affordability before
executing the full grid.

Thirty test episodes estimate each controller's performance; they do not create
30 independent training runs. Resample independent training/source realizations
and retain pairing across arms; account for shared sources/scenarios across
folds in any aggregate uncertainty estimate.

Choose the number of training seeds using a development pilot, a predeclared
practically meaningful effect, and simulated power or interval precision. Six
seeds do not become adequate merely by increasing evaluation episodes. A
candidate range such as 20–30 training seeds is a planning estimate, not a
guarantee of statistical power, and must be checked before committing compute.

A reliability certification is separate from the comparative success fraction.
For 30/30 independent successes by one frozen controller, the exact one-sided
95% lower bound is approximately 90.5%; the two-sided lower bound is 88.4%.
At 29/30 the one-sided lower bound is only 85.1%. Conversely, a truly 95%-reliable
controller gets 30/30 only about 21.5% of the time, so an all-successes binary
criterion is an inefficient primary comparison. Zero observed violations also
does not establish zero violation risk.

## Implementation prerequisites

- Add checkpoint weights, policy IDs, explicit shot roles, and final policies.
- Add independent scenario-driven frozen-policy evaluation with a non-mutation
  test of optimizer state and training RNG.
- Implement and test sham exchanges and the matched-total single-source arm.
- Align source-region and safety-rate sampling windows; preserve full-trajectory
  information if claims concern transient operating regimes.
- Save every source and target shot, source membership, update/adoption records,
  seeds, environment versions, and code hashes.
- Dry-run all arms against simple deterministic mock plants before an expensive
  simulator campaign. Keep regression tests separate from claims about physics.

The running study remains useful as pilot evidence about learning curves and
compute cost. Its raw logs permit checks excluding selection trials and checks
for later failures, but cannot recover unrecorded policy weights for independent
frozen-policy validation without replaying training.

Methodological reference: Agarwal et al., *Deep Reinforcement Learning at the
Edge of the Statistical Precipice*, NeurIPS 2021, documents uncertainty from
few training runs and recommends interval-based evaluation:
https://proceedings.nips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
