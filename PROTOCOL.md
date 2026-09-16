# What a new reactor is allowed to know

This document exists because the cold-start claim is meaningless without it.
"A new machine reaches competence in N shots" is a statement about what the
machine started with, and until that is written down, any N can be defended.

The audit (`AUDIT.md`, 2026-09-16) put it directly: `TaskSpec.resolve_for`
requires the new device's `BETA_N_BANDS`, which are *measured from that
device's own simulator responses at zero and full command*. So the experiment
as it stood had already given the newcomer target-derived information while
describing it as having none.

---

## 1. The setting, stated once

**Claim under test.** A tokamak that has never fired a training discharge
reaches competent, safe control in fewer of its own shots by inheriting a
controller from machines that have, than by any alternative available to it.

**Currency.** Shots on the *new* device. Every shot fired on it counts,
whatever it was for. Shots fired on source devices are a separate quantity —
they are the historical investment the newcomer inherits, and pretending they
are free is as dishonest as pretending they are the newcomer's.

Three budgets, tracked separately and never summed into one number:

| budget | what it is |
|---|---|
| **calibration** | shots on the new device used to characterise the plant before any controller is installed |
| **training** | shots on the new device used to improve a controller |
| **evaluation** | shots on the new device used to measure a frozen controller |

A result that moves cost between these columns is not a result.

**The tolerance carries a hidden calibration cost, and it caught me out.** The
joint endpoint needs a tolerance; on a `band_fraction` task that tolerance is a
fraction of the device's *measured* `BETA_N_BANDS`, which costs a five-shot
sweep per device. That sweep is upstream of every experiment script, cached in
the registry and shared by every arm, so it is easy to omit — I reported the
classical arm as costing two calibration shots when the true figure is seven.
Every arm pays it, so no comparison between arms changes; but "a new machine
needs N shots" is a claim about the total, and the total must include it.
Report it once per experiment, not per arm. A `floor_multiple` tolerance does
not escape this — it needs a measured open-loop reference, which costs the same
sweep.

---

## 2. Allowed, refused, and why

We adopt the audit's **claim 1**: *no target training shots, with a design
model available*. It is the honest description of what this simulation study
can support, and it is also the realistic one — nobody commissions a tokamak
without a design.

### Allowed without charge (design information)

Known before the machine ever runs, from engineering drawings and the design
point:

- geometry: `R_major`, `a_minor`, `B_0`, `Ip_nominal`, aspect ratio, `q_cylindrical`
- actuator inventory and envelopes: which sources exist and their power limits
- the simulator itself, including transport model and solver settings
- the operating limits: Greenwald, Troyon-normalised beta, `q95` — these are
  physics, not measurements of this machine
- everything about the **source** devices, including their trained controllers

### Allowed but charged as calibration

Requires firing the new machine, so it is counted:

- the command-to-`beta_N` static map (the sweep `gate_authority` performs)
- `BETA_N_BANDS` for this device — **this is the one the audit caught**. It is
  measured, so from now on it is either charged as calibration or derived from
  design information, never silently assumed
- the device's dimensionless operating region, which `torax_adapter` measures
  and the federation weighting needs

### Refused

- any discharge from the new device used to *fit* a controller before the
  experiment's training budget starts
- tuning any hyperparameter — bandwidth, clipping factor, rejection threshold,
  smoothing window — on the held-out device. These are chosen on the source
  devices or on a separate development device, and the choice is recorded
  before the held-out run.

  **Tuning means optimising an outcome, not evaluating a fixed rule.** The
  kernel bandwidth is set by the median heuristic over the whole device set,
  the joiner included, and that is deliberate: `suggest_bandwidth` is a stated
  rule with no reference to any result, and computing it over the incumbents
  alone would move it by 6–16% from fold to fold (1.1024 to 1.3644 across the
  four leave-one-out folds, against 1.1760 for the full set), changing the
  kernel width between folds for a reason that has nothing to do with the arms
  being compared. The joiner's coordinates are charged as calibration under the
  clause above, like any other measurement of the new machine. What is refused
  is picking a bandwidth because it made a number look better

### Explicitly permitted, and not a loophole

**Observing the plasma while controlling it.** The controller reads the state
during a shot; that is what a controller is. "Zero previous training data" and
"no live measurements" are different settings and conflating them would make
the experiment a different, sillier one.

---

## 3. The baselines, and what each one rules out

The audit's central point: three arms — scratch, uniform federation, similarity
federation — cannot attribute a win to federation, because every federated arm
also enjoys pretraining, multiple sources, repeated exchange, and similarity
weighting at once. Each control below removes exactly one of those.

| arm | what it has | what a win over it proves |
|---|---|---|
| **scratch** | nothing but its own shots | that having *anything* helps |
| **conventional** | design info + calibration, no learning at all | that the learned controller beats classical control on the same information — without this, the whole RL framing is unearned |
| **single-source** | one source device's trained controller | that *multiple* sources matter, not merely a warm start |
| **handover-merge** | all sources, merged once at handover, *uniform weights* | that repeated federation matters, not merely combining final models |
| **fedbuff-uniform** | all sources, repeated exchange, equal weights | that federating during training matters |
| **fedbuff-similarity** | the above, plus physics weighting | that SPEC.md §4b earns its place |

Read down the column: each row adds one ingredient. A gap between two adjacent
rows is attributable to the ingredient that differs, and to nothing else.

**This property was false in the code until 2026-09-16, at exactly the rung it
was written for.** `handover_merge` built its server with similarity weighting
while `fedbuff-uniform` weights uniformly, so climbing from one to the next
*removed* the physics weighting rather than adding anything; and the merge
published no violation rates, so it also escaped the safety downweighting
every federated update pays. Measured weights for a `tcv_like` joiner:

    handover_merge      diiid 0.558  sparc 0.232  iter 0.210
    federated_uniform   iter  0.377  sparc 0.377  diiid 0.245

Three differences in one comparison. Both are fixed and pinned by
`tests/test_ladder_integrity.py`; every cold-start result recorded before that
date compares those two rungs invalidly.

**"Equal weights" means equal before the safety factor.** Every federated
update is downweighted by `(1 - violation_rate)`, and two of the four devices
cannot cross a limit at any command, so their violation rate is structurally
zero. Uniform weighting is therefore uniform only among devices with equal
headroom: in the row above, the two devices that *cannot* be unsafe hold 0.377
each and the one that can holds 0.245. The safety rule is measuring plant
headroom as much as controller behaviour, and it penalises DIII-D — the only
source whose experience of the limit is relevant to a TCV joiner — for having
had that experience.

**The conventional controller is the one that matters most and is easiest to
skip.** If a PI controller built from the calibration sweep matches the
federated policy, then this is a control-engineering problem with an RL
solution bolted on, and the honest paper says so.

**And it is bound by this document's own rule.** Section 2 refuses
hyperparameter tuning on the held-out device. The conventional arm's
proportional gain is a hyperparameter, so it is chosen on the source devices
like any other. Measured on `moderate`, that changes the answer completely:

| device | gain chosen here | gain chosen on the sources |
|---|---|---|
| diiid_like | 100% | 100% |
| iter_like | 100% | 100% |
| **sparc_like** | **100%** | **0%** |
| tcv_like | 100% | 100% |

Three devices score 100% at every gain from 0.05 to 0.9, so the sources' choice
is settled by the error tie-break, which prefers a high gain. SPARC is the only
device where that fails — its tolerance is 26× tighter than DIII-D's and its
actuator gain 80× weaker than TCV's — and nothing observable on the other three
predicts it.

So two conventional numbers get reported, never one: the **tuned-on-target
ceiling**, which is what classical control can do, and the **transferred**
number, which is what it costs to arrive with. A cold-start comparison uses the
second, because the joiner has no shots to tune with — that is what makes it a
cold start.

---

## 4. The endpoint

A shot counts as a success only if all three hold:

1. **completed** — ran to the end of the episode;
2. **contained** — crossed no limit, solver failures included;
3. **tracked** — mean `|beta_N - target|` inside that device's own tolerance.

Tracking alone would let a controller that crashes near its setpoint look
excellent, because the error is averaged over surviving steps. Containment
alone would let a controller that never drives the plant look safe. The
conjunction is the only form of the endpoint that cannot be gamed by doing
nothing.

**On two of the four devices the conjunction is not doing that work, and the
claim above has to be read with this.** Driving each device from zero to full
thermal command and taking the worst margin over the episode:

| device | worst beta_N margin at full command | can it cross a limit? |
|---|---|---|
| iter_like | +4.17 | no |
| sparc_like | +5.04 | no |
| diiid_like | -0.38 | yes, above ~60% command |
| tcv_like | -7.48 | yes, above ~40% command |

On `iter_like` and `sparc_like` no action sequence available to the controller
can fail containment — they sit four to five soft-to-hard bands clear of the
Troyon limit at maximum power. There, the endpoint is *completed + tracked*,
and a result on those devices says the controller tracked; it does not
additionally say the controller was safe, because nothing there is unsafe.

q95 and the Greenwald fraction do not move with command on any device: `Ip`
lives in the `current` cluster, which no preset actuates, and the density is
pinned by the pedestal and the edge boundary condition. So the three-limit
envelope is in practice a one-limit envelope, and that limit is reachable on
two devices.

Anything that needs containment to be a live constraint — the catastrophe
transfer, the `brink` task, any safety claim at all — belongs on `diiid_like`
or `tcv_like`. `exp_catastrophe.py` now refuses a target that cannot cross a
limit rather than reporting a perfect score for a question it never asked.

Reported at a **fixed shot budget**, as a reach rate with censored runs
retained — not as a ratio over the seeds that happened to succeed. A ratio
between survivors compares different populations, and with few seeds it reports
a precise-looking interval that has nothing to do with the effect.

---

## 5. What a positive result here would and would not mean

**Would:** in a 1-D core transport simulation with circular geometry and
manually imposed limit thresholds, a controller assembled from other simulated
devices reached the joint endpoint on an unseen simulated device in fewer of
its shots than the alternatives above.

**Would not:** that disruption avoidance transfers on real machines; that the
similarity coordinates are the right ones for real plasmas; that any of this is
deployment-ready. TORAX's scope bounds the claim, and the devices here are
nominal configurations, not the machines they are named after.

The 1000-shot headroom floor in `METRICS.md` is a study-design choice, not a
law. For cold start the relevant effect can be small: avoiding a handful of
commissioning shots on a machine that fires 20-40 a day is worth having. The
effect size that would count is stated before the run, not after it.
