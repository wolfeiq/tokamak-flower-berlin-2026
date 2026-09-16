# HFMARL for Research Tokamaks — Build Spec

**Executor:** Claude Code
**Simulator:** TORAX (Google DeepMind, Apache-2.0, JAX, differentiable)
**Federation:** Flower, FedBuff aggregation
**Scope:** research prototype in simulation. Not a control system.

---

## 1. The claim

Federating role-matched control agents across research tokamaks makes them
learn faster and avoid catastrophic regimes they have never personally
entered — without any device sharing raw discharge data.

The second half is the one that matters. Large tokamaks cannot generate
training disruptions, because they cannot tolerate them. A device must
therefore learn its limits from devices that have already crossed them.

---

## 2. Architecture

A flat team of agents per device, federated role-matched across devices.

```
                    PHYSICS LAYER (dimensionless encoder/decoder)
                                    |
    DEVICE A                        |                      DEVICE B
  ┌─────┬────────┬─────┐            |            ┌─────┬────────┬─────┐
  │therm│particle│curr │            |            │therm│particle│curr │
  └─────┴────────┴─────┘            |            └─────┴────────┴─────┘
   one agent per cluster,           |             same, per device
   shared team reward               |

  FEDERATION CHANNELS (FedBuff, role-matched):
    thermal(A)  <--->  thermal(B)       thermal never mixes with particle
    particle(A) <--->  particle(B)
    current(A)  <--->  current(B)       ...
```

**Agents.** One per cluster, owning that cluster's actuators. Small dense
policy nets; PACMAN's deployed controllers infer in 0.2–15 ms and these
must stay in that range. Cross-cluster conflict is resolved by the shared
reward, the way it is in ordinary multi-agent RL — there is no arbitrator.

**Federation.** Role-matched only. Thermal agents aggregate with thermal
agents across devices; never with particle or current agents. Different
action semantics means mixed gradients are meaningless, and this is the
mechanism that prevents it.

**There is no control hierarchy, and that is a deliberate removal.** This
spec previously called for cluster heads (level 2) setting targets for
per-actuator agents, under a device head (level 3) resolving cross-cluster
contention. It was removed on 2026-09-15; FINDINGS.md, "Design change: the
three-level hierarchy is removed", records the reasoning and the evidence.
The short version is that the plant cannot exhibit the problem a hierarchy
solves: two live actuators, one scalar objective, and an isolated flat
policy that reaches threshold in 240 shots.

**The condition for bringing it back** is specific, so this is a decision
rather than an abandonment. Reinstate a coordinator when Phase 2 shows two
agents driving the same actuator toward incompatible targets in a way the
shared reward demonstrably cannot resolve — measured as a flat team that
loses to a monolithic controller with the same actuator set. Until that
measurement exists, a coordinator has nothing to coordinate.

---

## 3. Cluster decomposition

Derived from TORAX's own coupled equations, not invented. A cluster is a
functional grouping of actuators and the federation channel its agent
belongs to. It is **not** a command layer -- nothing sets targets for
anything else.

| Cluster | State | Actuators | TORAX equation |
|---|---|---|---|
| Thermal | Ti, Te profiles | NBI, ECRH, ICRH power | ion + electron heat transport |
| Particle | ne profile, Greenwald fraction | gas puff, pellets, pumping | particle transport |
| Current | q profile, li, non-inductive fraction | ECCD, NBCD, ohmic, Ip | current diffusion |
| Stability | tearing, ELM, Alfvén activity | localized ECCD, RMP, pellet pacing | (derived limits) |
| Exhaust | divertor heat flux, detachment | impurity seeding, strike-point sweep | (outside TORAX 1D core — see §7) |

Start with thermal + particle + current. These map directly onto what TORAX
solves. Stability and exhaust are Phase 5+ -- and note that TORAX 1-D core
cannot represent either, which is why the old device-head argument ("thermal
wants more NBI, stability wants less") had no venue in this simulator.

---

## 4. The physics layer — this is the core contribution

Naive problem: device A is small (B ≈ 2 T, R ≈ 1.6 m), device B is SPARC-like
(B ≈ 12 T). Both have thermal agents. Their observations are SI quantities that
mean different things. Role matching is necessary but not sufficient — averaging
those weights is still meaningless.

Fusion's own foundation solves this. Tokamak transport is scale-invariant in
dimensionless parameters: ρ\* (normalized gyroradius), ν\* (collisionality),
β, q, Mach number. Connor–Taylor similarity holds that plasmas matched in these
behave identically regardless of absolute size. The entire field's extrapolation
to ITER rests on it.

The physics layer therefore does **two** jobs:

### 4a. Dimensionless encoder / decoder
- Encoder: local SI plasma state → dimensionless state vector.
- All agents observe and act in dimensionless space.
- Decoder: dimensionless action → device-specific SI actuator command.
- **Physics-informed:** encoder trained with a residual loss on the normalized
  transport equations, so the shared latent space is physically consistent by
  construction rather than merely learned.

Without this, cross-device federation is not meaningful. This is the component
that makes the whole system possible.

### 4b. Physics-informed aggregation weights
FedAvg weights clients by sample count — an arbitrary rule with no physics in
it. Replace with weighting by **distance in dimensionless parameter space**.
A device operating near your (ρ\*, ν\*, β, q) contributes strongly; a device
far away contributes weakly.

This is physically justified by similarity theory rather than invented, and no
prior work found does it. Nearest relatives: inertia-weighted FedAvg for grid
stability (arXiv, 2026) uses a static per-node physical property; graph-based
aggregation in traffic forecasting uses network adjacency. Neither uses a
similarity metric derived from scale invariance.

---

## 5. Why FedBuff, not FedAvg

Not a preference — synchronous rounds are physically impossible here.

- Tokamaks run in **shots**: ~20–40/day, each seconds of plasma.
- Devices are on different campaigns; some are down for months of maintenance.
- Updates arrive whenever a device happens to be running.

FedBuff's asynchronous buffer with staleness weighting is the only thing that
fits. **Extension:** make staleness physics-informed too. An update from a
device that has since changed wall material, divertor geometry, or heating
configuration is stale in a way wall-clock time does not capture.

---

## 6. Build phases

### Phase 0 — Environment
- Install TORAX, run `examples/basic_config.py`, then the ITER-inspired config.
- Wrap as Gymnasium env. **Confirm JIT survives the wrapper** — if every step
  recompiles, the project is too slow to proceed. Gate on one episode of a
  fixed policy in a few seconds.
- Define 3–4 device configs: ITER-like, SPARC-like, and 1–2 smaller / different
  aspect ratio. These configs are the federation clients and the heterogeneity
  is an experimental variable. Keep them in version control.

### Phase 1 — Single agent, single device
- One thermal agent controlling heating power, tracking normalized pressure.
- Implement limits: Greenwald density fraction, beta limit, q constraint.
- Gate: tracks setpoint, respects limits.

### Phase 2 — Multiple clusters, one device, flat
- Add particle (`gas_puff`) and current (`Ip`) agents alongside the thermal
  one. One agent per cluster, no heads, shared team reward.
- The task must be genuinely multi-objective — tracking β_N alone gives the
  particle and current agents nothing to disagree about, and a coordination
  experiment with nothing to coordinate is what §2 just removed.
- Gate: the flat team matches or beats a single monolithic controller with
  the same actuator set. If it loses, report that and federate the monolith
  instead — the federation claim does not depend on multi-agent decomposition.

### Phase 3 — Does anything need coordinating?
- This phase exists to answer the question §2's removal left open, and it is
  allowed to conclude "no".
- Measure, on the Phase 2 task: do two agents drive the same actuator toward
  incompatible targets? Does the shared reward resolve it, or does the team
  oscillate? Report the actuator-level conflict rate, not an impression.
- Gate: **either** conflict is measurably unresolved, which reinstates a
  coordinator with a number behind it, **or** it is not, and the flat team
  is the architecture. Both outcomes close the phase.

### Phase 4 — Physics layer
- Build dimensionless encoder/decoder with transport-residual loss.
- Validate: an agent trained on device A, transferred zero-shot to device B via
  the encoder, should outperform the same agent transferred on raw SI state.
- **This gate is the make-or-break.** If normalized transfer does not beat raw
  transfer, the physics layer is not earning its place and the federation story
  collapses. Do not proceed past a negative result here without rethinking.

### Phase 5 — Federation
- Flower, FedBuff, role-matched channels.
- Aggregation weights from dimensionless distance (§4b).
- Baselines, all on identical seeds:
  1. isolated per-device training (no federation)
  2. FedBuff, role-matched, uniform weights
  3. FedBuff, role-matched, dimensionless-similarity weights ← the method
  4. naive FedAvg across all agents regardless of role (should be worst —
     this demonstrates why role matching matters)
  5. centralised training on all devices' data (upper bound)
- Metrics: sample efficiency to target performance, and per-device final
  performance.

### Phase 6 — The catastrophe claim
- Restrict device A's training to a safe operating envelope. Let devices B and
  C explore regimes that trigger limit violations.
- Test device A in the regime it never trained in.
- **Claim holds if** federated device A avoids the violation and isolated
  device A does not.
- This is the headline experiment. Design Phase 0 configs with it in mind —
  the devices must have enough overlap in dimensionless space for transfer to
  be possible at all.

### Phase 7 — Extensions
Exhaust cluster (divertor detachment), stability cluster (tearing mode
avoidance), campaign-level slow agent.

---

## 7. Use cases to target beyond PACMAN's five

PACMAN (DIII-D, arXiv 2511.08818) demonstrated: RL heating control, tearing-mode
avoidance, ELM prediction, Alfvén-eigenmode suppression, profile MPC.
Underexplored and well suited to this architecture:

- **Divertor detachment control.** Strongest candidate. Divertor heat flux is
  the hard engineering limit, melting is a genuine catastrophe, physics is rich
  (2-point model, SOL transport), and divertor geometry differs sharply between
  devices — a real test of whether normalized federation transfers. Requires
  coupling TORAX to a SOL/divertor model; check current TORAX scope first.
- **Impurity and radiation control** — tungsten accumulation, seeding for
  divertor protection.
- **Runaway electron mitigation** and disruption-mitigation trigger timing.
- **Sawtooth pacing**, pellet fueling efficiency, wall recycling.
- **Campaign-level planning** — a very slow agent choosing which shot to run
  next given what the device has learned.

---

## 8. Engineering notes

- Small networks throughout. Anything large contradicts the millisecond
  control-loop premise. Federated payloads will be kilobytes — state this, it
  removes the standard bandwidth objection to federated RL.
- Log full state trajectories per episode. Cross-device comparisons cannot be
  reconstructed afterwards.
- Seed everything and run multiple seeds. Sample-efficiency claims are trend
  claims and will not survive single runs.
- Clock separation between levels must be explicit, not emergent.

---

## 9. Objections to have answers for

**"Why multi-agent instead of one multi-objective controller?"**
The field's default is single-agent multi-actuator (KSTAR's multi-objective
DDPG/SAC with a PINN Grad-Shafranov solver). Answer structurally: incompatible
decision rates, controllers built by different groups in different algorithm
families, independent development and swapping (PACMAN's stated modularity
benefit), and across devices, different owners. These are reasons a centralised
controller is *unavailable* — not a claim that decomposition performs better.

**"Why federate when fusion labs already share data?"**
Public labs do — DisruptionBench pools C-Mod, DIII-D and EAST. Private fusion
(CFS, TAE, Helion, Tokamak Energy) will not. The need is arriving, not
established, and the spec should say so honestly.

**"Isn't this just cross-machine transfer learning?"**
Cross-machine transfer for disruption prediction is an established subfield
(J-TEXT→EAST, parameter-based transfer in Communications Physics). Every paper
in it pools the data, all of it is prediction rather than control, and none is
multi-agent. Cite them up front.

---

## 10. Prior art to verify before writing

Web search was used to scope this; it is not a literature review. Check in
Scopus / Web of Science, and in plasma venues (Nuclear Fusion, PPCF) plus
AAMAS / L4DC:

- MARL for tokamaks. None found — field appears to be single-agent
  multi-objective. This is load-bearing for novelty; confirm it.
- Federated learning in fusion. None found. Confirm.
- Dimensionless-similarity-weighted federated aggregation. None found.
  Nearest: inertia-weighted FedAvg (power grids), graph-adjacency aggregation
  (traffic forecasting).
- Read arXiv 2511.08818 (PACMAN) properly. Determine whether any single
  discharge ran more than one model concurrently — that fact changes how this
  work positions itself relative to them.

---

## 11. Honest scope

DIII-D is a research device; no fusion reactor exists. The PACMAN authors state
their own work does not demonstrate reactor operation and is unsuitable for
sub-millisecond events. This project is a simulation study in TORAX's 1D core
transport model. Claims beyond that are unsupportable, and TORAX's own fidelity
limits (1D, core only, transport-level) bound what can be concluded.
