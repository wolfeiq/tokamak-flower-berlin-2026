# Fusion app and tokamak research

## Fusion app — separate from the experiments

The combined **Fusion Investigator + THERMAL** app lives in [`flower-app/`](flower-app/README.md).
It combines the live frontend, Flower investigator, facility stewards, disclosure
gateway, reduced transport profiles, and evidence replay in one application.

This repository keeps the following components separate:

| Project | Location | Purpose |
|---|---|---|
| **Fusion app: Fusion Investigator + THERMAL** | [`flower-app/`](flower-app/README.md) | Between-experiment investigation, controlled cross-site evidence, radial profiles, and an independent toy sandbox |
| **Presentation** | [`presentation/`](presentation/README.md) | Separate browser deck, speaker script, 3D geographic atlas and reactor assembly; no run-submission APIs |
| **3D visualization / CAD viewer** | `app.py`, `assets/3d/`, [viewer notes](visual-additions.md) | Separate visualization and geometry-export work |
| **Original TORAX/control experiments** | `hfmarl/`, `scripts/`, `configs/`, root `tests/` | Simulation studies and cold-start/control experiments described below |

The Flower app has its **own environment, lockfile, tests, and launch instructions**.


**Open the frontend:** follow the [app quick start](flower-app/README.md#local-frontend).
**Open the presentation:** `python presentation/serve.py` then visit
<http://127.0.0.1:8788/>. The application remains on port 8787 with ordinary
navigation. Presentation files and slide controls live only in `presentation/`.
Source code is included; runtime logs, account credentials, virtual environments,
and new experiment outputs are not part of the app commit.

---

# HFMARL for research tokamaks

Federated multi-agent RL for tokamak control, simulated in
[TORAX](https://github.com/google-deepmind/torax) (DeepMind's differentiable
1-D core transport solver, JAX).

**The claim** (SPEC.md §1): federating role-matched control agents across
research tokamaks makes them learn faster and avoid catastrophic regimes they
have never personally entered — without any device sharing raw discharge data.
The second half is the one that matters: large tokamaks cannot generate
training disruptions because they cannot tolerate them, so a device must learn
its limits from devices that have already crossed them.

This is a **simulation study**, not a control system. TORAX's fidelity limits
(1-D, core only, transport-level) bound what can be concluded.

**The `H` in the name is now historical.** The three-level control hierarchy
(per-actuator agents under cluster heads under a device head) was removed on
2026-09-15 — see SPEC.md §2 and the FINDINGS.md entry "Design change: the
three-level hierarchy is removed". What remains is a flat team of agents per
device, one per cluster, federated role-matched. The repository has not been
renamed; that is a separate decision, since it touches the package name, the
remote and every import.

---

## Status — read this before trusting anything

| | |
|---|---|
| Physics, metrics, plots, limits, actuator logic | **tested** — 365 tests |
| TORAX environment wrapper | **executed** — gates 0a/0b pass on TORAX 1.4.3 |
| Gate 0d (actuator authority) | **passes 4/4 devices** (failed 3/4 before the audit) |
| Phase 1 gate | **passes 4/4 devices** on `moderate` |
| Identification (gate + recovery) | **runs end to end**; see the caveats below |
| Phases 2–3, 5–6 | scaffolded, not implemented |
| Phase 7 (exhaust) | **blocked** — see below |

**TORAX 1.4.3 installs and runs natively on Windows, CPU, no WSL.** JAX
resolves to 0.11.1; only the TPU plugin is missing, which is harmless.

**The make-or-break gate passed decisively.** `gate0_jit.py`: 4.4 s to compile,
then **6 ms per step**, spread 1.22×, with the growing-array negative control
570–600× slower. The no-recompilation contract holds, so the environment design
is sound — and a shot is ~8× cheaper than PLAN.md's 0.05 s/step budget, so
re-derive the Phase 5 cost from the measurement rather than the estimate.

### The bug the gates found, and the fix

`gate_authority.py` (added by the audit) failed **3 of 4 devices**: the task
setpoint was unreachable on ITER, SPARC and TCV. Root cause was one line of
normalisation missing. Density was already specified as a Greenwald *fraction*
so the devices start comparable; temperature was a hard-coded 1.0 keV pedestal
and 6.0 keV core for all four. Since

> β_N ~ f_GW · T / (a·B₀)   once density is a Greenwald fraction

one absolute temperature put four devices at normalised pressures spanning a
factor of 30. `tcv_like` started every episode already past the β limit — every
policy scored *exactly* −550.00, a perfectly flat landscape — and `iter_like`
saturated at full power 4× short of target.

Three fixes:

1. **Pedestal temperature scales as a·B₀.** Anchored on ITER's real 4.5 keV, it
   reproduces every other device's published pedestal with no per-device tuning
   (SPARC 2.95, DIII-D 0.57, TCV 0.15 keV) — which is the reason to believe it
   rather than the convenience of it.
2. **`sparc_like` got its heating back.** Its NBI limit was 0.1 MW by design
   ("SPARC baseline heating is ICRF") and ICRH was `available=False` because
   1.4.3 has no independent ICRH source. Both decisions were individually right;
   composed, they left the device with 5.1 MW against B₀ = 12.2 T and 0.012 of
   β_N authority. The `generic_heat` channel is now `aux_heat` — *the device's
   primary auxiliary heating*, NBI on three devices and ICRF on SPARC — which is
   also what makes role-matched federation honest, since channel 0 then means
   the same thing everywhere.
3. **Setpoint *and tolerance* are fractions of each device's measured band.**
   The tolerance half matters as much as the setpoint half: 0.15 in β_N was 7.5%
   of DIII-D's control authority and 197% of SPARC's, so even a reachable
   setpoint would have made the task a different difficulty on every device —
   and shots-to-threshold would have measured device calibration rather than
   learning, which is exactly what SPEC.md §5 compares across the federation.

Result: gate 0d passes 4/4, and so does Phase 1. TCV went from a flat −550
landscape to a real learning curve (−215 → −102 → −0.5 → 0, violations
40% → 0%); DIII-D and TCV beat the best fixed-power baseline by more than 10×.

Bands live in `devices/registry.BETA_N_BANDS`, keyed by **task as well as
device** — the transport model moves them, and `iter_like` has 0.444 of β_N
authority under `constant` against 0.011 under `qlknn`. That is profile
stiffness doing what profile stiffness does (above the critical gradient extra
power raises transport, not temperature), it is physics rather than a bug, and
the gate reports it rather than hiding it.

### Still open

- **Headroom is unmeasured.** `gate_headroom.py` has not been run. Phase 1
  passing quickly on every device is a hint the presets may be too easy, which
  is precisely the blocking question that gate exists to answer.
- **`OPERATING_POINTS` is still the nominal table**, not TORAX output, so the
  similarity weighting is computed from hand-written numbers. RUNBOOK's
  "Before Phase 5" prompt says to re-derive it from real runs; that has not
  been done.
- **0 of 4 candidate unknowns are identifiable** at 10% — Z_eff and
  `resistivity_multiplier` are collinear at |r| = 0.998. The CRLB screen then
  correctly *predicted* which parameter would fail recovery.
- The inverse PINN still has not beaten the classical χ estimator.

## Start here

**`RUNBOOK.md`** — what to run on a new machine, in order, with what a pass
looks like and what to do when it fails.

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'

uv run pytest                                          # 197 tests, no TORAX
uv run python scripts/describe_devices.py              # no TORAX
uv run python scripts/demo_figures.py                  # no TORAX — see the figures

uv run python scripts/gate0_env.py                     # needs TORAX
uv run python scripts/gate0_jit.py --negative-control  # needs TORAX
uv run python scripts/gate_authority.py --plot         # needs TORAX — blocking
uv run python scripts/gate1_thermal.py                 # needs TORAX
uv run python scripts/gate_headroom.py                 # needs TORAX — blocking
```

Full instructions, including the two WSL mistakes that cost an afternoon each:
**SETUP.md**.

---

## Layout

```
SPEC.md          the research spec, verbatim and unedited
RUNBOOK.md       <- START HERE on a new machine: step-by-step, in order
METRICS.md       the deliverables: statement, six metrics, six figures
PLAN.md          phase-by-phase execution plan; every gate is a number
SETUP.md         WSL2 + CUDA + uv install
TORAX_NOTES.md   what we learned reading TORAX 1.4.3; read before touching the env
FINDINGS.md      gate results, appended by the scripts

hfmarl/
  devices/       4 tokamaks as circular-geometry configs + actuator envelopes
  physics/       dimensionless encoder (rho*, nu*, beta, q) -- pure NumPy
  envs/          TORAX wrapper, actuator ramps, operating limits, task difficulty
  agents/        small policies + CEM trainer
  federation/    similarity distance, staleness, role-matched aggregation
  metrics/       shot accounting, the six metrics, the six figures
  experiments/   the five conditions, headroom assessment
  identification/ per-device physics ID: closure-free chi(rho), CRLB screen,
                 scalar recovery by differentiating through TORAX
scripts/         gate0_{env,jit,bench}, gate_authority, gate_headroom,
                 gate1_thermal,
                 gate_identify, exp_recover_scalars, exp_identify_chi,
                 plot_torax, describe_devices, demo_figures
tests/           runs without TORAX, without a GPU, in ~2 seconds
```

---

## The deliverables

`METRICS.md` is the output contract: the statement, six metrics, six figures.

**The currency is shots** — one discharge, one episode. Not wall-clock, not
gradient steps. That is what costs machine time and what a fusion audience
reads, and it is enforced structurally: there is no code path that plots a
learning curve against gradient steps.

**What kills the claim is a blocking gate.** If isolated training reaches
threshold in a few hundred shots, there is no headroom and every run in the
Phase 5 matrix measures nothing. `scripts/gate_headroom.py` trains
isolated-only at increasing difficulty and will not green-light Phase 5 until
headroom, measurability and affordability hold together. Task difficulty is a
real knob (`hfmarl/envs/task.py`), five presets from `trivial` to `brutal`.

**The statistics are built not to flatter.** Runs that never converge are
right-censored and counted, never dropped — and `speedup_ratio` refuses to give
a clean number when two conditions are censored at different rates. Smoothing
is strictly causal. A threshold crossing must persist. The catastrophe claim
requires *zero* federated violations, not merely fewer; anything less reports
as PARTIAL. Details and rationale in `METRICS.md`.

---

## Three things worth knowing up front

**1. The spec's make-or-break gate moved.** SPEC.md Phase 0 gates on "does JIT
survive the Gym wrapper". It does — TORAX ships a public `torax.experimental`
API built for stepped use, its runtime params are traced JAX leaves, and the
TORAX lead endorsed this exact call sequence for RL in
[discussion #1625](https://github.com/google-deepmind/torax/discussions/1625).
Gym-TORAX is a published, working existence proof.

So the real make-or-break is **Phase 4**: normalised transfer must beat raw-SI
transfer. If it does not, the physics layer is not earning its place and the
federation story collapses.

**2. The device set overlaps in dimensionless space — by physics.** ρ* spans a
decade (ITER-like 1.5×10⁻³ → TCV-like 1.7×10⁻²), yet SPARC-like sits *adjacent*
to ITER-like despite being a third the size, because its field is 2.3× higher.
That is the Connor–Taylor overlap Phase 6 needs, and it is not an artefact of
how the configs were chosen. `scripts/describe_devices.py` prints it.

**3. Two things in the spec are not buildable as written, and are flagged
rather than fudged.**

- *Three thermal agents (NBI/ECRH/ICRH).* TORAX 1.4.3 has **two** independently
  drivable auxiliary heat sources. ICRH is defined but marked unavailable so it
  cannot silently alias onto NBI's config path and halve the commanded power
  with no error. Phase 2 runs with two agents.
- *Phase 7 divertor detachment* — SPEC.md §7's "strongest candidate" use case —
  is **blocked**. TORAX refuses edge/SOL models on circular geometry, and
  circular is the only file-free geometry. It needs real equilibria (FBT/EQDSK)
  for several devices. There is also **no SPARC equilibrium in existence** to
  use; "SPARC-like" here means a compact high-field circular plasma with
  SPARC's R, a, B₀ — not SPARC.

---

## The federated payload is 776 bytes

194 parameters, float32, for the Phase-1 thermal policy: a 9-dim observation,
16 hidden units, 2 thermal actuators. SPEC.md §8 asks for this to be stated
because it removes the standard bandwidth objection to federated RL. It is
measured, not asserted: `MLPPolicy.payload_bytes()`, tested in
`tests/test_policy.py`.

(It read 162 parameters / 648 bytes until the audit. That was the 7-dim
observation; `_observe` later gained the tracking error and the current target
so a moving setpoint is visible to the policy at all, and the stated figure was
never updated. The test only asserted "< 4096 bytes", so nothing caught the
drift -- it now pins the exact count.)

---

## Credit

The two-breakpoint actuator ramp and the `numerics.t_final` window bookkeeping
— the non-obvious mechanics that keep TORAX JIT-compiled across steps — come
from **Gym-TORAX** (Mouchamps et al., MIT licence,
[arXiv:2510.11283](https://arxiv.org/abs/2510.11283), *Software Impacts*). We
build our own environment because its abstractions are single-agent and
ITER-hybrid-shaped, but it got those two details right and we copy them.

TORAX is Apache-2.0, Google DeepMind ([arXiv:2406.06718](https://arxiv.org/abs/2406.06718)).

## Cross-facility thermal investigation (Flower AgentApp)

THERMAL is now merged into the **[Fusion app](flower-app/README.md)**.
Use that folder for the frontend, agent, disclosure gateway, tests, and setup.
The old `apps/thermal-investigation/` path is a migration pointer.
The original TORAX/RL experiments and their results remain separate.
