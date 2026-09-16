# Build plan

SPEC.md is the research argument and does not change. This file is the
execution plan, with every gate stated as a **pass/fail on a number** so that a
negative result is recognisable rather than negotiable.

Status legend: **[done]** built and unit-tested · **[ready]** code written,
needs a live TORAX to validate · **[todo]** not written.

---

## Phase 0 — Environment  **[ready]**

Run on the WSL box, in order (see SETUP.md §4).

| Gate | Script | Pass criterion |
|---|---|---|
| 0a environment | `gate0_env.py` | TORAX imports; all 4 device configs validate; a sim steps; `beta_N`, `q95`, `fgw_n_e_line_avg` all present |
| 0b **JIT survival** | `gate0_jit.py --negative-control` | median(steps 2..N) < 0.1 × step 1, **and** max/min < 3, **and** the negative control is ≥2× slower |
| 0c hardware | `gate0_bench.py` | informational — decides CPU vs GPU for Phase 5 |
| 0d device set | `describe_devices.py` | off-diagonal aggregation weights neither all ≈0 nor all ≈1 |

Gate 0b is framed as *confirm on your hardware*, not make-or-break: TORAX's
`torax.experimental` API is built for stepped use and Gym-TORAX is a working
existence proof (TORAX_NOTES.md). **Run the negative control anyway** — a test
that cannot fail proves nothing.

Devices are defined in `hfmarl/devices/registry.py`: `iter_like`, `sparc_like`,
`diiid_like`, `tcv_like`. All `circular` geometry.

---

## Phase 1 — Single agent, single device  **[ready]**

One thermal agent on heating power, tracking normalised pressure, under the
Greenwald / beta / q limits.

    uv run python scripts/gate1_thermal.py --device iter_like

**Gate (all three):** beats the best constant-power baseline on mean return;
mean |β_N − target| < 0.3; no limit violation in evaluation.

The baseline comparison is the one that matters — a policy that sits still can
score well under a badly shaped reward.

Trainer is CEM (`hfmarl/agents/cem.py`), chosen because it is gradient-free and
dependency-free, so Phase 1 gates without putting PyTorch next to JAX. **It is
not the end state**: SPEC.md §2 specifies PPO or SAC, and Phase 5's
sample-efficiency claims need a real RL algorithm. CEM's job is to prove the
environment is learnable at all.

---

## Phase 1.5 — Headroom  **[ready]**  ← BLOCKING, run before Phase 5

    uv run python scripts/gate_headroom.py --device iter_like

Trains isolated-only at increasing task difficulty and reports whether the task
admits a measurable federation benefit at an affordable cost.

**Gate (all three):** isolated median ≥ 1000 shots (headroom); ≥80% of seeds
converge within 20000 shots (measurable); full matrix ≤ 72 h serial
(affordable).

If isolated converges in a few hundred shots there is nothing for federation to
show, and the entire Phase 5 matrix would measure noise. The gate names the
binding constraint and gives advice specific to it. Difficulty presets live in
`hfmarl/envs/task.py` (`trivial` → `brutal`); the gate escalates automatically.

See METRICS.md for the full rationale.

---

## Phase 2 — Multiple clusters, one device, flat  **[todo]**

Add particle (`gas_puff`) and current (`Ip`) agents alongside the thermal one.
One agent per cluster, no heads, shared team reward. The cluster heads and
device head this phase used to build were removed — see SPEC.md §2 and the
FINDINGS.md entry "Design change: the three-level hierarchy is removed".

**The task has to be multi-objective**, or this phase measures nothing: with
β_N alone as the target, the particle and current agents have nothing to
disagree with the thermal agent about. Track β_N **and** a density target,
with the Greenwald fraction as a live constraint rather than a passive limit.

**Gate:** the flat team matches or beats a single monolithic controller with
the same actuator set. *If it loses, report that and federate the monolith* —
the federation claim does not depend on multi-agent decomposition, and this
is exactly the ablation the old spec never had.

⚠ **Scope correction.** The old Phase 2 wanted 2–3 thermal agents (NBI, ECRH,
ICRH). TORAX 1.4.3 has exactly **two** independently drivable auxiliary heat
sources: `generic_heat` and `ecrh`. There is no third. ICRH is defined in the
registry but marked `available=False` so it cannot silently alias onto NBI's
config path. One thermal agent owning both channels is the shape that fits.

---

## Phase 3 — Does anything need coordinating?  **[todo]**

This phase exists to answer the question the hierarchy removal left open, and
it is allowed to conclude *no*.

On the Phase 2 task, measure whether two agents drive the same actuator toward
incompatible targets, and whether the shared reward resolves it or the team
oscillates. Report an actuator-level conflict rate, not an impression. Agents
running at different decision rates belongs here too — SPEC.md §8 requires the
clock separation be **explicit, not emergent**, so it goes in the scheduler,
not in differing episode lengths.

**Gate:** *either* conflict is measurably unresolved, which reinstates a
coordinator with a number behind it, *or* it is not, and the flat team is the
architecture. Both outcomes close the phase.

---

## Phase 3.5 — Per-device physics identification  **[ready]**

Not in the original spec; it makes Phase 4 and Phase 5 sharper.

Every device has physics you cannot read off a diagnostic — transport
coefficients, Z_eff, resistivity and bootstrap multipliers. Identify them from
that device's own shots, then (a) weight aggregation by distance in *identified*
parameter space rather than analytic dimensionless space, and (b) condition each
policy on its device's parameters, so the policy is "what to do given a plasma
with these transport characteristics" rather than "what to do on device A".

    uv run python scripts/gate_identify.py         # identifiable at all?
    uv run python scripts/exp_recover_scalars.py   # hide a config, recover it
    uv run python scripts/exp_identify_chi.py      # chi(rho), no TORAX needed

**Gate:** TORAX parameters recovered from trajectories to <5%, clean and again
with `--noise 0.02 --subsample 4`.

Method split, deliberately: **scalars by differentiating through TORAX** (exact,
they are traced leaves), **chi(rho) by the classical power-balance integral**
(conservation is certain, the closure is a model, so impose the first and leave
chi free). An inverse-PINN arm was tried and lost — `docs/pinn_comparison.md`.

Run `gate_identify.py` first: several candidates are degenerate with each other
(confinement deviation vs transport multiplier; wall recycling is probably
invisible in a 1-D core model), and fitting a degenerate set returns a confident
wrong number rather than an error.

---

## Phase 4 — Physics layer  **← THE REAL MAKE-OR-BREAK**  **[partial]**

**[done]** The dimensionless encoder (`hfmarl/physics/dimensionless.py`):
ρ*, ν*, β_N, q95, Mach, validated against textbook values (ITER ρ* ≈ 1.5×10⁻³).

**[todo]** The learned encoder/decoder with a transport-residual loss, and the
transfer experiment.

**Gate:** an agent trained on device A and transferred **zero-shot** to device B
*through the encoder* must outperform the same agent transferred on **raw SI
state**.

Phase 0 turned out to be settled in advance; this gate is where the project
actually lives or dies. If normalised transfer does not beat raw transfer, the
physics layer is not earning its place and the federation story collapses.
**Do not proceed past a negative result here without rethinking.** Do not tune
until it passes.

Encouraging sign already visible from `describe_devices.py`: ρ* spans a decade
across the device set, yet `sparc_like` sits adjacent to `iter_like` because
its field is 2.3× higher. The similarity overlap Phase 6 needs exists by
physics, not by construction.

---

## Phase 5 — Federation  **[partial]**

**[done]** Similarity metric, staleness (including the physics-informed config
epoch of SPEC.md §5), and role-matched aggregation weights
(`hfmarl/federation/similarity.py`), with mixed-cluster aggregation raising
rather than silently averaging.

**[done]** Shot accounting, all six metrics, all six figures, and the five
conditions as a single registry (`hfmarl/metrics/`, `hfmarl/experiments/`).
Statistics are censoring-aware — see METRICS.md. Render them on synthetic data
with `scripts/demo_figures.py`.

**[todo]** Flower transport, the FedBuff buffer, and the experiment runner.

**Run the headroom gate (Phase 1.5) first.** Without it the matrix can burn
days measuring a task that was too easy to show anything.

Baselines, identical seeds:

1. isolated per-device training (no federation)
2. FedBuff, role-matched, **uniform** weights
3. FedBuff, role-matched, **dimensionless-similarity** weights ← the method
   (and, if Phase 3.5 passes, a fourth weighting by distance in *identified*
   parameter space -- if that beats analytic similarity, learned deviations
   from ideal scaling carry information algebra does not)
4. naive FedAvg across all roles (should be worst — demonstrates why role
   matching matters)
5. centralised training on all devices' data (upper bound)

Metrics: sample efficiency to target performance; per-device final performance.

**Before running any of it**, check `describe_devices.py`. The bandwidth has
two silent failure modes: too small and baseline 3 collapses onto baseline 1
(every device an island), too large and it collapses onto baseline 2 (uniform).
Either would destroy the experiment without producing an error.

Federated payload is **648 bytes** for the Phase-1 policy (162 params, float32)
— state this, it removes the bandwidth objection of SPEC.md §8 with a measured
number.

---

## Phase 6 — The catastrophe claim  **[todo]**

The headline experiment.

- Restrict device A to a safe envelope (`LimitSet.restricted()`, **[done]**).
- Let B and C explore regimes that trigger violations.
- Test A in the regime it never trained in.

**Claim holds if** federated A avoids the violation and isolated A does not.

Both arms must be run at multiple seeds. This is a trend claim and will not
survive a single run (SPEC.md §8).

---

## Phase 7 — Extensions  **[blocked]**

**Exhaust / divertor detachment is blocked, not merely unimplemented.** TORAX
refuses edge models on circular geometry
(`_check_edge_with_circular_geometry` raises), and the only bundled equilibria
are ITER-hybrid and STEP. SPEC.md §7 calls divertor detachment the strongest
candidate use case; reaching it requires obtaining real equilibria (FBT or
EQDSK) for several devices first. Budget for that or descope honestly.

Stability cluster (tearing-mode avoidance) and a campaign-level slow agent are
not blocked, just unbuilt.

---

## Standing engineering constraints (SPEC.md §8)

- Small networks throughout; report payload size in bytes.
- Log full state trajectories per episode — cross-device comparisons cannot be
  reconstructed afterwards. (`StepResult` in `envs/torax_env.py`.)
- Seed everything; multiple seeds for any sample-efficiency claim.
- Clock separation between levels explicit, not emergent.
- Every gate writes to `FINDINGS.md`. Commit it.
- The currency is **shots**, everywhere. See METRICS.md.
