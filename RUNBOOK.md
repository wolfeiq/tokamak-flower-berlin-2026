# Runbook — what to do on the WSL machine

Step-by-step, in order. Each step says **what to run**, **what a pass looks
like**, and **what to do when it fails**. You can paste the prompts verbatim
to Claude Code in this repo.

Nothing in this repo has ever touched a live TORAX. Expect step 2 to fail the
first time — that's what it's for.

---

## 0 · Clone and install (~15 min)

Read `SETUP.md §1` first — two WSL mistakes cost an afternoon each (the NVIDIA
driver goes on **Windows**, not inside WSL; keep the repo on the Linux
filesystem, **not** `/mnt/c`).

```bash
cd ~ && git clone <repo-url> hfmarl-fusion && cd hfmarl-fusion
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'
uv pip install -e '.[gpu]'          # optional, see §5 of SETUP.md first

export JAX_COMPILATION_CACHE_DIR=~/.cache/jax
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
```

**Sanity, no TORAX needed — should take ~2 seconds:**

```bash
uv run pytest                                  # expect 300+ passed
uv run python scripts/describe_devices.py      # device set in similarity space
uv run python scripts/demo_figures.py          # the 6 deliverable figures (synthetic)
uv run python scripts/exp_identify_chi.py      # chi(rho) identification maths
```

If those pass, the physics, metrics, figures and identification maths are
sound. Everything after this is about whether the TORAX *interface* works.

---

## 1 · Does TORAX run here? (~10 min, mostly first-time JIT)

```bash
uv run python scripts/gate0_env.py
```

**Pass:** TORAX imports, all four device configs validate, a sim steps, and
`beta_N` / `q95` / `fgw_n_e_line_avg` are all present.

**When it fails** — likely, this is first contact between my code and the real
API:

> Claude: `gate0_env.py` failed with the traceback below. Fix
> `hfmarl/envs/torax_config.py` and `hfmarl/envs/torax_env.py` against the
> installed TORAX version, then re-run. Check the installed version's config
> schema first — the repo pins 1.4.3 and writes the flat
> `transport: {'model_name': ...}` form.
>
> ```
> <paste traceback>
> ```

---

## 2 · THE gate: does JIT survive per-step actuator changes? (~10 min)

```bash
uv run python scripts/gate0_jit.py --negative-control
```

**Pass:** median of steps 2..N under 10% of step 1, spread under 3×, **and**
the negative control ≥2× slower. Always pass `--negative-control` — a test that
can't fail proves nothing.

**When it fails:** check that every update is exactly two breakpoints
(`hfmarl/envs/actuators.py`), that no action touches a `JAX_STATIC` path, and
that `delta_t_a` is an integer multiple of `numerics.fixed_dt`. The fallback is
episode-level control, which changes the RL formulation — decide before Phase 1.

---

## 3 · CPU or GPU? (~30 min, four compiles)

```bash
uv run python scripts/gate0_bench.py
```

Informational. **Do not assume the GPU wins** — TORAX defaults to float64,
which runs ~1/32 rate on consumer NVIDIA, and this is a 1-D problem on ~25
cells. The likely real win from this machine is cores and RAM for parallel
Phase-5 clients.

---

## 4 · Look at the plasma (~15 min)

```bash
uv run python scripts/plot_torax.py --all-devices
```

Writes to `results/torax_plots/`: profiles, scalar traces with operating limits
drawn on, and a **step response** — power in, β_N out.

Look at the step response before anything else. If β_N responds instantly the
control problem is trivial; if it never responds the actuator isn't connected.
Either way you learn it here, in minutes, rather than after a training run.

For exploring one run interactively, TORAX ships its own plotter:
`from torax.experimental import create_plotly_figure`.

---

## 4b · Can the actuators reach the setpoint? (BLOCKING, ~5 min)

```bash
uv run python scripts/gate_authority.py --task easy --plot
```

Cheap, and it has to come before gate 1, because a gate-1 failure caused by an
unreachable setpoint looks exactly like a learning failure and costs an
afternoon to tell apart.

It holds each actuator at a fixed fraction of its envelope for a whole shot and
records where beta_N lands. **Pass** needs the setpoint strictly inside that
band, the band wider than the task tolerance, and the zero-command shot to
survive step 1.

It failed 3 of 4 devices on its first run and now passes all four; the
three fixes that got it there (pedestal temperature scaled as a·B₀, SPARC's
real ICRF power restored, and setpoint **and tolerance** expressed as fractions
of each device's measured band) are in FINDINGS.md "Gate 0d".

**Re-run it after any change to the plant** — thermal scaling, actuator
envelopes, transport model, episode length — and paste the bands it prints into
`hfmarl/devices/registry.BETA_N_BANDS`. They are keyed by task as well as
device: `iter_like` has 0.444 of β_N authority under `constant` transport and
0.011 under `qlknn`, which is profile stiffness, not a bug.

---

## 5 · Identification — the PINN work

This is the part that couples RL and PINNs rather than stacking them. Run it in
this order; each step gates the next.

### 5a · Is identification possible at all? (~10 min)

```bash
uv run python scripts/gate_identify.py
```

Answers three questions:

1. **Does `prescribed` transport accept χ as a profile over ρ?** In TORAX
   `main`, `PrescribedTransportModel` declares `chi_i`/`chi_e`/`D_e`/`V_e` as
   `TimeVaryingArray` with only `model_name` marked `JAX_STATIC` — so they're
   traced leaves and differentiable. **Whether 1.4.3 exposes them identically
   is unverified**, and this checks it. If yes, the closure-free loop closes:
   identify χ(ρ), feed it back, confirm the forward solve reproduces the data.
2. **Do gradients flow to each candidate unknown?** A zero gradient means
   either unidentifiable or untraced — those look identical and need opposite
   fixes.
3. **Is the candidate set degenerate?** CRLB over all of them. Note the
   per-parameter bound alone will *lie*: in a deliberately degenerate test it
   reported 0.1% "identifiable" for two parameters that were the same
   parameter. The correlation and rank-deficiency checks are what catch it.

**Expected casualties** — say so rather than fitting them anyway:
- confinement-deviation-from-scaling ≈ a transport multiplier (τ_E deviation
  *is* an integrated measure of χ)
- wall recycling is likely invisible in a 1-D core model with a prescribed edge
  density boundary condition

### 5b · Ground-truth recovery — the credibility result (~30 min)

```bash
uv run python scripts/exp_recover_scalars.py
uv run python scripts/exp_recover_scalars.py --noise 0.02 --subsample 4
```

Generate trajectories from a known TORAX config, hide it, recover the
parameters by differentiating through the simulator. Clean pass/fail, no
ambiguity — and it's what makes everything built on identification credible.

**Run the noisy/sparse version too.** Handed full noiseless profiles at every
timestep, identification is far easier than from real diagnostics, and every
downstream claim inherits that optimism. Report clean and noisy separately.

**Pass:** worst relative error under 5%.

**When it fails**, in this order: is it identifiable (5a)? is a gradient dead?
are two parameters degenerate? *Only then* touch the optimiser.

### 5c · χ(ρ) closure-free identification

```bash
uv run python scripts/exp_identify_chi.py --plot
uv run python scripts/exp_identify_chi.py --noise 0.03 --points 40 --plot
uv run python scripts/exp_identify_chi.py --noise 0.05 --points 25 --plot
```

Runs **without TORAX** on a manufactured solution, so you can do it now.

The estimator integrates the conservation law and divides by the measured
gradient: no network, no training loop, ~1% error on clean data, instant.

The physics is the reason this is worth doing at all. Conservation is
**certain**; the closure q = −nχ∇T is a **model**. Imposing the first exactly
and treating χ(ρ) as free is something inverting TORAX's own parameters
structurally cannot do, because TORAX bakes its closure in. The output is a
per-device transport fingerprint — a field, not a scalar, and exactly what
makes one machine differ from another.

Once 5a confirms `prescribed` transport accepts a χ(ρ) profile, the loop
closes: identify χ(ρ), feed it back into TORAX, confirm the forward solve
reproduces the data.

> An inverse-PINN arm exists behind `--pinn`. It did not beat the classical
> estimator anywhere it was tried and is kept only as a documented negative
> result (`docs/pinn_comparison.md`). Not on the critical path.

---

## 6 · Is the task hard enough? (BLOCKING, overnight)

```bash
# Get a rate first — do NOT launch the full thing blind.
uv run python scripts/gate_headroom.py --tasks easy --seeds 1 --shot-budget 50
```

That prints seconds-per-shot and estimated matrix hours. At defaults the full
gate is ~540,000 TORAX steps — 8 hours at 0.05 s/step, several times that with
`qlknn`. Scale from the pilot, then run overnight.

**Measured on this box** (Windows, CPU, warm JAX cache): `easy` on
`iter_like` runs at **0.66 s/shot** over 600 shots at 20 action steps per
shot (0.80 over 50 shots, 1.07 cold — the 4.4 s compile amortises), so the
default matrix (5 × 4 × 3 seeds × 2000 shots) is **~22 h serial** — inside
the 72 h budget without parallelism. `hard` and `brutal` use `qlknn` and
twice the episode length, so scale accordingly and re-measure.

**First real verdict: `easy` FAILS on headroom.** 600 shots, 1 seed,
`iter_like`: isolated reaches threshold in **240 shots**, against a
1000-shot floor. Cost is not the binding constraint and never was — the
task being too easy is. The gate's own advice is the next preset up;
`moderate` adds a ramp setpoint, a tighter tolerance and per-shot
disturbances. Record in FINDINGS.md, "Headroom gate -- iter_like (FAILED)".

**The 50-shot pilot cannot pass or fail the gate, by construction.** Hill
climbing has not annealed its step size in 50 shots, so the logged curve
(candidate returns, not the incumbent's) has not risen above where it
started, and a threshold defined as a fraction of the way from start to
plateau does not exist. The gate reports **INCONCLUSIVE (budget)** and
still prints the rate — which is the only thing the pilot is for. Do not
read that as the task being too easy or too hard; it is neither claim.

**Pass** requires three things together: headroom (isolated ≥1000 shots),
measurability (≥80% of seeds converge), affordability (matrix ≤72 h serial).

**This gate is biased toward passing** — it measures with hill climbing, and a
task hill climbing needs 3000 shots for might take PPO 300. A PASS is
*provisional*; re-measure with whatever algorithm the matrix uses. A FAIL is
trustworthy.

If it fails on headroom, that's the claim-killer arriving early and cheap. The
gate names the binding constraint and escalates the task presets.

---

## 7 · Phase 1 and onward

```bash
uv run python scripts/gate1_thermal.py --task easy
```

Then `PLAN.md` for Phases 2–7, and `METRICS.md` for what the experiments have
to produce.

---

## Prompts worth pasting

**After each gate:**
> Claude: read `FINDINGS.md`, summarise what the gates have established so far,
> and tell me what the next blocking unknown is.

**When something fails:**
> Claude: `<script>` failed with the traceback below. Diagnose it against the
> installed TORAX source (don't guess the API — read it), fix, re-run, and tell
> me whether the fix was in my code or my assumptions.

**Before Phase 5:**
> Claude: run `describe_devices.py` and check the aggregation weight matrix.
> Confirm the off-diagonal weights neither collapse to zero (federation does
> nothing) nor to one (uniform). Then re-derive the operating points from
> actual TORAX runs rather than the nominal table, and tell me if the
> similarity structure changed.

---

## Where to be sceptical

- `hfmarl/envs/torax_env.py` and `hfmarl/identification/torax_inverse.py` have
  **never been executed**. Everything else has unit tests.
- The reward *scale* (−50 violation penalty vs ~−0.2/step tracking) is a guess
  that needs calibrating against real β_N trajectories.
- "SPARC-like" is a compact high-field **circular** plasma with SPARC's R, a,
  B₀. It is not SPARC; no SPARC equilibrium exists to use.
- Phase 7 (divertor/exhaust) is **blocked** — TORAX refuses edge models on
  circular geometry, and circular is the only file-free geometry.
- The inverse PINN has not yet beaten the classical estimator anywhere. That's
  an open question, not a result.
