# TORAX API notes

Findings from reading TORAX at tag **v1.4.3** and at `main` (`e9c5cdc`,
2026-09-11), plus the Gym-TORAX source, on 2026-09-14. This file exists so the
next person does not have to rediscover any of it.

---

## The headline: the spec's Phase 0 gate is already answered

SPEC.md Phase 0 says *"Confirm JIT survives the wrapper — if every step
recompiles, the project is too slow to proceed."*

**It survives, by design.** TORAX added a public `torax.experimental` namespace
specifically for stepped/RL use:

```python
from torax.experimental import (
    make_step_fn, get_initial_state_and_post_processed_outputs,
    SimulationStepFn, RuntimeParamsProvider, TimeVaryingScalarUpdate, SimState,
)
```

TORAX config models register as JAX pytrees, splitting fields by annotation:
runtime parameters are **traced leaves**, and only fields annotated
`JAX_STATIC` become compile-time constants. `update_provider_from_mapping` is
implemented with `eqx.tree_at` — pure leaf replacement, no Pydantic
re-validation — and its docstring says *"Works under `jax.jit`"*.

TORAX lead `jcitrin` endorsed exactly this call sequence for an RL environment
in [discussion #1625](https://github.com/google-deepmind/torax/discussions/1625)
(2026-07-09).

This is why `scripts/gate0_jit.py` is framed as *confirm on your hardware*
rather than *make-or-break*. **The real make-or-break moved to Phase 4** — the
dimensionless transfer gate.

---

## The call sequence

```python
cfg       = torax.ToraxConfig.from_dict(CONFIG)
step_fn   = make_step_fn(cfg)
state, pp = get_initial_state_and_post_processed_outputs(t=t0, step_fn=step_fn)

# per action window:
provider  = step_fn.runtime_params_provider.update_provider_from_mapping({
    "sources.generic_heat.P_total": TimeVaryingScalarUpdate(
        time=np.array([t, t + dt]), value=np.array([prev, new])),
    "numerics.t_final": float(t + dt),
})
step_fn   = SimulationStepFn(
    solver=step_fn.solver,
    time_step_calculator=step_fn.time_step_calculator,
    runtime_params_provider=provider,
    geometry_provider=step_fn.geometry_provider,
)
state, pp = step_fn.jitted_fixed_time_step(dt, state, pp)
err       = step_fn.check_for_errors(state, pp)
```

Implemented in `hfmarl/envs/torax_env.py::ToraxCore`.

---

## What breaks the compilation (all three matter)

**1. Changing a leaf's shape.** A waveform array with a different number of
breakpoints is a new JAX signature. Hence the **two-breakpoint rule**: every
actuator update is exactly `[t, t+dt] → [prev, new]`, always. Enforced and
tested in `hfmarl/envs/actuators.py` and `tests/test_actuators.py`.

**2. Changing a `JAX_STATIC` field.** None of these may ever be an action:

```
geometry.n_rho              numerics.evolve_ion_heat       transport.model_name
geometry.hires_factor       numerics.evolve_electron_heat  solver.solver_type
numerics.exact_t_final      numerics.evolve_current        time_step_calculator.calculator_type
numerics.adaptive_dt        numerics.evolve_density
numerics.enable_fast_ions   sources.<name>.mode            sources.<name>.is_explicit
*.interpolation_mode        *.is_bool_param
```

Guarded by `assert_not_static()`, which fires when the action space is
*defined* rather than during training.

Note `numerics.t_final` is **not** static — it is a plain float leaf, and it
must be updated every step to track the action window (TORAX compares `t + dt`
against it when handling the final, possibly cropped, substep).

**3. Changing a dtype.** TORAX runs float64 by default. A float32 update leaf
is a different signature. `build_updates` forces `np.float64`.

---

## Schema drift — why TORAX is pinned

`examples/basic_config.py` differs between the released tag and `main`:

```python
# v1.4.3 (what `pip install torax` gives you) — flat
'transport': {'model_name': 'constant'}

# main (post-1.4.3, unreleased) — a registry
'transport': {'core_transport_models': {'prescribed': {'model_name': 'prescribed'}}}
```

`main` also grew `transport.pedestal_transport_models` and a
`_v1_compatibility` before-validator carrying
`TODO(b/434175938): Remove this once V1 API is deprecated`.

**This repo writes the flat 1.4.3 form and pins `torax==1.4.3`.** An unpinned
upgrade fails validation in `gate0_env.py` — loudly, which is the good outcome.

Other renames worth knowing: `runtime_params` split into
`profile_conditions`/`numerics`/`plasma_composition`; `stepper` → `solver`
(`solver_type`, not `stepper_type`); `nrho` → `n_rho`; bootstrap current moved
under `neoclassical`. **`ToraxSimState` was renamed `SimState`** — SPEC.md uses
the old name.

---

## Outputs: everything needed is a direct field

Nothing is derived by us, which matters — recomputing e.g. the Greenwald
fraction ourselves would grade the policy against a different number than the
simulator enforces.

`PostProcessedOutputs` (scalars): `q95`, `li3`, `beta_N`, `beta_pol`,
`beta_tor`, `f_bootstrap`, `f_non_inductive`, **`fgw_n_e_line_avg`** and
`fgw_n_e_volume_avg` (Greenwald fractions), `n_e_line_avg`, `W_thermal_total`,
`tau_E`, `H98`/`H89P`/`H97L`/`H20`, `Q_fusion`, `P_heat_total`, `P_SOL_total`,
`P_aux_total`, `P_alpha_total`.

`SimState.core_profiles` (profiles): `T_i`, `T_e`, `n_e`, `n_i`, `psi`,
`q_face`, `s_face`, `Z_eff`, `j_total`, … `CellVariable` fields expose `.value`.

Read the dataclasses directly, **not** the xarray DataTree — Gym-TORAX's own
TODO notes `_to_xr` "takes as much time as the simulation itself".

---

## Geometry: why every device is `circular`

| Type | External file? | Notes |
|---|---|---|
| `circular` | **no — pure parameter** | `R_major`, `a_minor`, `B_0`, `elongation_LCFS`. The only file-free option. |
| `chease` | yes | Ships `iterhybrid.mat2cols` — a real ITER equilibrium, usable now. |
| `eqdsk` | yes, or in-memory | Ships `iterhybrid_cocos{02,11,17}` and `STEP_SPP_001_ECHD_ftop`. |
| `fbt` | yes | TCV-style; uniquely supplies divertor quantities. |
| `imas` | yes | Experimental; needs `imas-core`, not a TORAX dependency. |

Consequences, all of them limiting:

- **There is no SPARC equilibrium.** "SPARC-like" in this repo means *a compact
  high-field circular plasma with SPARC's R, a, B₀*. It is not SPARC. Say so.
- TORAX's own docstring says circular geometry is *"used for testing only"*;
  it assumes `r/a_minor = rho_norm` and supports no shaping beyond elongation.
- **`_check_edge_with_circular_geometry` raises** — edge/SOL models are refused
  on circular geometry. So SPEC.md Phase 7's exhaust cluster (the spec's own
  "strongest candidate" use case, §7) is **unreachable** without real
  equilibria. Budget for obtaining them, or descope Phase 7 honestly.

---

## Gym-TORAX: reference, not dependency

[antoine-mouchamps/gymtorax](https://github.com/antoine-mouchamps/gymtorax) —
MIT, v1.1.1 (2026-07-11), pins `torax 1.4.*`, published as
[arXiv:2510.11283](https://arxiv.org/abs/2510.11283) in *Software Impacts*.

**We build our own env** because its abstractions are single-agent and
ITER-hybrid-shaped, while this project needs device-parametric configs,
dimensionless observations and role-matched multi-agent control. But we copy
two non-obvious mechanics it got right, with attribution:

1. the two-breakpoint ramp;
2. updating `numerics.t_final` every step to track the action window.

Use it as a **regression oracle**: for an identical actuator sequence, our env
should reproduce its trajectory. It ships a physics regression suite
(`pytest --test-scenarios`) that replays an ITER-hybrid episode against a native
TORAX reference at single-float precision.

---

## Also worth knowing

- TORAX is **differentiable and vmappable** through `step_fn` — the repo ships
  `iter_hybrid_rampup_grad_and_vmap.ipynb` doing gradient ascent on a reward.
  That opens analytic policy gradients later, which would be a genuinely
  distinctive extension.
- **Open bug [#2331](https://github.com/google-deepmind/torax/issues/2331)**:
  `jax.grad` through `run_loop_jit` works, but `jax.jacfwd` and Hessians
  **silently return NaN**. Irrelevant for first-order RL; fatal if you ever
  want second-order sensitivity.
- TORAX does **not** adhere to a fixed COCOS convention (`docs/output.rst`).
  Matters when comparing signs against other codes.
- Upstream CI is **Linux-only** (`ubuntu-latest`, Python 3.12). macOS works in
  practice ([issue #1572](https://github.com/google-deepmind/torax/issues/1572),
  M3 Pro) but is not an upstream guarantee.

### One correction to SPEC.md §10

The spec says no TORAX RL environment was found. **Gym-TORAX exists and is
peer-reviewed** — cite it rather than implying the gap. Separately, a web
summary claiming arXiv 2606.07550 uses TORAX is **wrong**; that paper builds
dynamics from DIII-D historical discharge data. Do not cite it as a TORAX
wrapper.

---

# What first contact with TORAX 1.4.3 actually changed

Everything above this line was written from reading the sources. Everything
below was measured by running them. Where the two disagree, below wins.

## It runs on Windows

TORAX 1.4.3 installs and runs from a plain `uv venv --python 3.12` on Windows,
CPU backend, no WSL. JAX resolves to 0.11.1 and only the TPU plugin is missing
(`LoadPjrtPlugin is not implemented on windows yet`, harmless). SETUP.md's WSL
route is still right for GPU work; it is not a precondition for the gates.

## The no-recompile contract holds, with room to spare

`gate0_jit.py`: 4.4 s to compile, then **6 ms per step**, spread 1.22x, with
the growing-array negative control 601x slower. The two-breakpoint ramp does
what Gym-TORAX said it does.

This is ~8x faster than the 0.05 s/step PLAN.md budgeted, so the Phase 5
matrix is much cheaper than assumed. Re-derive the budget from a measurement,
not from the old estimate.

## There is no `prescribed` transport model. There does not need to be.

1.4.3's transport tags are exactly `qlknn`, `constant`, `combined`. The
`prescribed` model this repo looked for does not exist under that name --
which is why `gate_identify.py` question 1 originally answered "the
closure-free loop cannot close".

That answer was wrong. `ConstantTransportModel` declares `chi_i`, `chi_e`,
`D_e`, `V_e` as `TimeVaryingArray`, i.e. `constant` means "chi comes from the
config rather than from a model", not "chi is a constant". It accepts a full
chi(rho) profile:

```python
cfg["transport"] = {
    "model_name": "constant",
    "chi_i": {0.0: {rho_0: chi_0, ..., rho_n: chi_n}},
    "chi_e": {...},
}
```

Verified: validates, steps, and the values land as traced leaves. **The
closure-free loop closes.** Identify chi(rho), write it back this way,
re-solve, compare.

## Differentiating through TORAX: two hard constraints

Both were discovered by running the inverse harness, and both produce failures
that look like something else entirely.

**1. Reverse mode does not work. Use forward mode.**

`SimulationStepFn.fixed_time_step` consumes the action window inside a
`jax.lax.while_loop` with a dynamic trip count (`step_function.py:336`). JAX
refuses to reverse-differentiate that:

    ValueError: Reverse-mode differentiation does not work for lax.while_loop
    or lax.fori_loop with dynamic start/stop values.

So `jax.value_and_grad` is unusable here. Forward mode (`jax.jacfwd`, `jax.jvp`)
goes through `while_loop` unchanged, and for identification it is also the
right direction: cost scales with the number of unknowns, and there are four.
Reverse mode only starts to win in the hundreds, at which point the fix is a
scan-based stepper rather than a different `grad` call.

**2. `numerics.adaptive_dt` must be False on the differentiable path.**

This one is nastier, because it fails silently. With TORAX's default
`adaptive_dt=True`, a step branches on `solver_error_state` and retries at
half dt. The primal value is correct; the **tangent of that branch is NaN**,
and the NaN spreads to every parameter. Measured, same one-step JVP:

| `adaptive_dt` | primal | tangent |
|---|---|---|
| `True` (default) | 70.79 | **nan** |
| `False` | 57.65 | 0.765 |

A NaN gradient is indistinguishable from an unidentifiable parameter, so this
made `gate_identify.py` report all four candidates DEAD and call a set
unidentifiable that is merely degenerate. `hfmarl/identification/torax_inverse.build_loss`
now forces it off. `adaptive_dt` is JAX_STATIC, so this costs nothing at run
time -- but without adaptive retry a step that would have been rescued by
halving dt now simply fails, so pick a `fixed_dt` the forward solve is
comfortable with.

## Replacement values are typed, and the type is per-leaf

`update_provider_from_mapping` will not coerce. It dispatches on what the leaf
already is (`build_runtime_params.py:254`):

| leaf type | what you must pass |
|---|---|
| `TimeVaryingScalar` | `TimeVaryingScalarUpdate(time=(t,), value=(t,))` |
| `TimeVaryingArray` | `TimeVaryingArrayUpdate(time=(t,), rho_norm=(rhon,), value=(t, rhon))` |
| plain scalar / Array | the raw value, with **exactly** matching shape and dtype |

Note the asymmetry in `TimeVaryingArrayUpdate`: `value` is 2-D `(t, rhon)` but
`rho_norm` is 1-D `(rhon,)`. Passing `(1, rhon)` raises the confusing
"rho_norm and value must have the same trailing dimension" while printing two
shapes that look identical.

`plasma_composition.Z_eff` is a `TimeVaryingArray`, not a scalar -- it is the
first candidate unknown, so a bare float there stopped the inverse path before
it produced a single gradient.

## The device configs are physically miscalibrated

Not a TORAX issue, but it is what the gates actually found. See README "Status"
and FINDINGS.md "Gate 0d": the reachable beta_N band spans a factor of 50
across the four devices, and the single absolute setpoint of beta_N = 2.0 lies
outside three of them.

## `use_pereverzev` must be on for stiff transport

TORAX warns, loudly and on every step, whenever a linear solver meets a stiff
transport model:

    use_pereverzev=False in a configuration where setting
    use_pereverzev=True is recommended.

It fired on every `qlknn` shot and was being ignored. The Pereverzev-Corrigan
inner-loop terms are what keep the linear solver stable against stiff fluxes.
`build_config` now sets it whenever `solver_type == "linear"` and the transport
model is not `constant`; it stays off for `constant`, where it would add
numerical diffusion for nothing.

## Profile stiffness is real, and it eats the big machines' control authority

Reachable beta_N band (zero to full auxiliary command), same device, same
everything except transport:

| device | `constant` | `qlknn` |
|---|---|---|
| `iter_like` | 0.459 .. 0.904  (0.444) | 0.806 .. 0.816  (**0.011**) |
| `sparc_like` | 0.403 .. 0.479  (0.076) | 0.705 .. 0.837  (0.133) |
| `diiid_like` | 0.499 .. 5.433  (capped) | 0.765 .. 1.523  (0.758) |
| `tcv_like` | 0.656 .. 7.349  (capped) | 0.712 .. 3.274  (1.788) |

Above the critical gradient qlknn's transport rises steeply, so extra power
raises transport rather than temperature. On `iter_like` that leaves 0.011 of
beta_N authority for 53 MW of auxiliary heating -- a well-posed simulation
problem and a meaningless experimental one, since no diagnostic resolves beta_N
to a part in a thousand. `gate_authority.py` reports it as a WARNING.

The practical consequence: **the reachable band is a property of the plant, and
the task sets part of the plant.** `BETA_N_BANDS` is keyed by task as well as
device for this reason. Keying it by device alone made `brutal` resolve against
`easy`'s band and fail 3 of 4 devices for a reason that looked identical to the
original unreachable-setpoint bug.
