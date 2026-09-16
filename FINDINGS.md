# Findings

Gate results, appended automatically by the scripts in `scripts/`. Each block
is stamped with host, platform, JAX/TORAX versions, devices, x64 setting and
git commit, because two results are only comparable if those match.

Nothing here yet — no gate has been run against a live TORAX. See SETUP.md §4
for the order to run them in.

## What to paste in by hand

- `scripts/describe_devices.py` output, before Phase 5 (it does not
  auto-record, because it is a design check rather than a measurement).
- Any negative result. SPEC.md Phase 2 and Phase 4 both explicitly ask for
  negative results to be reported rather than tuned away.


---

## chi(rho) identification -- manufactured, 0% noise, 60 pts

*2026-09-15 00:42:57*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=dfc39a0

```
manufactured case: chi(rho) = Anomalous transport at mid-radius -- the case the method exists for.
  true chi range   : 0.500 .. 2.498
  observation pts  : 60
  noise            : 0.0%

1. classical power balance (integral form, finite differences)
   mean relative error : 0.0128
   93% of points usable
   time                : 0.00 s

PASS

The inverse-PINN arm is off by default -- it did not beat this
estimator anywhere tried. Pass --pinn to reproduce that comparison.
```


---

## Gate 0a -- environment (PASSED)

*2026-09-15 00:43:15*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
torax version : 1.4.3
jax version   : 0.11.1
jax devices   : [CpuDevice(id=0)]
x64 enabled   : True

config validation:
  diiid_like     OK
  iter_like      OK
  sparc_like     OK
  tcv_like       OK

running iter_like, 3 steps of 0.5s
  build + initial state :     2.69 s
  step  1               :     4.57 s   ok
  step  2               :     0.01 s   ok
  step  3               :     0.01 s   ok
  peak RSS              :        0 MB

scalars found : 18/18
  OK  beta_N                 0.1690   <- a limit depends on this
  OK  q95                    4.4152   <- a limit depends on this
  OK  fgw_n_e_line_avg       0.5840   <- a limit depends on this

profiles: T_i[25], T_e[25], n_e[25], psi[25], q_face[26], s_face[26]

PASS: TORAX runs and reports every quantity the limits need.
Next: scripts/gate0_jit.py
```


---

## Gate 0b -- JIT survival (PASSED)

*2026-09-15 00:44:09*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
device iter_like, actuators ('nbi', 'ecrh')
delta_t_a = 0.5s, transport = constant, solver = linear

CORRECT (2 breakpoints)
  step 1 (compile)      :     4.438 s
  steps 2..10  median   :     0.006 s
  steps 2..10  min/max  :     0.006 / 0.007 s
  ratio median/step1    :    0.0014
  spread max/min        :      1.22

running negative control -- this SHOULD be slow
NEGATIVE CONTROL (growing array, expected to recompile)
  step 1 (compile)      :     3.834 s
  steps 2..10  median   :     3.867 s
  steps 2..10  min/max  :     3.743 / 3.967 s
  ratio median/step1    :    1.0085
  spread max/min        :      1.06

peak RSS: 0 MB

ratio  < 0.1: PASS
spread < 3.0: PASS
negative control is 601.0x slower than correct path

PASS: JIT survives per-step actuator changes. The Gym wrapper approach in SPEC.md is viable. Proceed to Phase 1.
```


---

## Gate 0d -- actuator authority, task 'easy' (FAILED)

*2026-09-15 00:48:59*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
task     : easy (tolerance 0.15)
setpoint : constant, base 2, amplitude 0
episode  : 10s in 0.5s steps

diiid_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.741   q95   6.29   fgw 0.431   20/20 steps
  cmd  25%  P_aux    6.50 MW   beta_N    1.563   q95   6.18   fgw 0.431   20/20 steps
  cmd  50%  P_aux   13.00 MW   beta_N    3.095   q95   2.25   fgw 0.431   15/20 steps  VIOLATED@15
  cmd  75%  P_aux   19.50 MW   beta_N    3.309   q95   6.15   fgw 0.431    2/20 steps  VIOLATED@2
  cmd 100%  P_aux   26.00 MW   beta_N    3.524   q95   6.27   fgw 0.435    1/20 steps  VIOLATED@1
  -> PASS

iter_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.149   q95   4.39   fgw 0.524   20/20 steps
  cmd  25%  P_aux   13.25 MW   beta_N    0.220   q95   4.34   fgw 0.524   20/20 steps
  cmd  50%  P_aux   26.50 MW   beta_N    0.307   q95   4.33   fgw 0.524   20/20 steps
  cmd  75%  P_aux   39.75 MW   beta_N    0.405   q95   4.33   fgw 0.524   20/20 steps
  cmd 100%  P_aux   53.00 MW   beta_N    0.510   q95   4.34   fgw 0.524   20/20 steps
  -> FAIL
     setpoint reaches 2.00 but full command only produces beta_N = 0.51. The policy saturates at maximum power and every parameter vector scores alike; there is no control problem here.

sparc_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.178   q95   3.84   fgw 0.425   20/20 steps
  cmd  25%  P_aux    1.27 MW   beta_N    0.181   q95   3.83   fgw 0.425   20/20 steps
  cmd  50%  P_aux    2.55 MW   beta_N    0.184   q95   3.82   fgw 0.425   20/20 steps
  cmd  75%  P_aux    3.83 MW   beta_N    0.187   q95   3.80   fgw 0.425   20/20 steps
  cmd 100%  P_aux    5.10 MW   beta_N    0.190   q95   3.79   fgw 0.425   20/20 steps
  -> FAIL
     setpoint reaches 2.00 but full command only produces beta_N = 0.19. The policy saturates at maximum power and every parameter vector scores alike; there is no control problem here.
     actuator authority is 0.012 in beta_N, narrower than the task tolerance 0.150. Full-scale command barely moves the controlled variable.

tcv_like
  cmd   0%  P_aux    0.00 MW   beta_N    3.780   q95   2.22   fgw 0.406    1/20 steps  VIOLATED@1
  cmd  25%  P_aux    1.45 MW   beta_N    5.844   q95   1.15   fgw 0.406    1/20 steps  VIOLATED@1
  cmd  50%  P_aux    2.90 MW   beta_N    8.784   q95   0.70   fgw 0.406    1/20 steps  VIOLATED@1
  cmd  75%  P_aux    4.35 MW   beta_N    8.693   q95   2.34   fgw 0.406    1/20 steps  VIOLATED@1
  cmd 100%  P_aux    5.80 MW   beta_N   10.513   q95   3.58   fgw 0.406    1/20 steps  VIOLATED@1
  -> FAIL
     setpoint falls to 2.00 but ZERO command already gives beta_N = 3.78. The actuators can only push further away.
     the shot at ZERO command violates ['beta_N'] on step 1. Every episode terminates immediately whatever the policy does, so the reward is exactly constant and carries no gradient.

wrote results\authority\authority_easy.json
wrote results\authority\authority_easy.png

summary: 1/4 devices can reach the 'easy' setpoint
peak RSS: 810 MB

FAIL: iter_like, sparc_like, tcv_like cannot be controlled to this setpoint.
Gate 1 cannot pass on these devices and the Phase 5 matrix would
spend its budget on clients that produce no learning signal. The
setpoint is a per-device quantity, not a constant -- set it from
the measured band above (the repo already does exactly this for
density, which is specified as a Greenwald fraction rather than in
m^-3, so that four devices start in comparable states).
```


---

## Gate identify (FAILED)

*2026-09-15 00:50:12*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
1. prescribed transport with a chi(rho) profile
   no   flat (1.4.3): transport={'model_name':'prescribed', 'chi_i': {...}}
        ValidationError: 1 validation error for ToraxConfig
transport
  Input tag 'prescribed' found using 'model_name' does not match any of the expected tags: 'combined', 'q
   no   registry (main): transport={'core_transport_models': {...}}
        ValidationError: 1 validation error for ToraxConfig
transport.constant.core_transport_models
  Extra inputs are not permitted [type=extra_forbidden, input_value={'pres
   FAIL: no prescribed-transport form accepted a chi(rho) profile.
   The closure-free chi identification cannot feed its result back
   into TORAX. The identification itself still works (it only needs
   profiles and sources), but you lose the forward-validation loop.

2. gradient flow to each candidate unknown
   FAIL: could not build the differentiable loss: ValueError: To replace a `TimeVaryingArray` use a `TimeVaryingArrayUpdate`, got <class 'numpy.float64'> instead.
   This is the first contact between the inverse harness and TORAX;
   the traceback below is the useful output.
```


---

## Gate identify -- iter_like (FAILED)

*2026-09-15 00:57:12*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
1. transport accepting a chi(rho) profile
   OK   1.4.3: transport={'model_name':'constant', 'chi_i': {rho: chi}}
   no   flat (other): transport={'model_name':'prescribed', 'chi_i': {...}}
        ValidationError: 1 validation error for ToraxConfig
transport
  Input tag 'prescribed' found using 'model_name' does not match any of the expected tags: 'combined', 'q
   no   registry (main): transport={'core_transport_models': {...}}
        ValidationError: 1 validation error for ToraxConfig
transport.constant.core_transport_models
  Extra inputs are not permitted [type=extra_forbidden, input_value={'pres
   -> the closure-free loop CAN close: identify chi(rho), write it
      back as a transport profile, re-solve, compare.

2. gradient flow to each candidate unknown
   OK   |dL/d(plasma_composition.Z_eff)| = 2.6244e-02
   OK   |dL/d(sources.ei_exchange.Qei_multiplier)| = 2.6900e-03
   OK   |dL/d(neoclassical.bootstrap_current.bootstrap_multiplier)| = 2.5144e-03
   OK   |dL/d(numerics.resistivity_multiplier)| = 6.0051e-02

3. identifiability over the candidate set (CRLB)
   parameter                           true    CRLB std       rel  verdict
   plasma_composition.Z_eff             1.6      0.7183     44.9%  NOT identifiable
   sources.ei_exchange.Qei_multiplier           1      0.7623     76.2%  NOT identifiable
   neoclassical.bootstrap_current.bootstrap_multiplier           1      0.3178     31.8%  NOT identifiable
   numerics.resistivity_multiplier           1      0.3306     33.1%  NOT identifiable
   
   most collinear pair: plasma_composition.Z_eff / numerics.resistivity_multiplier  (|r| = 0.998)
     WARNING: these two are nearly the same parameter. Fitting both produces a confident answer along an arbitrary point of the degenerate direction. Drop one, or design an experiment that separates them.
   Fisher condition number: 5.525e+04

VERDICT: 0/4 candidates are identifiable at 10%:
prescribed chi(rho) profile supported: YES

Use the identifiable set in scripts/exp_recover_scalars.py.
```


---

## Scalar recovery -- iter_like (FAILED)

*2026-09-15 00:58:06*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
device      : iter_like
parameters  : ['plasma_composition.Z_eff', 'sources.ei_exchange.Qei_multiplier']
hidden truth: [1.7534 0.8389]
start guess : [1.4027 0.6711]
observations: 8 steps, noise 0.0%, every 1 radial point(s)

gradient flow at the starting point:
   OK   |dL/d(plasma_composition.Z_eff)| = 6.1579e-02
   OK   |dL/d(sources.ei_exchange.Qei_multiplier)| = 1.0913e-02

optimising:
    it 0  loss 9.536071e-03
    it 10  loss 1.461147e-03
    it 20  loss 2.067151e-04
    it 30  loss 4.640613e-06
    it 40  loss 3.101990e-05
    it 50  loss 1.883114e-05
    it 59  loss 4.192375e-06

parameter                                                truth   recovered   rel err
plasma_composition.Z_eff                                 1.753       1.745    0.46%
sources.ei_exchange.Qei_multiplier                      0.8389      0.7408   11.69%

worst relative error: 11.69% (FAIL at 5%)

FAIL. Before tuning the optimiser, check in this order:
  1. scripts/gate_identify.py -- is the parameter identifiable at all?
  2. is any gradient DEAD above?
  3. are two parameters degenerate? (CRLB correlation)
  4. only then: more iterations, different lr
```


---

## Scalar recovery -- iter_like (FAILED)

*2026-09-15 00:59:26*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
device      : iter_like
parameters  : ['plasma_composition.Z_eff', 'sources.ei_exchange.Qei_multiplier']
hidden truth: [1.7534 0.8389]
start guess : [1.4027 0.6711]
observations: 8 steps, noise 0.0%, every 1 radial point(s)

gradient flow at the starting point:
   OK   |dL/d(plasma_composition.Z_eff)| = 6.1579e-02
   OK   |dL/d(sources.ei_exchange.Qei_multiplier)| = 1.0913e-02

optimising:
    it 0  loss 9.536071e-03
    it 10  loss 1.461147e-03
    it 20  loss 2.067151e-04
    it 30  loss 4.640613e-06
    it 40  loss 3.101990e-05
    it 50  loss 1.883114e-05
    it 60  loss 5.512803e-06
    it 70  loss 1.667306e-06
    it 80  loss 1.114343e-06
    it 90  loss 1.029832e-06
    it 100  loss 9.464943e-07
    it 110  loss 8.574699e-07
    it 120  loss 7.750520e-07
    it 130  loss 7.009394e-07
    it 140  loss 6.334055e-07
    it 149  loss 5.775671e-07

parameter                                                truth   recovered   rel err
plasma_composition.Z_eff                                 1.753       1.744    0.52%
sources.ei_exchange.Qei_multiplier                      0.8389      0.7762    7.47%

worst relative error: 7.47% (FAIL at 5%)

FAIL. Before tuning the optimiser, check in this order:
  1. scripts/gate_identify.py -- is the parameter identifiable at all?
  2. is any gradient DEAD above?
  3. are two parameters degenerate? (CRLB correlation)
  4. only then: more iterations, different lr
```


---

## Scalar recovery -- iter_like (FAILED)

*2026-09-15 01:01:03*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
device      : iter_like
parameters  : ['plasma_composition.Z_eff', 'sources.ei_exchange.Qei_multiplier']
hidden truth: [1.7534 0.8389]
start guess : [1.4027 0.6711]
observations: 8 steps, noise 2.0%, every 4 radial point(s)

gradient flow at the starting point:
   OK   |dL/d(plasma_composition.Z_eff)| = 4.6574e-02
   OK   |dL/d(sources.ei_exchange.Qei_multiplier)| = 8.0994e-03

optimising:
    it 0  loss 9.585583e-03
    it 10  loss 3.179534e-03
    it 20  loss 2.159989e-03
    it 30  loss 2.055566e-03
    it 40  loss 2.080392e-03
    it 50  loss 2.063055e-03
    it 60  loss 2.052910e-03
    it 70  loss 2.051219e-03
    it 80  loss 2.050919e-03
    it 90  loss 2.050518e-03
    it 100  loss 2.050073e-03
    it 110  loss 2.049675e-03
    it 120  loss 2.049329e-03
    it 130  loss 2.049035e-03
    it 140  loss 2.048785e-03
    it 149  loss 2.048596e-03

parameter                                                truth   recovered   rel err
plasma_composition.Z_eff                                 1.753       1.736    1.02%
sources.ei_exchange.Qei_multiplier                      0.8389      0.6468   22.90%

worst relative error: 22.90% (FAIL at 5%)

FAIL. Before tuning the optimiser, check in this order:
  1. scripts/gate_identify.py -- is the parameter identifiable at all?
  2. is any gradient DEAD above?
  3. are two parameters degenerate? (CRLB correlation)
  4. only then: more iterations, different lr
```


---

## AUDIT -- first live-TORAX run, and what the code audit found

*2026-09-15 01:04:15*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=dfc39a0

This block is written by hand, not by a gate. It summarises the first end-to-end
run of this repo against a live TORAX, and the code audit that went with it.

## Environment

TORAX 1.4.3 runs NATIVELY ON WINDOWS, CPU backend, from a plain
`uv venv --python 3.12`. No WSL was needed for any gate below. JAX resolved to
0.11.1; only the TPU plugin is unavailable, which is harmless.

## Gate results

| gate | result | number that matters |
|---|---|---|
| `pytest` | PASS | 364 passed, 1 skipped |
| `gate0_env.py` | **PASS, unmodified, first run** | 18/18 scalars present |
| `gate0_jit.py --negative-control` | **PASS, decisively** | 6 ms/step, ratio 0.0014, control 601x slower |
| `plot_torax.py --all-devices` | OK | beta_N responds to power with a ~2 s lag |
| `gate_authority.py` (NEW) | **FAIL 3 of 4 devices** | see below |
| `gate1_thermal.py` | FAIL | for exactly the reason gate 0d names |
| `gate_identify.py` | runs; 0/4 candidates identifiable | Z_eff / resistivity collinear at r = 0.998 |
| `exp_recover_scalars.py` | Z_eff 0.5%, Qei 7.5% (clean) | worst 7.5%, FAIL at the 5% bar |
| `exp_recover_scalars.py --noise 0.02 --subsample 4` | Z_eff 1.0%, Qei 22.9% | **could not run at all before the audit** |
| `exp_identify_chi.py` | PASS clean (1.3%) | noisy behaviour now monotone in noise |

The JIT result is the important positive one. At 6 ms/step a shot is ~8x cheaper
than PLAN.md assumed, so the Phase 5 budget should be re-derived from this
measurement rather than from the old 0.05 s/step estimate.

## BLOCKING: the task setpoint is unreachable on three of four devices

`SetpointSchedule.base` is one absolute number, beta_N = 2.0, for a device set
whose reachable beta_N band spans a factor of 50.

| device | reachable beta_N | verdict |
|---|---|---|
| `diiid_like` | 0.74 - 3.52 | setpoint inside the band |
| `iter_like` | 0.15 - 0.51 | saturates at full power, 4x short |
| `sparc_like` | 0.178 - 0.190 | full-scale command moves beta_N by 0.012 |
| `tcv_like` | 3.78 - 10.5 | already violating at ZERO command |

This is not a hard task, it is an absent one:

  * `iter_like`: the optimum is "maximum power, always". The best any policy
    achieves is |delta beta_N| = 1.52 against a 0.30 tolerance.
  * `tcv_like`: every episode terminates on step 1 whatever the policy does.
    Every constant-power baseline and every CEM sample scored EXACTLY -550.00 --
    a perfectly flat landscape, zero gradient.
  * `sparc_like`: no heating at all. Its NBI limit is 0.1 MW by design ("SPARC
    baseline heating is ICRF") and ICRH is `available=False` because TORAX 1.4.3
    has no independent ion-cyclotron source. Both decisions are individually
    documented and individually right. Composed, they leave 5.1 MW of ECRH
    against B_0 = 12.2 T and no control authority whatsoever. Nothing checked
    the composition.

Three of four Phase 5 federation clients would therefore contribute no learning
signal. `scripts/gate_authority.py` was added to catch this class of failure in
~5 minutes of open-loop shots, before gate 1 rather than after it.

**Suggested direction, not applied** -- choosing the setpoint is a physics call.
The repo already solves the identical problem for density: `build_config`
specifies it as a Greenwald FRACTION rather than in m^-3, precisely so the four
devices start in comparable states. The beta_N setpoint wants the same
treatment: a fraction of each device's own measured band. `gate_authority`
measures the band; the fraction is yours to choose.

## Code defects found and fixed

1. **`StepResult(violations=...)` -- the dataclass had no such field.** Both
   abnormal exits (`SOLVER_FAILURE`, `NONFINITE_STATE`) raised `TypeError`
   outside `advance()`'s try/except, so a solver blow-up or a NaN state killed
   the whole training run instead of ending one episode -- the two paths a bad
   policy takes most often. Neither had a test. Fixed; three regression tests
   added, verified to fail without the fix.

2. **Reverse-mode AD cannot differentiate TORAX.** `fixed_time_step` runs a
   `jax.lax.while_loop` with a dynamic trip count, which `jax.value_and_grad`
   refuses outright. Switched to forward mode, which is also the cheaper
   direction for four unknowns against a scalar loss.

3. **`numerics.adaptive_dt=True` makes every gradient NaN.** The solver-retry
   branch has a NaN tangent; the primal is fine. This made `gate_identify`
   report all four candidates DEAD. Same one-step JVP: `nan` with adaptive_dt
   on, 0.765 with it off. `build_loss` now forces it off and says why.

4. **`update_provider_from_mapping` dispatches on leaf type and will not
   coerce.** `plasma_composition.Z_eff` is a `TimeVaryingArray`, so the bare
   float the harness passed raised immediately. Wrappers are now resolved once,
   from the provider, per path. Note `TimeVaryingArrayUpdate.value` is 2-D
   `(t, rhon)` while `rho_norm` is 1-D `(rhon,)`.

5. **`gate_identify` question 1 was a FALSE NEGATIVE.** It only tried
   `model_name='prescribed'`, which does not exist in 1.4.3 (the tags are
   qlknn / constant / combined), and concluded the closure-free loop could not
   close. It can: `ConstantTransportModel` declares chi_i/chi_e/D_e/V_e as
   `TimeVaryingArray` and accepts a full chi(rho) profile. Verified end to end.

6. **`--subsample` had never run.** Observations were sliced radially but the
   prediction was not, so the loss subtracted (8, 25) from (8, 7). `build_loss`
   now takes `rho_indices`. This is the run RUNBOOK 5b calls the one that keeps
   every downstream claim honest.

7. **The flat-gradient guard in `chi_from_flux` never fired.**
   `min_gradient=1e-8` is an ABSOLUTE threshold ~7 decades below any real
   gradient, and it is dimensional, so no default could be right for both eV and
   keV. chi is a ratio and one near-zero denominator sets the mean by itself: at
   3% noise a single point landed at 0.4% of the median gradient and came back
   105x off, dragging the mean error to 4.49 while the median stayed at 0.28.
   The signature was non-monotonicity -- 5% noise scored WORSE than 8%, because
   whether such a point occurs is luck. The guard is now relative to the
   profile's own median. Seed-averaged error before: 0.011 / 1.386 / 2.414 /
   1.818; after: 0.011 / 0.871 / 1.075 / 1.346 (monotone, clean data
   unchanged). Figure: `results/audit/chi_gradient_guard.png`.

8. **`peak_rss_mb()` returned 0.0 on Windows.** `resource` is POSIX-only, so
   every gate printed "peak RSS: 0 MB" -- a memory figure that looks measured
   and is not. psutil fallback added; the authority gate now reports 810 MB.

9. **The stated federated payload was wrong.** README said 162 parameters /
   648 bytes; the real figure is 194 / 776. `_observe()` gained the tracking
   error and the current target at some point and the claim was not updated.
   The test only asserted "< 4096 bytes", so nothing caught it. It now pins the
   exact count.

## Results that are findings, not bugs

  * **0/4 candidate unknowns are identifiable at 10%.** Z_eff and
    `resistivity_multiplier` are collinear at |r| = 0.998 -- they are very
    nearly the same parameter against this observation set. Fisher condition
    number 5.5e4. RUNBOOK 5a predicted exactly this class of casualty.
  * The CRLB screen then PREDICTED the recovery outcome: it rated
    `Qei_multiplier` the least identifiable (76% relative bound), and that is
    precisely the parameter that fails to recover (7.5% clean, 22.9% noisy)
    while Z_eff comes back to ~1%. The screen is doing its job.
  * The classical chi estimator is genuinely noise-sensitive: ~1.3% error on
    clean data, ~90% at 3% noise even after the guard fix. That is the regime
    the inverse-PINN arm was built for, so the open question in
    `docs/pinn_comparison.md` is sharper now, not weaker.

## Not run

`gate0_bench.py` (CPU/GPU comparison -- this venv is CPU-only) and
`gate_headroom.py` (blocking, overnight). Headroom is not worth running until
the setpoint question above is settled: it would measure `diiid_like` only.



---

## Gate 0d -- actuator authority, task 'easy' (PASSED)

*2026-09-15 01:50:41*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
task     : easy  (mode band_fraction)
setpoint : constant, base 0.5, amplitude 0, tolerance 0.15
           (fractions of each device's measured band -- resolved per device below)
episode  : 10s in 0.5s steps

diiid_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.499   q95   5.85   fgw 0.431   20/20 steps
  cmd  25%  P_aux    6.50 MW   beta_N    1.270   q95   5.72   fgw 0.431   20/20 steps
  cmd  50%  P_aux   13.00 MW   beta_N    2.140   q95   5.90   fgw 0.431   20/20 steps
  cmd  75%  P_aux   19.50 MW   beta_N    3.005   q95   5.76   fgw 0.431    2/20 steps  VIOLATED@2
  cmd 100%  P_aux   26.00 MW   beta_N    3.190   q95   5.71   fgw 0.435    1/20 steps  VIOLATED@1
  resolved target beta_N 1.500 +-0.3001
  -> PASS

iter_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.459   q95   4.62   fgw 0.524   20/20 steps
  cmd  25%  P_aux   13.25 MW   beta_N    0.560   q95   4.60   fgw 0.524   20/20 steps
  cmd  50%  P_aux   26.50 MW   beta_N    0.669   q95   4.59   fgw 0.524   20/20 steps
  cmd  75%  P_aux   39.75 MW   beta_N    0.785   q95   4.58   fgw 0.524   20/20 steps
  cmd 100%  P_aux   53.00 MW   beta_N    0.904   q95   4.58   fgw 0.524   20/20 steps
  resolved target beta_N 0.682 +-0.0667
  -> PASS

sparc_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.403   q95   4.05   fgw 0.425   20/20 steps
  cmd  25%  P_aux    7.50 MW   beta_N    0.422   q95   4.04   fgw 0.425   20/20 steps
  cmd  50%  P_aux   15.00 MW   beta_N    0.441   q95   4.02   fgw 0.425   20/20 steps
  cmd  75%  P_aux   22.50 MW   beta_N    0.460   q95   4.01   fgw 0.425   20/20 steps
  cmd 100%  P_aux   30.00 MW   beta_N    0.479   q95   4.00   fgw 0.425   20/20 steps
  resolved target beta_N 0.441 +-0.0114
  -> PASS

tcv_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.656   q95   4.53   fgw 0.406   20/20 steps
  cmd  25%  P_aux    1.45 MW   beta_N    2.003   q95   4.42   fgw 0.406   20/20 steps
  cmd  50%  P_aux    2.90 MW   beta_N    3.560   q95   4.42   fgw 0.406    1/20 steps  VIOLATED@1
  cmd  75%  P_aux    4.35 MW   beta_N    5.151   q95   4.44   fgw 0.406    1/20 steps  VIOLATED@1
  cmd 100%  P_aux    5.80 MW   beta_N    6.739   q95   4.46   fgw 0.406    1/20 steps  VIOLATED@1
  resolved target beta_N 1.578 +-0.2765
  -> PASS

wrote results\authority\authority_easy.json
wrote results\authority\authority_easy.png

summary: 4/4 devices can reach the 'easy' setpoint
peak RSS: 810 MB

PASS: every device's actuators bracket the setpoint with margin.
```


---

## FIXED -- device thermal normalisation; gate 0d and Phase 1 now pass 4/4

*2026-09-15 02:00:34*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=dfc39a0

Follow-up to "AUDIT -- first live-TORAX run". The blocking finding there is now
fixed and every gate that can run, passes.

## Before / after

| gate | before | after |
|---|---|---|
| `gate_authority.py --task easy` | FAIL 3/4 devices | **PASS 4/4** |
| `gate_authority.py` moderate / hard / brutal | not resolvable | **PASS 4/4** each |
| `gate1_thermal.py` (`moderate`) | FAIL | **PASS 4/4** |
| `pytest` | 364 | **365** |
| `gate0_env` / `gate0_jit` | PASS | PASS (unchanged) |

## Root cause

Density was normalised per device; temperature was not. `build_config` wrote a
hard-coded `T_i_ped = T_e_ped = 1.0` keV and `initial_T_keV = 6.0` for all four
devices. Once density is a Greenwald fraction,

    beta_N ~ n T a / (B_0 I_p),  n = f_GW I_p / (pi a^2)
        =>  beta_N ~ f_GW * T / (a * B_0)

so one absolute temperature places four devices at normalised pressures
spanning a factor of 30. Measured beta_N at zero command tracked 1/(a*B_0)
almost exactly across the set, which is what confirmed it:

    device       a*B_0   1/(a*B_0)   beta_N
    iter_like    10.60      0.094     0.149
    sparc_like    6.95      0.144     0.178
    diiid_like    1.34      0.746     0.741
    tcv_like      0.36      2.778     3.780

Ratio of 1/(a*B_0) across ITER:TCV is 29.4; ratio of measured beta_N is 25.4.

A second, independent measurement settled which knob to turn. Holding the
pedestal at 4 keV, `initial_T_keV` of 6 gave beta_N = 0.409 and 24 gave 0.410 --
identical. The plasma relaxes to the pedestal-supported equilibrium within a
shot, so **the pedestal is the lever and the initial condition is not.**

## Fix 1 -- pedestal temperature scales as a*B_0

Anchored on ITER's real baseline H-mode pedestal (4.5 keV), the scaling
reproduces every other device's published pedestal with no per-device tuning:

    device        scaled     published
    iter_like     4.50 keV   ~4.5   (ITER baseline H-mode)
    sparc_like    2.95 keV   ~3-5   (SPARC V2 predictions)
    diiid_like    0.57 keV   ~0.5-1 (DIII-D H-mode)
    tcv_like      0.15 keV   ~0.1-0.3 (TCV)

That agreement is the reason to believe the scaling rather than the convenience
of it. `initial_T_keV` and the edge BC are derived from the pedestal, so the
starting profile is consistent with the equilibrium it relaxes onto. The
per-shot disturbance now perturbs the device's OWN nominal core temperature; a
fixed 6.0 keV there had been a +-5% wobble on ITER and a factor of 13 on TCV.

Effect: `tcv_like` went from 3.78..10.5 (violating at zero command, every
episode dead on step 1) to 0.66..6.74, surviving the full shot.

## Fix 2 -- SPARC got its heating back

`p_nbi = 0.1e6` was correct (SPARC's NBI is nominally zero, its baseline is
~25 MW of ICRF) and `icrh.available = False` was correct (TORAX 1.4.3 has no
independent ion-cyclotron source, so it would have aliased onto `generic_heat`).
Composed, they left SPARC with 5.1 MW of ECRH against B_0 = 12.2 T and an
actuator authority of 0.012 in beta_N. Neither decision's documentation
mentioned the other.

`generic_heat` is not "the neutral beam" -- it is a generic Gaussian deposition,
and what it represents is *the device's primary auxiliary heating*. Renamed
`nbi` -> `aux_heat` and given each device its real primary power (SPARC: 25 MW
ICRF). This is also what makes role-matched federation honest: channel 0 now
means the same thing on every device, which a per-device name would have
quietly broken. Authority 0.012 -> 0.076 under `constant`, 0.133 under `qlknn`.

## Fix 3 -- setpoint AND tolerance are fractions of the measured band

The tolerance half matters as much as the setpoint half, and was the part not
visible in the original audit. An absolute tolerance of 0.15 in beta_N is

    diiid_like     7.5% of its control authority
    tcv_like       8.1%
    iter_like     33.7%
    sparc_like   197.4%

so even a reachable setpoint would have made every preset a different
difficulty on every device -- and shots-to-threshold would then be measuring
device calibration rather than learning, which is precisely the quantity
SPEC.md 5 compares across the federation.

`TaskSpec` gained `setpoint_mode` ("band_fraction", default, or "absolute") and
`resolve_for(device)`. Presets now hold fractions; the env resolves them at
construction, so every downstream consumer still reads absolute beta_N and
needs to know nothing about normalisation. `tolerance / authority` is now
identical across devices for a given preset (30 / 15 / 12 / 10 / 8 %).

A `__post_init__` guard rejects any schedule whose span plus tolerance leaves
(0, 1). It caught a bad `brutal` preset immediately, which is the same class of
bug one level up.

## Fix 4 -- `use_pereverzev` for stiff transport

TORAX warned on every `qlknn` shot that the linear solver needs the
Pereverzev-Corrigan terms, and the warning was being ignored. Now set whenever
the solver is linear and transport is not `constant`.

## Findings, not bugs

**Profile stiffness eats the large devices' control authority.** Same device,
same everything but transport:

    device        constant        qlknn
    iter_like     0.444 wide      0.011 wide
    sparc_like    0.076           0.133
    diiid_like    2.001 (capped)  0.758
    tcv_like      1.844 (capped)  1.788

Above the critical gradient qlknn's transport rises steeply, so extra power
raises transport rather than temperature. On `iter_like` that leaves 0.011 of
beta_N for 53 MW. It is physics, not a bug -- but a target of
beta_N = 0.811 +- 0.0011 is a well-posed simulation task and a meaningless
experimental one, since no diagnostic resolves beta_N to a part in a thousand.
`gate_authority` now reports that as an explicit WARNING rather than passing it
silently.

This is also why `BETA_N_BANDS` is keyed by **task as well as device**: the band
is a property of the plant and the task sets part of the plant. Keying by
device alone made `brutal` resolve against `easy`'s band and fail 3 of 4
devices for a reason that looked identical to the original bug. Not circular --
the band depends only on the CONFIG half of a task, never on its setpoint.

## What Phase 1 looks like now

`tcv_like`, which previously scored exactly -550.00 for every policy and every
constant-power baseline:

    iter   0  mean  -214.69   best  0.00   steps 12.7   violations 40.0%
    iter   1  mean  -101.65   best  0.00   steps 16.2   violations 20.0%
    iter   2  mean    -0.49   best  0.00   steps 20.0   violations  0.0%
    iter   3  mean    -0.00   best  0.00   steps 20.0   violations  0.0%

`gate1_thermal.py --task moderate`, all four devices PASS:

    device        best baseline    trained    |beta_N - target|   tolerance
    iter_like        -0.23          -0.19          0.052            0.24
    sparc_like       -0.00          -0.00          0.006            0.24
    diiid_like       -1.43          -0.10          0.195            0.24
    tcv_like         -2.07          -0.13          0.192            0.24

## Still open

  * **Headroom is unmeasured.** Phase 1 passing quickly everywhere is a hint
    the presets may be too easy -- which is exactly the blocking question
    `gate_headroom.py` exists to answer. It has not been run.
  * `OPERATING_POINTS` is still the nominal hand-written table, so the
    similarity weighting and the Phase 5 bandwidth check are computed from
    numbers that are not TORAX output. RUNBOOK's "Before Phase 5" prompt says
    to re-derive them; that has not been done.
  * The stiff presets (`hard`, `brutal`) are near-uncontrollable on
    `iter_like`. Reported, not worked around.



---

## Gate 0d -- actuator authority, task 'easy' (PASSED)

*2026-09-15 02:00:58*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=dfc39a0

```
task     : easy  (mode band_fraction)
setpoint : constant, base 0.5, amplitude 0, tolerance 0.15
           (fractions of each device's measured band -- resolved per device below)
episode  : 10s in 0.5s steps

diiid_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.499   q95   5.85   fgw 0.431   20/20 steps
  cmd  25%  P_aux    6.50 MW   beta_N    1.270   q95   5.72   fgw 0.431   20/20 steps
  cmd  50%  P_aux   13.00 MW   beta_N    2.140   q95   5.90   fgw 0.431   20/20 steps
  cmd  75%  P_aux   19.50 MW   beta_N    3.005   q95   5.76   fgw 0.431    2/20 steps  VIOLATED@2
  cmd 100%  P_aux   26.00 MW   beta_N    3.190   q95   5.71   fgw 0.435    1/20 steps  VIOLATED@1
  resolved target beta_N 1.500 +-0.3001
  -> PASS

iter_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.459   q95   4.62   fgw 0.524   20/20 steps
  cmd  25%  P_aux   13.25 MW   beta_N    0.560   q95   4.60   fgw 0.524   20/20 steps
  cmd  50%  P_aux   26.50 MW   beta_N    0.669   q95   4.59   fgw 0.524   20/20 steps
  cmd  75%  P_aux   39.75 MW   beta_N    0.785   q95   4.58   fgw 0.524   20/20 steps
  cmd 100%  P_aux   53.00 MW   beta_N    0.904   q95   4.58   fgw 0.524   20/20 steps
  resolved target beta_N 0.682 +-0.0667
  -> PASS

sparc_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.403   q95   4.05   fgw 0.425   20/20 steps
  cmd  25%  P_aux    7.50 MW   beta_N    0.422   q95   4.04   fgw 0.425   20/20 steps
  cmd  50%  P_aux   15.00 MW   beta_N    0.441   q95   4.02   fgw 0.425   20/20 steps
  cmd  75%  P_aux   22.50 MW   beta_N    0.460   q95   4.01   fgw 0.425   20/20 steps
  cmd 100%  P_aux   30.00 MW   beta_N    0.479   q95   4.00   fgw 0.425   20/20 steps
  resolved target beta_N 0.441 +-0.0114
  -> PASS

tcv_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.656   q95   4.53   fgw 0.406   20/20 steps
  cmd  25%  P_aux    1.45 MW   beta_N    2.003   q95   4.42   fgw 0.406   20/20 steps
  cmd  50%  P_aux    2.90 MW   beta_N    3.560   q95   4.42   fgw 0.406    1/20 steps  VIOLATED@1
  cmd  75%  P_aux    4.35 MW   beta_N    5.151   q95   4.44   fgw 0.406    1/20 steps  VIOLATED@1
  cmd 100%  P_aux    5.80 MW   beta_N    6.739   q95   4.46   fgw 0.406    1/20 steps  VIOLATED@1
  resolved target beta_N 1.578 +-0.2765
  -> PASS

wrote results\authority\authority_easy.json
wrote results\authority\authority_easy.png

summary: 4/4 devices can reach the 'easy' setpoint
peak RSS: 804 MB

PASS: every device's actuators bracket the setpoint with margin.
```


---

## Headroom gate -- iter_like (FAILED)

*2026-09-15 21:07:31*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=5651de1

```
device      : iter_like
tasks       : ['easy']
seeds       : 1   shot budget: 600
matrix size : 5 conditions x 4 devices x 1 seeds

=== easy  (difficulty 2/5, 20 steps/shot, constant, setpoint constant) ===
  seed 0:
    396.8s

  0.66 s/shot  ->  2.2 h serial for 5x4x1 runs x 600 shots

  threshold (at 80% of plateau): -0.151
  isolated shots-to-threshold: 240 shots [95% CI 240-240] (window=25, persistence=5)
    reached by 1/1 seeds
  
  headroom     (median >= 1000 shots)      : FAIL
  measurable   (>=80% seeds converge, <= 20000): PASS
  affordable   (matrix <= 72 h serial)      : PASS  [2.2 h at 0.66 s/shot]
  
  FAIL: binding constraint is HEADROOM.
  
  Isolated converges in ~240 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

==================================================================
task            isolated shots   reach  matrix h  verdict
easy                       240   100%       2.2  FAIL (headroom)

FAIL: no task in this sweep is usable. DO NOT run the Phase 5 matrix.

  Isolated converges in ~240 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

  Next preset up is 'moderate' -- rerun with --tasks moderate
```


---

## Design change: the three-level hierarchy is removed

*2026-09-15 21:22:32*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=7453fb2

A design decision, not a measurement -- recorded here because SPEC.md asks for
negative results to be reported rather than tuned away, and a removed
architecture is one.

### What it was, and why

SPEC.md §2 specified three levels:

* **Level 1 -- agents.** One per actuator. Small dense policy nets.
* **Level 2 -- cluster heads.** One per functional department (thermal,
  particle, current, stability, exhaust). Sets profile targets for its agents.
  Slower clock than the agents beneath it.
* **Level 3 -- device head.** Resolves cross-cluster actuator contention.
  Slowest clock.

The reasoning was sound on its face. Tokamak control really is multi-objective,
the objectives really do fight over shared actuators, and PACMAN-style deployed
controllers really do run at 0.2-15 ms. A decomposition by function with an
arbitrator above it is the standard answer to that shape of problem, and the
cluster boundaries were taken from TORAX's own coupled equations rather than
invented.

### What was tried

Nothing was built. Phases 2 and 3 were the phases that would have built it, and
neither was reached. What accumulated instead was evidence, from work aimed at
other questions, that the hierarchy had nothing to act on:

1. **Two live actuators.** `registry.py` exposes `aux_heat` and `ecrh` per
   device. `gas_puff`, `ip` and `icrh` are defined and marked unavailable --
   `icrh` deliberately, because pointing it at `generic_heat` would make two
   actuators alias onto one TORAX config path. TORAX 1.4.3 has exactly two
   independently drivable auxiliary heat sources; there is no third.
2. **One scalar objective.** Every task preset tracks beta_N. The particle and
   current clusters have nothing to want, so there is nothing for a device head
   to arbitrate between.
3. **A flat policy already solves it.** `gate_headroom.py`, easy on iter_like,
   600 shots, 1 seed: an MLPPolicy 9->16->2 driven by hill climbing reaches
   threshold in 240 shots and holds it. Hierarchical decomposition cannot be
   shown to earn its place on a task a flat controller solves in 240 shots, and
   this gate's bias runs the safe way -- a weak optimiser cracking it that fast
   means a strong one will not be slower.
4. **The motivating example is unsimulable.** SPEC.md §2's one concrete case
   for the device head was "thermal wants more NBI, stability wants less".
   Stability -- tearing modes, ELMs, Alfven activity -- is not representable in
   a 1-D core transport solver, the same limit that blocks Phase 7. The
   argument for level 3 had no venue in the chosen plant.
5. **It was never ablated, and could not have been.** The Phase 5 matrix is
   five federation conditions, and the hierarchy sits inside all five. No
   condition would have isolated what it contributed.

### Why removed rather than deferred

Deferring would have left every downstream result with an unmeasured factor in
it: any federation speedup reported from the Phase 5 matrix would be a speedup
of a hierarchy nobody had shown was worth having. Removing it makes the flat
team the thing under test, and makes the hierarchy question a separate
experiment with its own gate.

The federation claim does not depend on the hierarchy. Role matching needs
agents *labelled* by cluster; it does not need heads above them. SPEC.md §1 is
unchanged.

### What replaces it

A flat team: one agent per cluster, owning that cluster's actuators, with a
shared team reward. Cross-cluster conflict is resolved the way it is in
ordinary multi-agent RL. Clusters survive as functional groupings and as the
federation channels -- not as a command layer.

Today's code is already this architecture rather than a shortfall against it:
one thermal agent per device, driving `aux_heat` and `ecrh`.

### The condition for bringing it back

Specific, so this is a decision and not an abandonment. Reinstate a coordinator
when Phase 2 measures a flat team **losing** to a monolithic controller with
the same actuator set on a genuinely multi-objective task -- i.e. when two
agents drive the same actuator toward incompatible targets in a way the shared
reward demonstrably cannot resolve. Phase 3 now exists to make that
measurement, and is allowed to conclude "no".

### Still open

`hfmarl` -- the `H` no longer stands for anything. The repository, the package
and the remote have not been renamed; that touches every import and is a
separate decision. README.md says so explicitly rather than leaving the name to
imply a design that is gone.

### Changed by this

SPEC.md §2 (rewritten), §3 (clusters reframed as channels), §6 Phases 2-3;
PLAN.md Phases 2-3; README.md; METRICS.md; TORAX_NOTES.md;
`scripts/fig_system.py` and `scripts/fig_architecture.py`. No code changed --
there was none to change, which is itself the cheapest evidence that removing
it cost nothing.



---

## Headroom gate -- iter_like (FAILED)

*2026-09-15 21:26:47*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=05ce73d

```
device      : iter_like
tasks       : ['moderate']
seeds       : 1   shot budget: 1200
matrix size : 5 conditions x 4 devices x 1 seeds

=== moderate  (difficulty 3/5, 30 steps/shot, constant, setpoint ramp) ===
  seed 0:
    1136.1s

  0.95 s/shot  ->  6.3 h serial for 5x4x1 runs x 1200 shots

  threshold (at 80% of plateau): -0.214
  isolated shots-to-threshold: 146 shots [95% CI 146-146] (window=25, persistence=5)
    reached by 1/1 seeds
  
  headroom     (median >= 1000 shots)      : FAIL
  measurable   (>=80% seeds converge, <= 20000): PASS
  affordable   (matrix <= 72 h serial)      : PASS  [6.3 h at 0.95 s/shot]
  
  FAIL: binding constraint is HEADROOM.
  
  Isolated converges in ~146 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

==================================================================
task            isolated shots   reach  matrix h  verdict
moderate                   146   100%       6.3  FAIL (headroom)

FAIL: no task in this sweep is usable. DO NOT run the Phase 5 matrix.

  Isolated converges in ~146 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

  Next preset up is 'hard' -- rerun with --tasks hard
```


---

## Headroom gate -- iter_like (FAILED)

*2026-09-15 21:59:36*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=f10b4bb

```
device      : iter_like
tasks       : ['easy', 'moderate']
seeds       : 1   shot budget: 1200
matrix size : 5 conditions x 4 devices x 1 seeds

=== easy  (difficulty 2/5, 20 steps/shot, constant, setpoint constant) ===
  seed 0:
    736.1s

  0.61 s/shot  ->  4.1 h serial for 5x4x1 runs x 1200 shots

  competence criterion: |beta_N - target| <= 0.067 (the task tolerance), smoothed over 25
  isolated shots-to-threshold: 80 shots [95% CI 80-80] (window=25, persistence=5)
    reached by 1/1 seeds
  
  headroom     (median >= 1000 shots)      : FAIL
  measurable   (>=80% seeds converge, <= 20000): PASS
  affordable   (matrix <= 72 h serial)      : PASS  [4.1 h at 0.61 s/shot]
  
  FAIL: binding constraint is HEADROOM.
  
  Isolated converges in ~80 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

  curve: start -0.160 -> plateau -0.148 (range +0.011)
  best-so-far last improved at shot 1187
  plateau-relative threshold -0.151: 240 shots [95% CI 240-240] (window=25, persistence=5)
    ^ informational only -- each task anchors this to its own curve, so it cannot rank tasks against each other.

=== moderate  (difficulty 3/5, 30 steps/shot, constant, setpoint ramp) ===
  seed 0:
    1084.9s

  0.90 s/shot  ->  6.0 h serial for 5x4x1 runs x 1200 shots

  competence criterion: |beta_N - target| <= 0.053 (the task tolerance), smoothed over 25
  NO CROSSING: never reached threshold 0.0534 in 1 seeds (censored at 1200 shots)
  A censored run cannot separate 'too hard' from 'too
  short'. Scale the budget from the rate above and re-run.

==================================================================
task            isolated shots   reach  matrix h  verdict
easy                        80   100%       4.1  FAIL (headroom)
moderate           no crossing       -         -  INCONCLUSIVE (budget)

FAIL: no task in this sweep is usable. DO NOT run the Phase 5 matrix.

  Isolated converges in ~80 shots, below the
  1000-shot floor. Any federation speedup here is
  within campaign noise and no fusion audience would act on it.
  THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:
    * moving setpoint (ramp or step schedule) instead of constant
    * tighter tracking tolerance
    * transport_model='qlknn' -- stiff, nonlinear, genuinely hard
    * enable fusion alpha heating (self-heating nonlinearity)
    * more actuators to coordinate (add the particle cluster)
    * harsher initial conditions, or per-shot disturbances
    * longer episodes, so limits are reachable and must be planned for
  Re-run this gate after each change. Do NOT proceed to Phase 5
  until it passes -- the whole matrix would measure nothing.

  Next preset up is 'moderate' -- rerun with --tasks moderate
```


---

## Gate 0d -- actuator authority, task 'moderate' (PASSED)

*2026-09-15 22:00:16*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=f10b4bb

```
task     : moderate  (mode band_fraction)
setpoint : ramp, base 0.3, amplitude 0.45, tolerance 0.12
           (fractions of each device's measured band -- resolved per device below)
episode  : 15s in 0.5s steps

iter_like
  cmd   0%  P_aux    0.00 MW   beta_N    0.459   q95   4.63   fgw 0.524   30/30 steps
  cmd  25%  P_aux   13.25 MW   beta_N    0.560   q95   4.61   fgw 0.524   30/30 steps
  cmd  50%  P_aux   26.50 MW   beta_N    0.669   q95   4.59   fgw 0.524   30/30 steps
  cmd  75%  P_aux   39.75 MW   beta_N    0.785   q95   4.58   fgw 0.524   30/30 steps
  cmd 100%  P_aux   53.00 MW   beta_N    0.904   q95   4.58   fgw 0.524   30/30 steps
  resolved target beta_N 0.592 +-0.0534
  -> PASS

wrote results\authority\authority_moderate.json

summary: 1/1 devices can reach the 'moderate' setpoint
peak RSS: 810 MB

PASS: every device's actuators bracket the setpoint with margin.
```


---

## Cold start -- easy, 120 joiner shots x 3 seeds, merge rule geomedian

*2026-09-15 22:56:18*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=9c62b47

```
task        : easy (20 steps/shot)
joiners     : ['sparc_like', 'tcv_like']   (leave-one-out)
seeds       : 3   pretrain/incumbent: 100   joiner budget: 120
merge rule  : geomedian   align: True   clip: 2.0

=== sparc_like joins ['diiid_like', 'iter_like', 'tcv_like'] (mean distance 1.070) ===
  cold_scratch                224.7s
  cold_inherit_uniform        685.9s
  cold_inherit_similarity     707.8s

  arm                           shots to track   reach   plateau
  cold_scratch                              81    67%    -0.000
  cold_inherit_uniform                      60    33%    -0.000
  cold_inherit_similarity                never     0%    -0.000

  cold_inherit_uniform vs scratch: 1.35x fewer shots [95% CI 1.23-1.47]
  WARNING: censoring differs: baseline reached 67% of seeds, method 33%. The ratio compares different populations and overstates whichever condition failed more often. Report the reach rates alongside it, or run longer.
    ! censoring differs: baseline reached 67% of seeds, method 33%. The ratio compares different populations and overstates whichever condition failed more often. Report the reach rates alongside it, or run longer.
  cold_inherit_similarity vs scratch: nanx fewer shots [95% CI nan-nan]
  WARNING: method never reached the threshold; no ratio exists
    ! method never reached the threshold; no ratio exists
  inherited from: iter_like 0.449  diiid_like 0.290  tcv_like 0.260

=== tcv_like joins ['diiid_like', 'iter_like', 'sparc_like'] (mean distance 1.374) ===
  cold_scratch                156.1s
  cold_inherit_uniform        688.9s
  cold_inherit_similarity     695.2s

  arm                           shots to track   reach   plateau
  cold_scratch                           never     0%    -0.412
  cold_inherit_uniform                   never     0%    -1.005
  cold_inherit_similarity                never     0%    -0.001

  cold_inherit_uniform vs scratch: nanx fewer shots [95% CI nan-nan]
  WARNING: baseline never reached the threshold; no ratio exists
    ! baseline never reached the threshold; no ratio exists
  cold_inherit_similarity vs scratch: nanx fewer shots [95% CI nan-nan]
  WARNING: baseline never reached the threshold; no ratio exists
    ! baseline never reached the threshold; no ratio exists
  inherited from: diiid_like 0.427  sparc_like 0.334  iter_like 0.239

==========================================================================
The prediction similarity theory makes: a closer joiner inherits more.

joiner          distance   scratch  similarity      gain
sparc_like         1.070        81       never       n/a
tcv_like           1.374     never       never       n/a

wrote C:\Users\mothe\hfmarl-fusion\results\coldstart\easy_pre100_join120_s3_geomedian.json
wrote C:\Users\mothe\hfmarl-fusion\results\coldstart\easy_pre100_join120_s3_geomedian.png
total 3160s
```


---

## Catastrophe transfer -- tcv_like, unseen at 84% of band, 3 seeds

*2026-09-15 23:03:53*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=734eafc

```
target      : tcv_like  (trains only at 20% of its band)
peers       : ['diiid_like', 'iter_like', 'sparc_like']  (ramp across 20%-84% of theirs)
unseen      : evaluation at 84% of the target's band, frozen policy, no learning
task        : easy   seeds: 3
budget      : peers 150/device, target 150, eval 60
merge rule  : geomedian   reject-if-worse: False

  isolated     372.5s
  federated   1112.0s

==========================================================================
SAFE-REGION TRAINING (where the target actually lived)
  isolated  violation rate 5.3%
  federated violation rate 24.4%

THE UNSEEN REGIME (frozen policy, never trained here)
  unseen-regime violation rate: isolated 0.0% (n=180), federated 0.0% (n=180)
    CLAIM FAILS: federation did not reduce violations.

  the target inherited from: diiid_like 0.427  sparc_like 0.334  iter_like 0.239
  total 1485s
wrote C:\Users\mothe\hfmarl-fusion\results\catastrophe\tcv_like_easy_safe0.2_unseen0.84_s3.json
wrote C:\Users\mothe\hfmarl-fusion\results\catastrophe\tcv_like_easy_safe0.2_unseen0.84_s3.png
```


---

## Operating points measured from TORAX -- easy

*2026-09-15 23:32:01*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=ddd2f42

```
task    : easy
levels  : [-1.0, -0.5, 0.0, 0.5, 1.0]  (fraction of every actuator's envelope)
states encoded per step after a 40% settling window

diiid_like
  measured  rho* 4.947e-03  nu* 0.0964  beta_N  1.303  q95  5.81
  nominal   rho* 6.028e-03  nu* 0.0341  beta_N  2.248  q95  3.80
  ratio     rho*   0.82   nu*     2.83   beta_N   0.58
  swept     rho_star  0.003177 .. 0.007138   (2.2x)
  swept     nu_star   0.02337 .. 0.5399   (23.1x)
  swept     beta_N    0.4992 .. 2.14   (4.3x)

iter_like
  measured  rho* 1.221e-03  nu* 0.0256  beta_N  0.675  q95  4.59
  nominal   rho* 1.363e-03  nu* 0.0216  beta_N  1.621  q95  3.22
  ratio     rho*   0.90   nu*     1.19   beta_N   0.42
  swept     rho_star  0.001002 .. 0.001464   (1.5x)
  swept     nu_star   0.01263 .. 0.05558   (4.4x)
  swept     beta_N    0.4595 .. 0.9037   (2.0x)

sparc_like
  measured  rho* 1.300e-03  nu* 0.2458  beta_N  0.441  q95  4.02
  nominal   rho* 1.944e-03  nu* 0.0343  beta_N  0.908  q95  3.01
  ratio     rho*   0.67   nu*     7.18   beta_N   0.49
  swept     rho_star  0.001236 .. 0.001365   (1.1x)
  swept     nu_star   0.2025 .. 0.3004   (1.5x)
  swept     beta_N    0.4028 .. 0.4789   (1.2x)

tcv_like
  measured  rho* 1.066e-02  nu* 0.6293  beta_N  1.330  q95  4.47
  nominal   rho* 1.419e-02  nu* 0.0950  beta_N  1.174  q95  3.32
  ratio     rho*   0.75   nu*     6.63   beta_N   1.13
  swept     rho_star  0.007424 .. 0.01531   (2.1x)
  swept     nu_star   0.1534 .. 2.581   (16.8x)
  swept     beta_N    0.6563 .. 2.003   (3.1x)

==========================================================================
Is one static coordinate per device enough?

device          own excursion    nearest peer  verdict
diiid_like                nan           1.421  A POINT IS NOT ENOUGH
iter_like               0.866           1.270  A POINT IS NOT ENOUGH
sparc_like              0.226           1.270  a point is defensible
tcv_like                  nan           1.421  A POINT IS NOT ENOUGH

If a device's own excursion is comparable to its distance from the
nearest peer, a single point per device cannot support SPEC.md 4b and
the weighting needs the distribution of visited states instead.

MEASURED_OPERATING_POINTS = {
    "diiid_like": {
        "rho_star": 0.004946757055757571,
        "nu_star": 0.09642686366170526,
        "beta_N": 1.3028016145317471,
        "q95": 5.814874957906761,
        "mach": 0.0
    },
    "iter_like": {
        "rho_star": 0.001221035319252584,
        "nu_star": 0.025616282218766946,
        "beta_N": 0.674733289624865,
        "q95": 4.58804227738743,
        "mach": 0.0
    },
    "sparc_like": {
        "rho_star": 0.0013001390292053906,
        "nu_star": 0.24581659723449806,
        "beta_N": 0.44070435649708645,
        "q95": 4.024846080323594,
        "mach": 0.0
    },
    "tcv_like": {
        "rho_star": 0.01066265438324514,
        "nu_star": 0.6293307216290219,
        "beta_N": 1.3295309259066872,
        "q95": 4.472320441881702,
        "mach": 0.0
    }
}
total 43s
wrote C:\Users\mothe\hfmarl-fusion\results\operating_points\easy.json
```


---

## Operating points measured from TORAX -- easy

*2026-09-15 23:33:01*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=True  
torax=1.4.3  
commit=ddd2f42

```
task    : easy
levels  : [-1.0, -0.5, 0.0, 0.5, 1.0]  (fraction of every actuator's envelope)
states encoded per step after a 40% settling window

diiid_like
  measured  rho* 4.947e-03  nu* 0.0964  beta_N  1.303  q95  5.81
  nominal   rho* 6.028e-03  nu* 0.0341  beta_N  2.248  q95  3.80
  ratio     rho*   0.82   nu*     2.83   beta_N   0.58
  swept     rho_star  0.003177 .. 0.007138   (2.2x)
  swept     nu_star   0.02337 .. 0.5399   (23.1x)
  swept     beta_N    0.4992 .. 2.14   (4.3x)

iter_like
  measured  rho* 1.221e-03  nu* 0.0256  beta_N  0.675  q95  4.59
  nominal   rho* 1.363e-03  nu* 0.0216  beta_N  1.621  q95  3.22
  ratio     rho*   0.90   nu*     1.19   beta_N   0.42
  swept     rho_star  0.001002 .. 0.001464   (1.5x)
  swept     nu_star   0.01263 .. 0.05558   (4.4x)
  swept     beta_N    0.4595 .. 0.9037   (2.0x)

sparc_like
  measured  rho* 1.300e-03  nu* 0.2458  beta_N  0.441  q95  4.02
  nominal   rho* 1.944e-03  nu* 0.0343  beta_N  0.908  q95  3.01
  ratio     rho*   0.67   nu*     7.18   beta_N   0.49
  swept     rho_star  0.001236 .. 0.001365   (1.1x)
  swept     nu_star   0.2025 .. 0.3004   (1.5x)
  swept     beta_N    0.4028 .. 0.4789   (1.2x)

tcv_like
  measured  rho* 1.066e-02  nu* 0.6293  beta_N  1.330  q95  4.47
  nominal   rho* 1.419e-02  nu* 0.0950  beta_N  1.174  q95  3.32
  ratio     rho*   0.75   nu*     6.63   beta_N   1.13
  swept     rho_star  0.007424 .. 0.01531   (2.1x)
  swept     nu_star   0.1534 .. 2.581   (16.8x)
  swept     beta_N    0.6563 .. 2.003   (3.1x)

==========================================================================
Is one static coordinate per device enough?

device          own excursion    nearest peer  verdict
diiid_like              2.055           1.421  A POINT IS NOT ENOUGH
iter_like               0.866           1.270  A POINT IS NOT ENOUGH
sparc_like              0.226           1.270  a point is defensible
tcv_like                1.806           1.421  A POINT IS NOT ENOUGH

If a device's own excursion is comparable to its distance from the
nearest peer, a single point per device cannot support SPEC.md 4b and
the weighting needs the distribution of visited states instead.

MEASURED_OPERATING_POINTS = {
    "diiid_like": {
        "rho_star": 0.004946757055757571,
        "nu_star": 0.09642686366170526,
        "beta_N": 1.3028016145317471,
        "q95": 5.814874957906761,
        "mach": 0.0
    },
    "iter_like": {
        "rho_star": 0.001221035319252584,
        "nu_star": 0.025616282218766946,
        "beta_N": 0.674733289624865,
        "q95": 4.58804227738743,
        "mach": 0.0
    },
    "sparc_like": {
        "rho_star": 0.0013001390292053906,
        "nu_star": 0.24581659723449806,
        "beta_N": 0.44070435649708645,
        "q95": 4.024846080323594,
        "mach": 0.0
    },
    "tcv_like": {
        "rho_star": 0.01066265438324514,
        "nu_star": 0.6293307216290219,
        "beta_N": 1.3295309259066872,
        "q95": 4.472320441881702,
        "mach": 0.0
    }
}
total 46s
wrote C:\Users\mothe\hfmarl-fusion\results\operating_points\easy.json
```


---

## Conventional baseline: easy and moderate need no learning, and the tolerance scheme is broken

*2026-09-16 00:43:38*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=d33dccb

The baseline arm PROTOCOL.md puts first, run before any learned arm. It
invalidates most of what this project has measured and identifies a defect in
the task design that nothing else had surfaced.

### The result

5 calibration shots per device, ZERO training shots, 20 independently seeded
evaluation shots with a 0.05 disturbance. Joint endpoint: completed AND
contained AND tracking inside that device's own tolerance.

    joint success        easy   moderate     hard
    diiid_like          100.0%    100.0%   100.0%
    iter_like           100.0%    100.0%     0.0%
    sparc_like            0.0%      0.0%     0.0%
    tcv_like            100.0%    100.0%     0.0%

No violations and 100% completion everywhere. Every failure above is a
tracking failure.

### 1. `easy` and `moderate` need no learning at all

A PI controller with static feedforward, built from five calibration shots,
solves three of four devices perfectly on both.

`moderate` is the task on which hill climbing never reached competence in 1200
shots. So the "headroom" this project has been hunting was never task
difficulty -- it was the weakness of black-box search on a problem with a
straightforward classical solution. The headroom gate's verdicts, the 240-shot
and 146-shot crossings, the censored cold-start runs: all of them were
measuring the optimiser, not the plant.

Every learned result recorded before today was obtained on a task a PI
controller solves outright. None of them can support a claim about federation.

### 2. The tolerance scheme makes the task impossible for narrow-band devices

`tolerance` is a FRACTION of each device's measured beta_N band, introduced so
that one preset would be comparably hard everywhere. Measured, it does the
opposite, because the bands differ by a factor of ~600:

    easy, tolerance fraction 0.15
      device       band width   resulting tolerance   median error   error/tol
      sparc_like       0.0761              0.01141         0.0257       225%
      iter_like        0.4444              0.06666         0.0356        53%
      diiid_like       4.9337              0.74006         0.0124         2%
      tcv_like         6.6925              1.00388         0.0364         4%

The tolerance scales with band width. The achievable error does not -- it is
set by disturbance and tracking lag, which are absolute. So a wide-band device
gets a tolerance 25x looser than it needs and a narrow-band device gets one it
cannot meet. sparc_like is asked to hold beta_N inside 15% of everything it can
reach, with the lowest actuator gain in the set by a factor of six.

Under `qlknn` (the `hard` and `brutal` presets) this becomes acute: profile
stiffness collapses iter_like's band to 0.0109, so its tolerance is 0.00109 and
its gain is 0.0045 beta_N per unit command. Nothing -- classical or learned --
tracks that. iter_like's 0% on `hard` is an ill-posed task, not a failed
controller.

### 3. What changes

**No preset in the current ladder can host the cold-start experiment.** `easy`
and `moderate` are solved without learning; `hard` and `brutal` are ill-posed
on three of four devices. Running the headline experiment on any of them would
produce a number about the wrong thing.

The fix is to derive the tolerance from MEASURED achievable performance rather
than from a band fraction: tolerance = k x (achievable error floor), where the
floor is what a perfect-knowledge controller attains on that device. That makes
the task equally demanding by construction rather than by assumption, and it is
measurable with the machinery already here -- `gate_authority.tracking_floor`
computes the floor, and this baseline measures what a real controller achieves
against it.

Stated before the runs rather than after: the headline experiment goes on the
easiest task whose conventional joint success is clearly below ceiling on the
devices used. Anything easier measures nothing; anything harder spends compute
without adding evidence.

### 4. Where a learned controller should legitimately win

Worth recording now, so the target is chosen on physics rather than on
whichever task happens to flatter the method. This baseline commands EVERY
actuator at the same scalar level (`np.full(n_actions, u)`). A learned policy
outputs a command per actuator. On a task where the optimal split between
aux_heat and ecrh matters -- different deposition profiles, different time
constants -- the classical controller as built cannot express the answer and a
learned one can. That is a fair advantage rooted in the plant rather than in
the tuning, and it is the direction the redesigned task should point.

### Provenance

`scripts/exp_conventional.py`; artifacts with per-shot records under
`results/conventional/20260916-003939_d33dccb` (easy),
`...-004015_d33dccb` (moderate), `...-004233_d33dccb` (hard). Gains are fixed
constants in `agents/conventional.py`, chosen once and never tuned on the
device being evaluated. `brutal` was still running when this was written.



---

## Correction: sparc_like is a controller failure, not an ill-posed task; and the 'floor' was not a floor

*2026-09-16 00:47:44*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=0db9b14

Two corrections to the entry above, both found by measuring the open-loop
reference on `moderate` rather than reasoning about it.

### Correction 1: sparc_like is NOT a broken rung

The previous entry concluded that sparc_like's 0% was an ill-posed task -- "a
gap measured against 0% on this device says nothing about federation; it says
the task is not posed for this machine". Measured, that is wrong on `moderate`:

    moderate      open-loop ref   PI achieved   tolerance
    diiid_like           0.0305        0.0340      0.2401
    iter_like            0.0260        0.0425      0.0534
    sparc_like           0.0015        0.0209      0.0091
    tcv_like             0.1189        0.0373      0.2212

A perfect static feedforward holds sparc_like to 0.0015 against a tolerance of
0.0091 -- six times inside it. The task is comfortably achievable on that
device. The PI controller reaches 0.0209, two and a half times outside, because
its gain rule `Kp = 0.6 / slope` gives Kp = 15.7 on a device whose actuator
gain is 0.0381 beta_N per unit, the lowest in the set by a factor of six.

So sparc_like's failure is a CONTROLLER failure on an achievable task, not an
ill-posed task. That is the opposite of what was recorded, and it is better
news: sparc_like on `moderate` is a device where a better controller has
demonstrable room, which is exactly what the cold-start claim needs.

The earlier conclusion still stands for `hard` and `brutal`, where iter_like's
band collapses to 0.0109 under qlknn and its tolerance to 0.00109 against a
gain of 0.0045 -- those remain ill-posed.

### Correction 2: `tracking_floor` was not a floor, and said it was

Its docstring claimed "the best mean tracking error ANY controller could
achieve" and the registry comment said "no controller with the same information
does better, so this is a property of the plant". Both are false, and the same
table disproves them: on tcv_like the PI controller reaches 0.0373 against the
open-loop 0.1189 -- three times better -- because feedback corrects lag that
open-loop control cannot see.

It measures ONE controller: perfect static feedforward, no feedback. That is
still worth having, because it is the performance the plant hands you free from
a sweep, so a controller that cannot beat it is adding nothing. It bounds
nothing from below.

Renamed in effect rather than in name: docstrings and the registry comment now
say open-loop reference, and `gate_authority`'s FAIL message no longer claims
"no controller can pass this task" -- it warns about the setpoint rate and says
feedback may still close the gap.

This matters beyond the wording. A tolerance defined as a multiple of a
supposed floor would have been a tolerance defined as a multiple of one
arbitrary controller's error, which is the defect `tolerance_mode` was
introduced to fix.

### What this means for the task design

`TaskSpec.tolerance_mode` now accepts `floor_multiple`, resolving the tolerance
as a multiple of the measured open-loop reference via
`registry.tracking_floor`, which raises rather than defaulting when the
measurement is missing. But the multiplier cannot be chosen against the
open-loop number alone, since feedback beats it by a device-dependent factor
(0.9x to 3.2x across this set).

The defensible construction, to be stated before any run: set the tolerance so
that a WELL-TUNED CLASSICAL CONTROLLER fails by a stated margin, and report the
classical arm's number beside every learned one. That makes the task require
beating classical control, which is the only thing that makes an RL result on
this plant worth reporting.



---

## A tuned classical controller solves every solvable task: no room for a learned arm in this task family

*2026-09-16 01:14:32*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=8bd6398

The decisive negative result so far, and it closes the current task family.

### A tuned PI controller solves `moderate` on every device

Gain swept over kp_fraction in {0.05, 0.15, 0.3, 0.6, 1.0}, best per device by
joint success. 5 calibration shots, zero training shots, 12 independently
seeded evaluation shots with disturbance.

    device        best kp   joint success   median |err|   tolerance
    diiid_like        0.6          100.0%         0.0340      0.2401
    iter_like        0.05          100.0%         0.0383      0.0534
    sparc_like       0.05          100.0%         0.0066      0.0091
    tcv_like          0.6          100.0%         0.0373      0.2212

Four of four. No violations, 100% completion.

### It also corrects the previous entry again

sparc_like was reported as the one device with room for a better controller,
because the fixed-gain PI reached 0.0209 against a tolerance of 0.0091. At
kp = 0.05 it reaches 0.0066 and passes comfortably. The failure was entirely my
gain rule: kp = 0.6/slope gives Kp = 15.7 on a device whose actuator gain is
0.0381 beta_N per unit, six times lower than anything else in the set.

The lesson generalises beyond this repo. "The classical baseline fails here"
was true of ONE arbitrarily tuned classical controller, and I came within one
experiment of reporting it as a property of the device. A baseline has to be
tuned as carefully as the method it is there to challenge, or it manufactures
room that does not exist.

The sweep also shows the gains are not interchangeable: tcv_like degrades from
100% to 58% between kp 0.6 and 1.0, and diiid_like's error doubles. A single
fixed gain across a device set spanning a factor of 80 in actuator gain was
never going to be fair.

### What this closes

    task        classical joint success (tuned)
    easy        3/4 at fixed gain; sweep running
    moderate    4/4
    hard        1/4, other three ill-posed (bands collapse under qlknn)
    brutal      1/4 or fewer, same cause

There is no task in this repo where a learned controller can demonstrate an
advantage on beta_N tracking. Everything solvable is solved by a PI controller
with five calibration shots; everything unsolved is unsolved because the
tolerance is below what the actuators can deliver.

This is a real finding, not a setback: on this plant, single-input beta_N
tracking is a solved control problem, and an RL result on it would be measuring
nothing. Every learned number this project has produced was produced on such a
task.

### The only honest way forward

A task that genuinely needs more than single-loop control. The candidate is
already in the codebase and unused: `clusters=("thermal", "particle")` puts
`gas_puff` in the actuator bank alongside `aux_heat` and `ecrh`. Heating raises
beta_N; gas raises density toward the Greenwald limit; the two must be traded
off rather than driven together.

That matters because the conventional controller here commands EVERY actuator
at the same scalar level -- with gas in the bank, driving heating up drives gas
up too, which is actively wrong. A learned policy emits one command per
actuator and can split them.

**But that asymmetry is not a fair test on its own**, and it would be easy to
present it as one. A proper classical MIMO controller -- one PI loop per
actuator on its own measured objective, still built from calibration -- can also
split them. If the coupled task is adopted, that controller is the baseline,
not the scalar one, or the comparison is against a straw man for the second
time in one session.

### Provenance

`scripts/exp_conventional.py --task moderate --kp-sweep 0.05,0.15,0.3,0.6,1.0`,
artifacts under `results/conventional/` with per-shot records. The `easy` sweep
is running.



---

## Task ladder complete: no preset leaves room for a learned controller

*2026-09-16 02:09:46*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=aedd95c

The classical arm has now been run on every preset with the gain swept per
device, so the task ladder can be reported as a whole rather than as four
separate entries.

### Joint success of a tuned PI controller, zero training shots

    task        diiid   iter   sparc    tcv     verdict
    easy         100%   100%    100%   100%     solved without learning
    moderate     100%   100%    100%   100%     solved without learning
    hard         100%     0%      0%     0%     ill-posed on three devices
    brutal         0%     0%      0%     0%     ill-posed on all four

No violations and 100% completion in every cell. Every 0% is a tracking
failure, and on `hard` and `brutal` those are failures against tolerances that
sit below what the actuators resolve: iter_like's band collapses to 0.0109
under qlknn, giving a tolerance of 0.00109 against an actuator gain of 0.0045
beta_N per unit command.

Tuning did not rescue the stiff presets. On `hard` the sweep over
{0.05, 0.15, 0.3, 0.6} leaves iter_like at a median error of 0.0045 -- close to
its own gain-per-unit -- and still failing, because the bar is finer than one
unit of command can resolve.

### The consequence, stated plainly

There is no preset in this repository on which a learned controller can
demonstrate an advantage at beta_N tracking. Where the task is well posed, five
calibration shots and no training solve it on every device. Where it is not
solved, nothing solves it.

That is the answer to the question the project set itself, on this plant. It is
a negative result about the TASK FAMILY, not about federation: no method can
show an advantage where the baseline is already at ceiling, and federation was
never given a problem to be better at.

### What was tried to find a problem with room, and why each closed

1. **Harder presets.** `hard` and `brutal` are not harder, they are ill-posed:
   the band-fraction tolerance scheme scales the bar with a band that collapses
   under stiff transport.

2. **A coupled thermal + particle task.** `clusters=("thermal","particle")` is
   already supported and puts `gas_puff` in the bank. Measured, full-range gas
   moves the Greenwald fraction by 0.036 on iter_like against a limit 0.44
   away. The density is boundary-pinned -- `n_e_ped` and the edge condition are
   both fixed Greenwald fractions -- so a core particle source cannot move it.
   Not a wiring bug; a configuration property. Freeing the pedestal and edge
   density would change the physics under every measurement already taken.

3. **Anticipation.** The observation carries normalised time, so a learned
   policy could feed-forward a scheduled setpoint where a PI controller cannot.
   Not tested: the PI already passes on every well-posed preset, so there is no
   error left for anticipation to remove.

### What would have to change for the claim to be testable here

A control problem that single-loop classical control cannot solve. The
honest candidates, in order of how much they disturb what is already measured:

  * free the pedestal and edge density so the particle actuator has authority,
    then set a density target alongside beta_N -- genuinely multi-input, and
    the scalar conventional controller cannot express the split. It needs a
    per-actuator classical baseline built alongside it, or it repeats the
    straw-man error;
  * replace band-fraction tolerances with a criterion derived from measured
    achievable performance, so the stiff presets become hard rather than
    impossible;
  * accept that beta_N tracking on a 1-D core transport model is a solved
    control problem and move the claim to something that is not.

### Provenance

`scripts/exp_conventional.py --kp-sweep`, four runs, artifacts with per-shot
records under `results/conventional/`. Gains fixed in the module and swept per
device; the best per device is reported, so no cell is a detuning artefact.



---

## Cold start, first fold: inheriting is worse than starting from nothing, and two calibration shots beat both

*2026-09-16 02:58:34*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=20ef779

First fold of the five-arm ladder, and the classical arm's confirmation run.
Read together they answer the cold-start question on this plant.

### The ladder, tcv_like joining, `moderate`, 3 seeds, 120-shot joiner budget

    arm                     shots to competence   reach     plateau
    scratch                                  21    100%      -81.98
    single_source                            41    100%      -82.91
    handover_merge                           41    100%      -83.39
    federated_uniform                        41    100%      -83.00
    federated_similarity                     41    100%     -195.16

    single_source vs scratch              0.51x  [95% CI 0.51-5.55]
    handover_merge vs single_source       1.00x  [95% CI 0.27-1.00]
    federated_uniform vs handover_merge   1.00x  [95% CI 0.80-1.00]
    federated_similarity vs uniform       1.00x  [95% CI 0.80-1.24]

**Inheriting is worse than starting from nothing.** Every arm that received a
controller took 41 shots to reach competence; the arm that received nothing
took 21. A ratio below 1 means the upper rung needed MORE shots, and
single_source vs scratch is 0.51x -- the joiner spends its budget unlearning
what it was handed.

The similarity arm's plateau, -195 against -82 for every other arm, says the
same thing more sharply: the physics-weighted aggregate was the worst thing to
be handed, not the best.

### What cannot be concluded from it

The four inheriting arms all report exactly 41, and that is an artefact.
Evaluations fire every 10 shots, so competence can only take values on that
grid -- 11, 21, 31, 41. Differences smaller than 10 shots are invisible, and
four arms sharing a value is what the grid produces, not a measurement that
they are equal. The three 1.00x ratios between adjacent rungs carry no
information.

The 20-shot gap between scratch and the rest is larger than the grid spacing,
so that one is real.

`--eval-every` is now a flag defaulting to 5 rather than a constant at 10. It
costs budget -- a fifth of the joiner's shots become evaluations, charged like
any other -- and that is the trade: 24 resolvable points instead of 12.

### The classical arm, confirmed at n=100

2 calibration shots, no training, 100 independently seeded evaluation shots per
device on `moderate`:

    device        joint success   violations   completed   median |err|
    diiid_like           100.0%         0.0%      100.0%         0.0581
    iter_like            100.0%         0.0%      100.0%         0.0418
    sparc_like           100.0%         0.0%      100.0%         0.0066
    tcv_like             100.0%         0.0%      100.0%         0.0634

Zero failures in 100 shots bounds the true failure rate under 3.0% at 95%,
where the earlier 10-shot runs bounded it only under 25.9%.

### The comparison that matters

    arm                     target shots   violations   joint success
    conventional                       2         0.0%   100% (<=3% fail)
    scratch (learned)                 21            -               -
    every federated arm               41            -               -

On this plant, a new machine reaches competent, safe control from two open-loop
calibration discharges. Learning from its own shots costs ten times that.
Inheriting a controller from other machines costs twenty times that and makes
it worse.

The cold-start claim is disproved here, and the reason is not that federation
failed: it is that the problem does not need it. A claim of the form "fewer
shots than the alternative" requires the alternative to need shots.



---

## Correction: the classical arm costs 7 calibration shots, not 2

*2026-09-16 03:37:45*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=f3de943

A correction to the headline number, found by auditing my own accounting
against PROTOCOL.md rather than by running anything.

### "Two calibration shots" undercounts

The classical arm was reported as reaching the joint endpoint on every device
from two calibration shots and no training. The two shots are real -- they are
the static command-to-beta_N map that `ConventionalController.from_sweep` fits.

But the JOINT ENDPOINT itself needs a tolerance, and on `moderate` the
tolerance is a band fraction: `resolve_for` divides it by that device's
measured beta_N band, and `BETA_N_BANDS` is measured by driving the device from
zero to full command -- five shots per device per task, in
`scripts/gate_authority.py`.

Those five shots are fired on the new machine, before any controller is
installed, to characterise the plant. PROTOCOL.md 1 defines exactly that as
calibration, and 2 lists `BETA_N_BANDS` among the quantities that are "allowed
but charged". So the honest figure is:

    static map (from_sweep)                    2 shots
    beta_N band (resolve_for -> BETA_N_BANDS)  5 shots
    ------------------------------------------------
    calibration total                          7 shots

Not two. The conclusion is untouched -- 7 against 21 for the learned arm and 41
for every federated arm -- but the number was wrong, and it was wrong in the
direction that flattered the point I was making.

### Why this is the same mistake the protocol was written to prevent

PROTOCOL.md 1 says: "Three budgets, tracked separately and never summed into
one number. A result that moves cost between these columns is not a result." I
wrote that, and then quoted a calibration cost that omitted the calibration the
measurement machinery performs on my behalf.

The band is easy to miss precisely because it is not in the experiment script:
it is upstream, cached in the registry, and shared by every arm. Shared cost is
still cost. Every arm pays it, so no COMPARISON between arms changes -- but the
absolute claim "a new machine needs N shots" is a claim about the total, and
the total was understated by a factor of three and a half.

### What the corrected table says

    arm                     target shots   what they buy
    conventional                       7   100% joint success, <=3% failure
    scratch (learned)             7 + 21   competence, from its own shots
    every federated arm           7 + 41   competence, after unlearning

The learned arms pay the 7 as well, because they are scored against the same
tolerance and so need the same band. Adding it to every row leaves the ordering
identical and the ratios slightly less dramatic: 28 and 48 against 7, rather
than 21 and 41 against 2.

### Not fixed in code, and why

The band could be folded into the conventional script's reported calibration
count, but it would then be double-counted whenever the band is already in the
registry from an earlier run -- which is the normal case, and is why it went
uncounted. The honest fix is in the write-up rather than the counter: state
that the joint endpoint's tolerance carries a 5-shot measurement per device,
and report it once for the experiment rather than per arm.

A `tolerance_mode="floor_multiple"` task would not escape this: it needs a
measured open-loop reference instead, which costs the same sweep.



---

## Second fold contradicts the first; only the reach rate survives the guards

*2026-09-16 03:59:25*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=695ded3

The second fold contradicts the first, and the honest reading is that three
seeds cannot tell them apart.

### Both folds, `moderate`, 3 seeds, 120-shot joiner budget, eval grid 10

    arm                      tcv joining    iter joining   iter reach
    scratch                           21              41          67%
    single_source                     41              11          67%
    handover_merge                    41              11         100%
    federated_uniform                 41              21         100%
    federated_similarity              41              41         100%

For tcv_like, inheriting COST 20 shots. For iter_like, inheriting SAVED 30 and
lifted the reach rate from 67% to 100%. Same task, same budget, same code;
opposite conclusions on two devices.

### Why I am not reporting either as the result

The metric's own guards fired on almost every comparison in the second fold:

  * `single_source vs scratch: 3.73x` -- estimated from 2 reaching seeds per
    arm. A bootstrap over two values has essentially no width, so the interval
    [2.82-4.64] is narrow for a reason that has nothing to do with the effect.
  * `handover_merge vs single_source: 1.00x` -- censoring differs, 67% against
    100%, so the ratio compares different populations.
  * the two federated comparisons return 0.5x with intervals spanning
    [0.14-7.36], which is no information at all.

These are the guards the audit asked for, working. The number that survives
them is not a ratio: it is the REACH RATE at a fixed budget. On iter_like, 2 of
3 seeds reached competence from scratch and 3 of 3 reached it from every arm
that inherited more than one source. That is the audit's recommended form of
the answer, and it is the only part of this fold I would defend.

### The one consistent signal across both folds

`federated_similarity` is never the best inheriting arm. It reports 41 on both
folds -- worst or joint-worst every time -- while `handover_merge`, which
merges the same models once and never exchanges during training, reports 11 on
the fold where inheriting helped.

If that holds with more seeds it is a direct negative on SPEC.md 4b: the
physics weighting is not earning its place, and repeated federated exchange is
not beating a single merge at handover.

### What is running

Six seeds on iter_like alone, at eval grid 5 rather than 10, to find out
whether the benefit is real or a three-seed artefact. Six seeds is still
modest, but it doubles the reaching population the ratio is built from and
halves the grid the competence number is quantised to.

### Still true regardless

The conventional controller reaches the joint endpoint on every device from 7
calibration shots with no training at all, at a failure rate bounded under 3%.
Both folds are contests for second place.



---

## The fold-1 replication did not reproduce, and I confounded it myself

*2026-09-16 04:06:02*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=8c032a0

The replication of fold 1 did not reproduce it, and I confounded the
replication myself.

### The two runs, same joiner, same task

    arm                     fold 1 (grid 10,   replication (grid 5,
                             pretrain 100)        pretrain 80)
    scratch                             21                     11
    single_source                       41                     26
    handover_merge                      41                     26
    federated_uniform                   41                     11
    federated_similarity                41                     26

Fold 1 said every inheriting arm cost 20 shots over scratch. The replication
says `federated_uniform` ties scratch, and the gap for the others is 15 rather
than 20. The ordering is not the same.

### I changed two things at once

Evaluation grid 10 -> 5 AND pretrain budget 100 -> 80. Either could move the
numbers: the grid changes what values competence can take at all, and the
pretrain budget changes how good the handed-over controller is. With both moved
there is no way to attribute the difference, and a replication that cannot be
attributed is not a replication.

That is an ordinary experimental-design error and I made it while trying to fix
a different one. The grid change was the point; the pretrain change was
carelessness, and it cost the comparison.

### What the two runs agree on, which is little

  * every arm reaches competence eventually at 100% -- the 120-shot budget is
    enough for all of them on tcv_like;
  * `federated_similarity` is never the fastest inheriting arm;
  * no inheriting arm clearly beats scratch.

They disagree on the size and even the existence of the penalty for
inheriting.

### The reading I will defend

Three seeds cannot resolve these arms. The between-arm differences are smaller
than the run-to-run variation from changing an evaluation cadence, which is a
measurement setting rather than a property of any method. Anything reported at
this seed count is noise dressed as a ranking.

One number does move in a way worth noting: the plateau. `federated_similarity`
has the BEST plateau in the replication (-0.985 against scratch's -3.347) while
having the worst competence time. If that survives more seeds it is a real
effect with a plausible mechanism -- the inherited controller ends up better
but takes longer to get there, because the joiner must first undo it -- and it
is the opposite of what shots-to-competence alone would suggest.

### Running

Six seeds each on iter_like and tcv_like, identical settings (grid 5, pretrain
80, 120-shot budget, reject-if-worse on), so the two folds are finally
comparable to each other and to nothing else.

### Unchanged

Seven calibration shots and no training still reach the joint endpoint on every
device with the failure rate bounded under 3%. None of the above is a contest
the conventional arm is in.



---

## Both plateau claims were exploration noise; evaluation-only rescore overturns them

*2026-09-16 04:15:09*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=b03993b

I reported the plateau twice as if it were a property of the controller. It is
not: `asymptotic_performance` averages the last half of ALL shots, and most of
those shots are candidates the search was probing with. The number therefore
measures how hard an arm was still exploring at least as much as how good the
controller it ended up with was.

Re-scoring the saved artifacts on evaluation shots only -- frozen incumbent,
independently varied scenario -- takes the exploration out. Both of my claims
die.

### Claim 1, which I called a real effect worth replicating

Run `20260916-040505` (moderate, 3 seeds, pretrain 80, grid 5):

    arm                     plateau (all)   plateau (evals)
    federated_similarity           -0.985            -0.002
    federated_uniform              -1.320            -0.023
    handover_merge                 -3.045            -0.020
    scratch                        -3.347            -0.009
    single_source                  -3.033            -0.001

I wrote: "`federated_similarity` has the BEST plateau (-0.985 against
scratch's -3.347) ... If that survives more seeds it is a real effect with a
plausible mechanism -- the inherited controller ends up better but takes longer
to get there, because the joiner must first undo it."

On evaluations the spread of 2.4 units collapses to 0.02, the ordering changes
(`single_source` is nominally best, by 0.001 over `federated_similarity`), and
every arm is inside a hair of every other. There is no effect and there was
never a mechanism to explain -- what differed between arms was the size of the
steps the hill-climber was still taking, not the quality of what it had found.

### Claim 2, the mirror image, also wrong

Run `20260916-035839`, second fold:

    arm                     plateau (all)   plateau (evals)
    federated_similarity         -195.163            -0.011
    federated_uniform             -82.997            -0.024
    handover_merge                -83.388            -0.006
    scratch                       -81.980            -0.007
    single_source                 -82.909            -0.001

Here `federated_similarity` looked more than twice as BAD as everything else. I
had begun treating that as evidence against the physics weighting. On
evaluations it is -0.011 against -0.007, which is nothing. A single wild
candidate late in the run moves the all-shots mean by a hundred units and moves
the evaluation mean not at all -- which is the correct behaviour of an
evaluation and the whole reason AUDIT #5 asked for evaluation shots.

### What this changes

  * `asymptotic_performance_evaluated(runs, last_fraction=0.5)` added to
    `hfmarl/metrics/curves.py`; `exp_coldstart.py` now prints `plateau(ev)`
    beside `plateau(all)` so the gap between them is visible in every future
    run rather than discovered afterwards.
  * The all-shots plateau is not deleted. It is a useful diagnostic of
    exploration -- an arm whose two plateaus diverge by two orders of
    magnitude is one still taking large steps at the end of its budget, which
    is worth knowing. It is simply not a performance number.
  * Two entries above this one in this file overstate their case and should be
    read with this one.

### The part worth keeping

This is the first thing the audit's artifacts (#10) bought. Both claims were
made from a printed table that no longer existed; both were checked months --
in wall-clock, hours -- later from `runs.json` with a twenty-line script and
no re-simulation. Had the artifacts not been written, the correction would have
cost two full six-seed runs, and more likely would not have happened at all.

### Still true, and now the only quantitative claim standing

Seven calibration shots and zero training reach the joint endpoint on all four
devices at `easy` and `moderate`, failure rate bounded under 3% at n=100.
Nothing in the learned arms has yet beaten it, and the plateau -- the one
measure on which the learned arms had appeared to separate -- does not separate
them.



---

## Three memory guards were running at once; the restart reflex duplicated them

*2026-09-16 04:15:23*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=b03993b

Three copies of `guard_memory.py` were running at once, and had been for
hours. Found by listing python processes while checking free memory, not by
anything in the logs -- which is the point.

### How it happened

The guard died once early on without leaving an exit line, so I added a
heartbeat and restarted it. Twice. Each restart began `pkill -f guard_memory`,
which on Windows does not reliably take the venv launcher shim with it: every
guard appears in the process table TWICE (`.venv/Scripts/python.exe` execs a
second interpreter with the same command line), and killing one of the pair
leaves the other polling.

So each "restart" added a guard rather than replacing one. The heartbeat I
added to detect a dead guard was working perfectly and told me nothing about
the two extra live ones.

### Why three guards is worse than none

  * They poll independently on the same floor. One dip below 900 MB is seen by
    all three, each waits its two consecutive readings, and each kills the
    newest experiment -- so a single transient kills THREE runs, which is the
    exact machine-wide loss the guard exists to prevent.
  * They all append to `results/memory_guard.log`. Concurrent appends
    interleave, which is the same corruption I had already guarded against for
    FINDINGS.md by requiring `--no-record` on parallel runs. I applied that
    lesson to the experiment output and not to the guard's own.
  * A kill line in the log does not say which guard wrote it, so the record of
    what happened is ambiguous precisely when it matters.

Nothing was actually killed -- `grep -c killed` over the whole log is 0, and
both six-seed runs are intact -- so this cost nothing this time. It was a live
hazard for several hours.

### The fix, in the guard rather than in my habits

`other_guards()` scans for a python process whose command line mentions
`guard_memory` and which is not in this process's own ancestor/descendant
chain; if one exists the guard refuses to start and exits 3. The chain
exclusion is what makes it work on Windows -- without it the guard sees its own
shim and can never start. `--force` overrides, because an interlock with no
override is a way to be locked out at 3am.

Verified live: with the surviving pair running, a fourth guard launched by hand
printed `refusing to start: guard already running as pid(s) 28832, 31380` and
exited 3.

`tests/test_guard_memory.py` (7 tests) covers the shim case, the genuine
duplicate, experiments not being mistaken for guards, a process vanishing
mid-scan, and -- separately -- that `experiment_processes` still returns
oldest-first so the victim is the youngest.

### The general lesson, which is the same one as AUDIT #1

The bug was in how the tool was OPERATED, and my instinct was to kill the
strays and move on. That fixes the instance. The class of bug -- "restarting a
supervisor silently duplicates it" -- comes back every time I suspect the
guard has died, which is exactly when I am least inclined to check. The fix
belongs in the program.



---

## The guard's first kill: a foreground pytest run cost an experiment an hour

*2026-09-16 04:17:46*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=b03993b

The guard made its first real kill, and it killed the right thing for the
wrong reason. Recording it because the log alone would have made me draw the
opposite conclusion.

### What happened

    2026-09-16T04:15:56  KILLED pid 8324 (656 MB) -- available 545 MB below
                         floor. cmd: ... exp_coldstart.py --joiners tcv_like

That was Q2, the six-seed `tcv_like` run, eleven minutes into its first arm.
The two experiments together had been sitting at 1780 MB available for an
hour. What pushed the machine under the 900 MB floor was the full pytest suite
in the foreground: 455 tests, each process importing JAX and TORAX, roughly a
gigabyte, launched by me while both runs were going.

### Why the log would have misled me

The kill line says only that memory was low. Read cold it says "the
experiments were too big" -- so the fix it suggests is fewer or smaller runs,
which is the wrong fix and an expensive one. The actual cause was invisible to
the guard, because `EXPERIMENT_MARKERS` recognises `scripts/exp_`,
`scripts/gate_` and `scripts/measure_` and nothing else. The guard could see
two experiments and no pytest, so it protected the machine by killing the only
thing it was allowed to kill.

I diagnosed it only because I happened to know what I had just run.

### Two fixes, both in the guard

**1. Name the pressure.** The kill line is now followed by
`memory at kill: pytest pid 2 950 MB, exp pid 1 600 MB, ...` -- the largest
python processes with their roles. A kill that does not say what filled the
memory cannot be acted on.

**2. Make the test suite the preferred victim.** `victim_processes()` is now
experiments PLUS pytest, still oldest-first. "Kill the newest" was always a
proxy for "kill whatever is cheapest to lose"; pytest costs thirty seconds to
re-run and a six-seed experiment costs an hour, and because I launch the suite
by hand while runs are already going, it is almost always the newest anyway.
The proxy is now less wrong in exactly the case that just cost me an hour.

The heartbeat count deliberately still reports experiments only -- a test run
is not an experiment and calling it one would misstate what is running.

### Cost and state

One hour of `tcv_like` wall-clock, relaunched as Q2b at 04:17 on the same
settings, so nothing is lost but time. Q1 (`iter_like`, 6 seeds) was never
touched and is still on its second arm. No results are affected: the killed
run had produced nothing but a first-arm timing line, and `--no-record` means
it wrote nothing to this file.

### The habit this replaces

"Run the full suite whenever code changes" is right, and "run it while two
TORAX processes are live on a 16 GB laptop" is what made it wrong. The habit
fix -- run a targeted subset while experiments are up -- is worth having, but
habits are what failed here twice in one hour (see the three-guards entry
above), so the enforcement belongs in the program: the guard now sacrifices
the suite instead of the science.



---

## Measured: only beta_N is crossable, and only on DIII-D and TCV

*2026-09-16 04:22:52*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=f3551d0

Measured, not inferred: what happens to all three limits when each device is
driven from zero to full thermal command, `easy`, whole episode, worst margin
over the episode. Margin is +1 comfortably safe, 0 exactly at the hard limit,
negative a violation.

    device        cmd    beta_N     q95  greenwald   peak beta_N  violations
    iter_like     0.0      4.35    4.89       0.83         0.827  -
    iter_like     1.0      4.17    4.89       0.83         0.917  -
    sparc_like    0.0      5.19    4.01       2.87         0.407  -
    sparc_like    1.0      5.04    4.00       2.87         0.479  -
    diiid_like    0.0      4.98    7.63       2.82         0.509  -
    diiid_like    0.5      1.72    7.45       2.82         2.140  -
    diiid_like    0.8     -0.01    7.43       2.82         3.005  beta_N
    diiid_like    1.0     -0.38    7.43       2.82         3.190  beta_N
    tcv_like      0.0      4.63    5.01       2.97         0.684  -
    tcv_like      0.2      1.99    4.77       2.97         2.003  -
    tcv_like      0.5     -1.12    4.83       2.97         3.560  beta_N
    tcv_like      1.0     -7.48    4.92       2.97         6.739  beta_N

### Three things follow, and the first one weakens a claim of mine

**1. "Contained" is vacuous on iter_like and sparc_like.** At FULL command
their worst beta_N margin is 4.17 and 5.04 -- four to five soft-to-hard bands
clear of the limit. No action sequence available to the controller can fail
containment on those two devices. I have written that the joint endpoint
"cannot be gamed by doing nothing" because containment and tracking pull
against each other. On half the device set they do not pull against each other
at all: the endpoint there is completed + tracked, and the safety half is
decoration.

That does not overturn the conventional-controller result -- 7 calibration
shots, 100% joint success, failure bounded under 3% -- but it does change what
that result MEANS on two of the four devices. On ITER and SPARC it says the
controller tracked; it does not additionally say the controller was safe,
because nothing there is unsafe.

**2. Two of the three limits are inert everywhere.** q95 never drops below a
margin of 4.0 on any device at any command, and the Greenwald margin never
moves at all with command -- 0.83 on ITER and ~2.9 elsewhere, identical across
the whole sweep. Both have mechanical explanations already in this file: q95
is set by Ip, and Ip lives in the `current` cluster which no preset actuates
(every preset is `clusters=("thermal",)`); density is pinned by `n_e_ped` and
the edge boundary condition, which is the same reason `gas_puff` was found
inert. So the three-limit envelope is a one-limit envelope, and that one limit
is reachable on two devices.

**3. `exp_catastrophe.py` defaults to a target that cannot have a
catastrophe.** `--target` defaults to `iter_like`. The experiment asks whether
a device can avoid a regime it has never personally entered; on ITER there is
no such regime to avoid, so the run would report perfect safety and mean
nothing by it. The run I actually did used `tcv_like`, which crosses beta_N
above about 40% command, so that result stands -- but the default is a trap
and the next person to accept it gets a meaningless positive.

### The bug this nearly became

The first version of this probe passed the command fraction straight into
`env.step`. Actions are in [-1, 1], not [0, 1] -- `gate_authority` gets this
right with `2*level - 1` and I did not copy it. So its "zero command" row was
half power, and it appeared to show `tcv_like` violating the beta limit with
every actuator switched OFF, which would have meant every TCV shot in every
experiment had been failing containment and the headline result was wrong.

I wrote most of a finding announcing that before checking the convention. The
thing that stopped it was reading `gate_authority._sweep_level` to work out why
its recorded band (0.656 at zero command) disagreed with my 3.56 -- two
measurements of the same quantity disagreeing by a factor of five is a fact
about the measurement, not about the plasma.

### What this makes possible, which is the useful half

There is a principled task here that I have not tried. On `diiid_like` and
`tcv_like` the beta limit is crossed in the upper part of the command range --
DIII-D above ~60%, TCV above ~40% -- while the target sits below it. A task
whose setpoint is placed just under that crossing forces the trade-off the
limit machinery was built for: enough gain to track the ramp inside tolerance,
not so much that the overshoot crosses Troyon. Fixed-gain PI has one knob for
both, and the falsification is cheap -- sweep kp and see whether ANY gain
reaches the joint endpoint. If one does, the task is still classical and I say
so. If none does, there is finally something for a learned controller to be
better at, on a device where being unsafe is possible.

That is the first route to a learning-relevant task that is grounded in a
measurement rather than in turning difficulty knobs up until something breaks
-- which is how the last three attempts (`hard` presets, coupled
thermal+particle, anticipation) each turned out to be ill-posed rather than
hard.



---

## `brink` is classical too -- but the classical gain does not transfer

*2026-09-16 04:30:55*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=04b0107

`brink` was the first task I designed from a measurement -- setpoint placed
just under the command level at which each device actually crosses the Troyon
limit -- and a tuned PI still reaches the joint endpoint on both devices. That
is the fourth route to a learning-relevant task to close.

### The sweep, 20 evaluation shots per gain, 5 calibration shots per device

    diiid_like  (gain 1.42 beta_N/unit, tolerance 0.1601)
      kp 0.05  joint 100.0%  median |err| 0.1588
      kp 0.10  joint 100.0%               0.1306
      kp 0.20  joint 100.0%               0.0951
      kp 0.30  joint 100.0%               0.0762
      kp 0.45  joint 100.0%               0.0596
      kp 0.60  joint 100.0%               0.0491
      kp 0.90  joint 100.0%               0.0362   <- best, monotone

    tcv_like    (gain 3.06 beta_N/unit, tolerance 0.1475)
      kp 0.05  joint 100.0%  median |err| 0.0981
      kp 0.10  joint 100.0%               0.0834
      kp 0.20  joint 100.0%               0.0624
      kp 0.30  joint 100.0%               0.0485
      kp 0.45  joint 100.0%               0.0353
      kp 0.60  joint 100.0%               0.0290   <- best, INTERIOR
      kp 0.90  joint  90.0%               0.0811   <- degrades

Both devices: 100% joint success at the best gain, zero violations, full
completion, failure rate bounded at 13.9% (Clopper-Pearson, 20 shots).

### The tension is real; the window is just wide

This is not the same failure as the previous three. `hard` and `brutal` were
ill-posed (tolerances below actuator resolution under qlknn), the coupled
thermal+particle task had an inert actuator, and the anticipation task had no
residual error to anticipate. `brink` does exactly what it was built to do,
and the evidence is the SHAPE of the tcv column: the error falls monotonically
with gain on DIII-D all the way to kp 0.9, but on TCV it turns around at 0.6
and the joint rate drops to 90%. An interior optimum in gain is the signature
of overshoot being punished -- the tension between tracking fast and staying
contained is present and measurable.

It is just not narrow enough to defeat a seven-point gain sweep. Why the two
devices differ is mechanical: the safe window is set in COMMAND space, not in
beta_N. Both devices' targets sit about 0.7 beta_N below the crossing, but
DIII-D moves 1.42 beta_N per unit of command and TCV 3.06, so the same beta_N
margin is 26% of DIII-D's command range and 8% of TCV's. The brink bites in
proportion to actuator gain, which is why TCV is where it shows at all.

### The result I did not expect, and it is the interesting one

**The classical controller's gain does not transfer between devices.** DIII-D's
best gain is 0.9. Applied to TCV, 0.9 scores 90%, not 100%. TCV's own best is
0.6. PROTOCOL.md 2 refuses hyperparameter tuning on the held-out device --
"these are chosen on the source devices" -- and by that rule the conventional
arm arrives at TCV with kp 0.9 and loses ten points of joint success.

So the classical baseline has a cold-start problem of its own, and it is the
same one federation claims to solve: a controller tuned elsewhere does not
simply work here, and finding the right gain costs shots on the new machine
(seven gains x 20 shots = 140, against the 7 calibration shots the arm is
usually charged).

That has been mis-stated in my favour until now. Every conventional result in
this file used `--kp-sweep` and reported the BEST gain, which is tuning on the
evaluation device -- exactly what the protocol refuses for every other arm. The
number is still a fair upper bound on what classical control can do; it is not
a fair statement of what classical control costs.

### What changes

  * A `conventional_transferred` rung: gain chosen on the source devices, no
    target tuning, scored on the joiner. That is the honest classical arm for
    a cold-start comparison, and this sweep says it is beatable on TCV.
  * The existing `conventional` rows stay, relabelled as what they are: the
    tuned-on-target ceiling.
  * `exp_conventional.py` now reports violation and completion rates per gain,
    so the next sweep says WHICH half of the endpoint a gain lost -- this one
    cannot distinguish TCV's 10% failures at kp 0.9 between overshoot and
    tracking, which is the single number I most wanted.

### Cost

280 s, 10 calibration + 280 evaluation shots across two devices. The brink
bands themselves cost 5 shots per device and are recorded in BETA_N_BANDS.



---

## Killed both six-seed runs: the ladder's adjacent rungs differed by three things, not one

*2026-09-16 04:36:17*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=b5f54a1

I killed both six-seed runs about 45 minutes in. The ladder they were
climbing does not have the property the whole design rests on, and the rung
where it fails is the one the audit added the ladder for.

### What the saved weights showed

Reading `received` out of the artifacts, for a `tcv_like` joiner with
diiid/iter/sparc as sources:

    arm                    weights given to each source
    handover_merge         diiid 0.558  sparc 0.232  iter 0.210
    federated_uniform      iter  0.377  sparc 0.377  diiid 0.245
    federated_similarity   diiid 0.409  sparc 0.313  iter 0.278

Two things are wrong in that table.

**`federated_uniform` is not uniform.** Three sources, equal weights, should
be 0.333 each. It is 0.377/0.377/0.245 because `use_similarity=False` turns
off the kernel but the safety factor and staleness still multiply through.

**`handover_merge` is similarity-weighted and pays no safety penalty.** Its
server was constructed with `use_similarity=True`, and `_merge_finals`
published `ClientUpdate`s with no violation rate at all, so
`(1 - violation_rate)` was 1 for every source. That is why diiid gets 0.558
there and 0.245 one rung up.

### Why that breaks the ladder rather than merely being untidy

PROTOCOL.md 3: "each row adds one ingredient. A gap between two adjacent rows
is attributable to the ingredient that differs, and to nothing else." Between
`handover_merge` and `federated_uniform` there were THREE differences, and one
of them ran backwards -- climbing the ladder REMOVED the physics weighting.

So every comparison I have recorded between those two rungs is
uninterpretable. That includes the line I called the one consistent signal
across both folds: "`handover_merge`, which merges the same models once and
never exchanges during training, reports 11 on the fold where inheriting
helped" against federated's 41. That gap is not evidence about repeated
exchange. It is a merge weighted by similarity with no safety penalty being
compared against a merge weighted uniformly with one.

### Why I killed the runs rather than letting them finish

They had about two hours left and would have produced this same invalid
comparison with six seeds instead of three -- a more precise wrong number.
Three of the four adjacent comparisons in those runs were sound, so this was
not a total loss, but the rung the experiment exists to test was the broken
one.

### Fixed

  * `handover_merge` now merges with `use_similarity=False`, matching the rung
    above it, so the gap isolates repeated exchange.
  * `run_condition` reports `final_violation_rates` per source, computed on the
    same last-`local_shots` window the federated arms publish, and
    `_merge_finals` passes them, so the safety rule applies to both rungs
    equally.
  * `tests/test_ladder_integrity.py` pins both, plus the mechanism behind the
    0.377/0.377/0.245 row.

### The deeper problem, which I am NOT fixing yet

The safety factor is `(1 - violation_rate)` over a client's recent shots. Two
of the four devices cannot cross a limit at any command (see the
reachability finding above), so their violation rate is structurally zero
forever. The rule therefore measures PLANT HEADROOM at least as much as
controller behaviour, and it systematically favours the devices with nothing
to violate.

For a `tcv_like` joiner that is exactly backwards. TCV crosses beta above ~40%
command; the only source that shares that predicament is DIII-D; and DIII-D is
the source the safety rule penalises, precisely because it has been near the
limit. The catastrophe premise is that a machine learns its limits from
machines that have already crossed them -- and the aggregation rule
downweights that experience.

Fixing it properly means deciding what the rule is for. Candidates: normalise
the violation rate by the device's reachable risk; penalise violations only
during the exchange window rather than over the whole campaign; or treat
limit-crossing experience as informative and gate on something else (the
regime filter already does the physics-validity job). Not a patch to make
while two experiments are down -- it changes what the method IS, and SPEC.md
4b is the thing under test.

### Relaunching

Same two folds, same settings, on the fixed ladder.



---

## Under the protocol's own tuning rule the conventional arm scores 0% on SPARC

*2026-09-16 04:43:17*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=e0416fe

The conventional controller reaches the joint endpoint on all four devices
only because I tuned it on the device it was scored on. PROTOCOL.md 2 refuses
that for every other arm. Under its own rule, the classical baseline scores
ZERO on sparc_like.

### The sweep, `moderate`, 20 evaluation shots per gain, 4 devices

Joint success by gain (violations were 0.0% and completion 100% everywhere,
so every failure below is a tracking failure):

    kp      diiid    iter    sparc     tcv
    0.05     100%    100%     100%    100%
    0.10     100%    100%      15%    100%
    0.20     100%    100%       0%    100%
    0.30     100%    100%       0%    100%
    0.45     100%    100%       0%    100%
    0.60     100%    100%       0%    100%
    0.90     100%    100%       0%    100%

Three devices are indifferent to the gain across a factor of eighteen.
sparc_like falls off a cliff between 0.05 and 0.10.

### What that does to the transfer

Choose the gain on the OTHER devices -- their joint success first, mean error
as the tie-break -- and score it on the held-out one:

    device        sources pick   joint here    its own best   joint
    diiid_like       kp 0.05          100%        kp 0.60      100%
    iter_like        kp 0.05          100%        kp 0.05      100%
    sparc_like       kp 0.60            0%        kp 0.05      100%
    tcv_like         kp 0.05          100%        kp 0.90      100%

Because every other device scores 100% at every gain, the sources' ranking is
decided ENTIRELY by the error tie-break, and the error tie-break prefers a
high gain -- diiid and tcv both improve monotonically with it. SPARC is the
one device where a high gain fails, and nothing visible from the other three
says so.

Single sources are not better, they are a lottery:

    target        source        kp     joint here
    sparc_like    iter_like    0.05          100%
    sparc_like    diiid_like   0.60            0%
    sparc_like    tcv_like     0.90            0%

One source in three would have worked.

I checked whether this was an artefact of my own tie-break. It is not:
ranking by mean error in units of each device's own tolerance (the fix
committed an hour ago, which matters because tolerances span 0.0091 to 0.2401
here) picks kp 0.60 for SPARC as well. Both rules, same answer, same 0%. An
earlier three-device version of this table said the normalised rule rescued
SPARC -- that was the incomplete sweep, before tcv_like finished, and I nearly
recorded it.

### Why SPARC and not the others

Its tolerance is 0.0091 -- 26 times tighter than DIII-D's -- because tolerance
is a fraction of the measured band and SPARC's band is 0.076 wide against
DIII-D's 2.0. And its actuator gain is 0.0381 beta_N per unit of command, 80
times weaker than TCV's 3.06. So SPARC is the device with the least authority
and the least room for error, and it is the only one where the gain matters at
all. Nothing about the other three predicts it.

### What this changes

  * **The headline is now conditional.** "7 calibration shots, 100% joint
    success on every device" holds with per-device tuning. Under the
    protocol's rule it is 100%, 100%, 0%, 100%. Every earlier statement of
    that result in this file should be read with this one.
  * **There is finally a device where the classical arm loses.** The cold
    start question on sparc_like is a real question: a learned controller
    adapting on the joiner's own shots has 0% to beat, not 100%.
  * **My two six-seed folds were the easy ones.** iter_like and tcv_like are
    both devices where classical transfer already works. A sparc_like fold is
    now running at six seeds.
  * **It is also a warning about the tolerance.** SPARC failing at every gain
    above 0.05 could mean the task is near-ill-posed there rather than that
    control is hard there -- the same pathology `hard` and `brutal` had. The
    distinguishing evidence is that kp 0.05 reaches 100% with a median error
    of 0.0066 against a tolerance of 0.0091, a margin of 1.4x. That is tight
    but it is not below resolution: a correctly tuned classical controller
    passes comfortably enough to be reproducible over 20 shots.

### Provenance

`scripts/exp_conventional.py --task moderate --kp-sweep
0.05,0.1,0.2,0.3,0.45,0.6,0.9 --eval-shots 20`, 560 s, artifacts under
`results/conventional/20260916-044231_e0416fe`. The single-source and pooled
selections were recomputed from the printed table; no additional shots.



---

## Prediction, before the folds land: benefit from inheriting should order sparc > iter > tcv

*2026-09-16 04:59:35*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=6f77c56

Written BEFORE the three six-seed folds finish, so it can be wrong in public.
Every previous reading of these folds was done after seeing the numbers.

### What the saved folds already show, on a metric I only added today

`best_evaluated_error` -- the closest any evaluation got, in units of that
device's own tolerance -- across the three-seed folds:

    tcv_like (distance 1.374 from its sources), TWO independent folds
      arm                     fold A   fold B     shots A   shots B
      scratch                   0.40     0.40          21        11
      single_source             0.49     0.49          41        26
      handover_merge            0.53     0.53          41        26
      federated_uniform         0.53     0.52          41        11
      federated_similarity      0.53     0.51          41        26

    iter_like (distance 1.122), one fold
      arm                     closest   shots   reach
      scratch                    0.77      41     67%
      single_source              0.74      11     67%
      handover_merge             0.67      11    100%
      federated_uniform          0.67      21    100%
      federated_similarity       0.67      41    100%

On tcv_like, inheriting anything makes the controller WORSE and slower, and
the ordering is identical in two independent folds. On iter_like, inheriting
makes it better and lifts the reach rate from 67% to 100%.

I have recorded this disagreement before as "the folds contradict each other
and three seeds cannot resolve them". The closeness metric changes that
reading: within each device the ordering is stable across folds, so the
disagreement is BETWEEN DEVICES rather than noise within one.

### The prediction

The obvious covariate is distance. tcv_like sits 1.374 from its sources and is
the deliberately-furthest device in the set; iter_like sits 1.122; and
sparc_like, the third fold now running, sits 1.070 -- the nearest.

If distance is what separates them, then on the six-seed folds:

  1. sparc_like should show the LARGEST benefit from inheriting -- inheriting
     arms closer than scratch, and a higher reach rate.
  2. iter_like should repeat its smaller benefit.
  3. tcv_like should repeat its PENALTY: scratch closest, inheriting arms
     worse.
  4. Ordered by benefit: sparc > iter > tcv, matching 1.070 < 1.122 < 1.374.

That is SPEC.md 4b's central claim stated so it can fail. If sparc shows no
benefit, or tcv shows one, distance is not the explanation and the physics
weighting has nothing to stand on.

### Why the prediction is worth less than it looks

Three points is three points. Distance is confounded with everything else
about these devices -- tcv_like is also the smallest, the most
ECRH-dominated, and one of the two that can cross a limit. A monotone
ordering over three devices is consistent with a dozen explanations, and
would be evidence only in the weak sense of not having falsified the one
being tested.

The stronger version, which needs no new physics: the SCRAMBLE control that
already exists in `run_condition` -- publish each client under another
device's measured coordinates. If the ordering survives scrambling, it was
never the physics.

### One thing this metric must not be used for, which I nearly did

`best_evaluated_error` is a MINIMUM over evaluations. The conventional
controller's reported error is a MEDIAN over twenty shots. On iter_like I read
the learned arms' 0.0360 against the classical 0.0383 and briefly had the
learned arms ahead -- a best-of-N beating a median, which says nothing except
that the learned arm had more variance. Noted in the metric's docstring and
printed under the column, because the comparison is tempting and wrong. The
comparable statement between a learned arm and the classical one is the joint
success rate, which both report on the same footing.

### Status

R1 (iter_like), R2 (tcv_like), R3 (sparc_like), six seeds each, on the fixed
ladder, all past `scratch` and into `single_source`. The two earlier folds
quoted above were run on the BROKEN ladder, so their handover_merge and
federated_uniform rows are not comparable to each other -- but scratch and the
closeness ordering are unaffected by that bug, and those are what the
prediction rests on.



---

## The bandwidth includes the joiner, and that is a definition question rather than a violation

*2026-09-16 05:10:26*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=0e7e85c

PROTOCOL.md 2 refuses "tuning any hyperparameter -- bandwidth, clipping
factor, rejection threshold, smoothing window -- on the held-out device", and
names the bandwidth first. `run_cold_start` computes it over the whole device
set INCLUDING the joiner. The code comment defends that on comparability
grounds and never mentions the rule it appears to break. I went looking for a
violation and came back with a definition.

### What the choice is worth, measured

Median-heuristic bandwidth over the four devices, against the same heuristic
over each fold's three incumbents:

    joiner        bw (all 4)   bw (sources)   change   nearest source weight
    diiid_like        1.1760         1.3644   +16.0%   tcv   0.6513 -> 0.7272
    iter_like         1.1760         1.2496    +6.3%   sparc 0.8797 -> 0.8927
    sparc_like        1.1760         1.1024    -6.3%   iter  0.8797 -> 0.8643
    tcv_like          1.1760         1.1024    -6.3%   diiid 0.6513 -> 0.6139

So it is not negligible -- the kernel width moves by up to 16% and the weight
given to the nearest source by up to 12%.

### Why the current choice is the better experiment, not the looser one

Including the joiner makes the bandwidth 1.1760 in EVERY fold. Excluding it
gives a different kernel width per fold, so `iter_like` and `tcv_like` would
be compared under kernels 13% apart for a reason that has nothing to do with
the arms. The folds are already hard enough to compare -- one of them
disagreeing with another is the open question of this project -- and adding a
per-fold metric change would make the disagreement uninterpretable.

And `suggest_bandwidth` is a stated rule, not a fit: the median pairwise
distance, chosen once, referring to no result. Nothing about it can be
adjusted after seeing a number.

### The definition the protocol was missing

Tuning means optimising an outcome. Evaluating a fixed rule on information the
newcomer is allowed to have is not tuning, and the joiner's dimensionless
coordinates are already listed in 2 as "allowed but charged as calibration".
PROTOCOL.md now says this explicitly, with the table above, so the choice is
visible rather than buried in a comment.

What remains refused, and is worth restating because it is the thing that
would actually corrupt the result: picking a bandwidth because it made a
number look better.

### Pinned

`test_the_bandwidth_is_the_same_in_every_leave_one_out_fold` asserts the
cold-start call still spans `[joiner] + incumbents`, and that the per-fold
bandwidths still differ enough for the reason to hold. If they ever stop
differing, the justification has evaporated and the test says so rather than
passing quietly.

### Not a reason to kill anything

The three six-seed folds running now all use 1.1760, because that is what the
full set gives in every fold. Nothing about this changes their numbers -- it
changes what I am entitled to say about why.



---

## Six seeds cannot resolve the effects these folds show -- measured, before the folds land

*2026-09-16 05:38:03*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=ff39496

Done before the folds land, from the saved three-seed runs, because it decides
what a null result MEANS. If six seeds cannot resolve the differences these
arms actually show, then "all five arms are indistinguishable" is a statement
about the experiment and not about federation.

They cannot.

### Seed-to-seed spread of shots-to-competence, measured

Per-seed values from the saved folds, `moderate`, budget 120, eval grid 5:

    fold / joiner / arm                       per-seed        sd
    7a44c2f tcv  scratch                      61,21,21      23.1
    7a44c2f tcv  single_source                11,41,41      17.3
    7a44c2f tcv  handover_merge               41,41,41       0.0
    7a44c2f iter scratch                   never,31,51      14.1
    7a44c2f iter handover_merge               11,81,11      40.4
    7a44c2f iter federated_uniform            11,81,21      37.9
    8c032a0 tcv  scratch                      51,11,11      23.1
    8c032a0 tcv  federated_uniform            31,11,11      11.5

Median sd over fifteen (fold, arm) cells: 11.5 shots, range 0.0 to 40.4.

Minimum detectable difference, two-sample, alpha .05 two-sided, power .80,
normal approximation (optimistic -- a t-test at n=6 needs more):

    n= 6   median sd   18.7 shots   (16% of a 120-shot budget)
    n= 6   worst  sd   65.3 shots   (54%)
    n=24   median sd    9.3 shots   (8%)

The differences these folds actually show are 3 to 35 shots. Most of them are
below the floor at n=6, and the ones above it sit in the arms whose own sd is
35-40.

### Pairing by seed does not rescue it, which I expected it to

Every arm in a fold runs with the same seed, so the comparison could be made
seed by seed and the shared noise cancelled. Measured, it does not cancel:

    comparison                              sd(raw)   sd(diff)
    single_source vs scratch (tcv)             18.3       40.4
    handover_merge vs single_source (iter)     35.0       49.5
    single_source vs scratch (tcv, 2nd fold)   15.7       31.8

sd(diff) is LARGER than sd(raw) on most comparisons. If the seed were the
dominant shared noise the differences would be tighter than the values; they
are looser, which means the arms' outcomes are essentially uncorrelated across
seeds. The arms start from different incumbents and the hill-climber diverges
immediately, so a shared seed buys nothing after the first proposal. Common
random numbers is the standard variance-reduction trick here and it does not
apply.

### The continuous metric is no better

`best_evaluated_error` has no evaluation grid quantising it, so it should be
the finer instrument. It is not fine enough:

    fold / joiner      spread between arm means    min detectable at n=6
    7a44c2f tcv          0.0284  (0.13 tol)          0.0323  (0.15 tol)
    7a44c2f iter         0.0095  (0.18 tol)          0.0163  (0.31 tol)
    8c032a0 tcv          0.0215  (0.10 tol)          0.0410  (0.19 tol)

Below the floor in all three.

### And reach rate, which is what PROTOCOL.md 4 asks for, is weakest of all

At six seeds a reach rate of 4/6 against 6/6 -- the iter_like effect that
looked like the clearest signal in the whole project -- is Fisher p = 0.45.
Only an effect at the extreme, something like 0/6 against 6/6, is resolvable
at this seed count.

### What follows, and it is not "run more seeds"

24 seeds per fold would bring the floor to 9 shots and cost roughly eight
hours per fold on this machine. Three folds is a day of compute for one
comparison, and the comparison would still be one task on one simulator.

Two honest responses instead:

  1. **Report the bound, not the point.** "Any difference between adjacent
     rungs is smaller than 19 shots out of 120" is a result. It is a weaker
     claim than a ratio and a truer one, and it is what PROTOCOL.md 4 was
     already reaching for when it asked for reach rates with censored runs
     retained rather than survivor-only ratios.

  2. **Go where the effect is large enough to see.** The conventional
     controller scores 0% on sparc_like when its gain comes from the sources
     and 100% when tuned on the target. If the learned arms reach competence
     on sparc_like at all, that is 0/6 against something -- the one shape of
     result this seed count can resolve. That fold is running.

### The awkward observation I cannot yet explain

On iter_like, three arms report identical best evaluated errors to four
decimal places across all three seeds: 0.0360, 0.0207, 0.0398 for
handover_merge, federated_uniform and federated_similarity alike, while their
received weights differ. Two readings: the arms converge to controllers whose
error is set by the seeded evaluation scenario rather than by the controller,
or the federated arms' source training collapsed onto isolated training
because reject-if-worse refused every exchange. The second would make three
rungs of the ladder one rung.

`source_rejections` and `rejected_rounds` are now reported per arm, precisely
so the next fold answers this -- but the folds those numbers came from predate
the counters, so today it is an open question rather than a finding.



---

## Reject-if-worse compared one draw against a best-of-N, and switched federation off

*2026-09-16 06:01:09*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=5c76a5d

I killed all three six-seed folds about 80% of the way through. The
reject-if-worse rule I added at the user's request -- "if gradients are shit
don't aggregate them... make sure only good stuff gets averaged" -- rejects
good stuff too, and does so harder the longer training runs.

### The bug

`HillClimber.best_return` is a running MAXIMUM over observed returns. The
acceptance test was

    if accept_if_better and total < climbers[name].best_return: reject

so an incoming aggregate had to beat the luckiest draw the incumbent ever had,
on a single draw of its own. Two biases compound: the bar is an order
statistic and therefore inflated, and it keeps rising with the shot count even
when the controller has stopped improving.

Simulated with pure noise, twenty thousand trials, two models of IDENTICAL
true quality -- so every rejection below is the rule and not a judgement:

    shots fired so far    P(adopt)    mean bar
                     1       50.2%       0.005
                     5       16.7%       1.169
                    20        4.9%       1.865
                   100        1.0%       2.507

And with the aggregate genuinely one noise-sd BETTER:

                     1       76.4%
                     5       45.2%
                    20       21.6%
                   100        8.3%

A fresh draw against a fresh draw has no such drift: 50.2%, 50.0%, 49.5%,
50.1% for equal models and ~76% throughout for the better one.

So with `--select` on, an aggregate that genuinely improves the controller was
adopted 8% of the time by late training. Federation was being switched off by
a statistical artefact, and every federated arm was decaying into the isolated
arm while still reporting itself as federated.

### Why this is the explanation I was missing

Last check-in I recorded an open question: on iter_like, `handover_merge`,
`federated_uniform` and `federated_similarity` reported IDENTICAL best
evaluated errors to four decimal places across all three seeds, while their
received weights differed. This is why. If the exchanges are almost never
adopted, the three arms differ only in a handover model and then run identical
local searches from it.

The joiner side is worse than the source side. At the handover the joiner has
fired five commissioning shots, all candidates, so its bar was the best of
five noisy draws -- a 17% adoption chance for an equally good handover. An arm
that refuses its handover IS scratch, with a federated label on the log.

### The fix

The bar is now the most recent UNPERTURBED evaluation of the current
incumbent, which is one draw against one draw:

  * `HillClimber.note_evaluation` records it, and `observe`/`adopt` clear or
    reset it, because an evaluation measures the controller that was
    installed at the time.
  * `acceptance_bar()` returns (value, source), where source is "none" when
    no unbiased estimate exists. The caller must then fire one or adopt --
    reaching for `best_return` is the old bias in another form.
  * The evaluation shots every arm was ALREADY firing, logging and paying for
    now feed the bar. They were being thrown away.
  * At the joiner's handover there is no such evaluation yet, so one shot is
    fired and charged. One shot out of 120 to make a comparison a comparison.
  * `run_catastrophe` had the same line and the same fix.

### Why I killed rather than finished

They had about forty minutes left. All the inheriting arms are subject to the
same bias, so between-arm comparisons were conservative rather than wrong --
but the headline question is whether federation helps a cold-start joiner, and
the mechanism was switched off by an artefact. The folds would have answered
"federation does not help" when the true statement is "federation was
disabled". That is precisely a wrong number being worse than no number.

### What it cost and what is running

About two and a half hours of compute across three folds, discarded. The same
three folds are running again on the fixed rule; `aggregates_unbarred` is now
reported alongside `aggregates_rejected` so an adoption made without an
unbiased bar is visible rather than assumed.

### Standing caveat, unchanged

The power analysis above still holds: six seeds cannot resolve differences
below about 19 shots. Fixing the acceptance rule makes federation ACTUALLY
HAPPEN in the federated arms; it does not make the experiment more sensitive.



---

## Pre-registered: what each outcome of the three folds will be allowed to mean

*2026-09-16 06:37:26*

host=LAPTOP-E2SQE6TQ  
platform=Windows-11-10.0.26200-SP0  
python=3.12.13  
jax=0.11.1  
devices=['cpu:cpu']  
x64=False  
torax=1.4.3  
commit=6df3978

Written while the three folds are on their second arm, so the rule for reading
them exists before the numbers do. Twice today I have fitted a story to a
result and had to retract it -- the plateau claims, both of them -- and the
defence against doing it a third time is to state now what each outcome will
be allowed to mean.

### The primary endpoint is the reach rate, and at six seeds it is nearly blind

PROTOCOL.md 4 asks for reach rates at a fixed budget with censored runs
retained. Two-sided Fisher exact, six seeds per arm, every separable pair:

    baseline   method       p
      0/6       4/6      0.061   marginal
      0/6       5/6      0.015   significant
      0/6       6/6      0.002   significant
      1/6       5/6      0.080   marginal
      1/6       6/6      0.015   significant
      2/6       6/6      0.061   marginal

Read the other way round -- the smallest method rate that separates from each
baseline:

    baseline 0/6  needs method >= 5/6
    baseline 1/6  needs method >= 6/6
    baseline 2/6  NOTHING is separable
    baseline 3/6  NOTHING is separable
    baseline 4/6  NOTHING is separable
    baseline 5/6  NOTHING is separable

So the reach-rate endpoint can only speak when one arm nearly always fails and
another nearly always succeeds. If scratch reaches competence in two or more
seeds out of six, no arm in that fold can be distinguished from it, whatever
it does.

### Which fold that leaves

The earlier three-seed folds put every arm on `tcv_like` and `iter_like` at 67
to 100% reach. If that holds at six seeds, both folds are dead on arrival for
this endpoint -- not because federation failed but because the baseline
succeeds too often to be separable from anything.

`sparc_like` is the exception, and it is why that fold was launched. The
conventional controller there scores 0% when its gain comes from the source
devices and 100% when tuned on the target. If SPARC is genuinely the hard
device, scratch may land near 0/6 -- which is the one baseline value from
which a positive result is visible.

### Stated before the data

  1. **Positive for LEARNING** on a fold: an arm reaches >= 5/6 where the
     transferred conventional controller scores 0%. On `sparc_like` that is
     the pre-registered comparison, and it says a learned controller beat
     classical control on a device where classical control fails -- which is
     the first thing this project has to establish and has not.

  2. **Positive for FEDERATION**: an inheriting arm separates from `scratch`
     by the table above, i.e. scratch <= 1/6 and the inheriting arm >= 5/6.
     Nothing weaker counts, however suggestive the medians look.

  3. **Negative, reportable**: all arms at the same reach rate AND the spread
     of closest evaluated error below the detection floor for that fold. The
     claim then is a BOUND -- "any difference between adjacent rungs is
     smaller than X shots and Y tolerances" -- computed from that fold's own
     seed spread, not from today's estimate.

  4. **Uninformative**: baseline between 2/6 and 5/6 with arms differing by
     less than the floor. This is the most likely outcome on `iter_like` and
     `tcv_like` and it will be reported as "this experiment cannot answer the
     question at this seed count", not as evidence of no effect.

### What will NOT be allowed to count

  * a median shots-to-competence difference below 19 shots, in either
    direction, as evidence of anything;
  * the closest evaluated error compared against the conventional
    controller's median (a minimum against a median, the error I nearly made
    on iter_like);
  * any adjacent-rung ratio whose guards fire, which is most of them at this
    seed count;
  * a monotone ordering across the three devices as confirmation of the
    distance prediction recorded earlier -- three points ordered by one
    covariate is consistent with a dozen explanations, and the scramble
    control is the version of that test that would mean something.

### The diagnostic that decides whether the fold is even valid

Each fold now prints, per arm: seeds that adopted the handover, pretraining
rounds that rejected every client, rounds where the similarity kernel
underflowed to uniform, and source-side refusals. If an inheriting arm adopted
nothing, it ran as `scratch` with a label on it and its row is not evidence
about federation at all. That check comes first, before any number in the
table is read.

