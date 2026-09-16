# Inverse PINN vs classical power balance — a negative result

**Conclusion: the classical estimator won. The PINN is not on the critical
path.** Kept in-tree because a negative result that took work to establish is
worth keeping, and because the comparison is reproducible.

## What was compared

Both estimate χ(ρ) from profiles and a known heating source, on a manufactured
solution where χ(ρ) is known exactly (a mid-radius bump — the "anomalous
transport" case the method exists to detect).

| estimator | method |
|---|---|
| classical | integrate the balance law, divide by the measured gradient. `identification/closure.py` |
| inverse PINN | smooth T(ρ) and χ(ρ) networks under the same integrated residual. `identification/pinn.py` |

## Result

| noise | points | classical | inverse PINN | winner |
|---|---|---|---|---|
| 0% | 60 | **0.013** | 0.95 | classical |
| 1% | 60 | **0.83** | 0.96 | classical |
| 3% | 40 | 4.49 | **0.95** | PINN (only because classical degrades) |
| 5% | 25 | 0.98 | **0.92** | marginal |

The PINN sits near ~0.95 relative error at *every* noise level, which is the
signature of a fit that is not converging rather than one that is robust. Its
apparent wins at high noise are the classical estimator degrading, not the
network improving.

## Why it underperformed

Almost certainly **step count, not formulation**. The NumPy gradient path uses
central differences over the parameter vector — O(n_params) loss evaluations
per step, ~100× slower than autodiff. At the iteration counts that fit in a
reasonable runtime the fit is undertrained: loss decreases (49 → 2.2) and χ
varies in roughly the right band, but the baseline level is wrong (recovered
1.7–3.0 against a true 0.5–2.5).

`identify_chi_pinn(..., use_jax=True)` uses `jax.grad` when JAX is present,
which should close most of that gap. **This was never run with the JAX path**
— no JAX on the machine the comparison was done on.

## What was done right, and worth keeping

The residual uses the **integral** form of the conservation law, not the
differential form. Writing it the textbook way needs second derivatives of the
network, and that is the configuration that failed in the Cosserat stiffness
work — when an unknown enters only through a high-order derivative the network
under-resolves and the residual becomes satisfiable without committing to the
parameter. Integrating once puts χ algebraically against a data-derived
quantity and needs only first derivatives.

χ is parameterised as `softplus`, so it is positive by construction. A negative
transport coefficient is unphysical and lets the residual be satisfied in a
nonsense way.

## If anyone revisits this

The honest test is a noise/sparsity sweep with the JAX gradient path and 20k+
iterations. The PINN's case is sparse, noisy diagnostics — if it does not win
there, it does not win anywhere, and the classical estimator is simpler,
instant, and has no training loop.

Until then: **use the classical estimator.**
