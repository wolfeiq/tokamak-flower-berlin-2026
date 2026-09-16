"""Cross-entropy method policy search.

WHY CEM AND NOT PPO/SAC
-----------------------
SPEC.md Phase 1 asks for an agent that tracks a setpoint under limits. CEM is
chosen as the FIRST trainer because it is gradient-free and dependency-free,
so Phase 1 can be gated without adding PyTorch or stable-baselines3 next to
JAX on a memory-constrained machine, and because it is robust to the episode
terminations that limit violations produce.

It is not the end state. SPEC.md §2 specifies PPO or SAC, and sample-efficiency
claims in Phase 5 need a proper RL algorithm. CEM's role is to prove the
environment is learnable at all -- if CEM cannot track a setpoint here, the
problem is the environment or the reward, not the algorithm, and swapping in
PPO would only hide that.

CEM is also genuinely expensive in samples (population x episodes per
iteration), so do not use it for anything that reports sample efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CEMResult:
    best_params: np.ndarray
    best_return: float
    history: list[dict] = field(default_factory=list)


def rollout(env, policy, max_steps: int = 1000) -> tuple[float, int, tuple[str, ...]]:
    """One episode. Returns (total reward, steps taken, violations seen)."""
    obs, _ = env.reset()
    total, steps = 0.0, 0
    violations: tuple[str, ...] = ()
    for _ in range(max_steps):
        obs, r, terminated, truncated, info = env.step(policy.act(obs))
        total += r
        steps += 1
        violations = violations + tuple(info.get("violations", ()))
        if terminated or truncated:
            break
    return total, steps, tuple(dict.fromkeys(violations))


def train_cem(
    env,
    policy,
    *,
    iterations: int = 10,
    population: int = 16,
    elite_frac: float = 0.25,
    init_std: float = 0.5,
    std_decay: float = 0.92,
    seed: int = 0,
    max_steps: int = 1000,
    verbose: bool = True,
) -> CEMResult:
    """Fit ``policy`` in place. Returns the best parameters found.

    Every run is seeded and the seed is part of the result, because SPEC.md §8
    is explicit that sample-efficiency claims are trend claims that will not
    survive single runs.
    """
    rng = np.random.default_rng(seed)
    mean = policy.get_flat().copy()
    std = np.full_like(mean, init_std)
    n_elite = max(2, int(round(population * elite_frac)))

    best_params, best_return = mean.copy(), -np.inf
    # `policy` is left holding the last sampled parameters unless a finite
    # return was seen; restoring `best_params` at the end covers that.
    history: list[dict] = []

    for it in range(iterations):
        samples = rng.normal(mean, std, size=(population, mean.size))
        returns = np.empty(population)
        step_counts = np.empty(population, dtype=int)
        all_viol: list[tuple[str, ...]] = []

        for i, s in enumerate(samples):
            policy.set_flat(s)
            returns[i], step_counts[i], viol = rollout(env, policy, max_steps)
            all_viol.append(viol)

        # A NaN return must rank WORST, not best. np.argsort places NaN last,
        # so without this the samples that produced unusable simulations became
        # the elite set and poisoned the search mean -- while `returns.max()`
        # stayed NaN, so `best_return` never updated and the whole run silently
        # did nothing. Mapping to -inf makes them strictly worst.
        n_bad = int((~np.isfinite(returns)).sum())
        returns = np.where(np.isfinite(returns), returns, -np.inf)

        # AUDIT. Snapshot the distribution BEFORE elite selection touches
        # it. The all-failed branch below means to hold position, but it ran
        # after mean and std had already been recomputed from an elite set
        # drawn entirely from failures -- so a seeded reproduction moved a
        # scalar mean from 0 to 0.186 while printing 'holding mean'.
        prev_mean, prev_std = mean.copy(), std.copy()

        elite_idx = np.argsort(returns)[-n_elite:]
        elite = samples[elite_idx]
        # Keep a floor on std so the search cannot collapse to a point in the
        # first few iterations, which it will happily do when most of the
        # population terminates early on a limit violation.
        std = np.maximum(elite.std(axis=0), 1e-3) * std_decay

        if n_bad == population:
            # Every member failed. Advancing the mean onto -inf samples would
            # walk the search somewhere arbitrary, so hold position instead.
            if verbose:
                print(f"  iter {it:3d}  ALL {population} rollouts non-finite -- "
                      "holding mean; check the simulation, not the optimiser")
            # Hold BOTH, and hold the distribution that existed before this
            # generation rather than the one its failures produced.
            mean = (best_params.copy() if np.isfinite(best_return)
                    else prev_mean)
            std = prev_std

        if np.isfinite(returns.max()) and returns.max() > best_return:
            best_return = float(returns.max())
            best_params = samples[int(np.argmax(returns))].copy()

        viol_rate = sum(1 for v in all_viol if v) / population
        finite = returns[np.isfinite(returns)]
        rec = {
            "iteration": it,
            # Averaged over FINITE members only; including -inf would make the
            # mean -inf and hide the rest of the population's behaviour.
            "mean_return": float(finite.mean()) if finite.size else float("nan"),
            "max_return": float(finite.max()) if finite.size else float("nan"),
            "mean_steps": float(step_counts.mean()),
            "violation_rate": viol_rate,
            "n_nonfinite": n_bad,
        }
        history.append(rec)
        if verbose:
            bad = f"  non-finite {n_bad}/{population}" if n_bad else ""
            print(
                f"  iter {it:3d}  mean {rec['mean_return']:10.2f}  "
                f"best {rec['max_return']:10.2f}  "
                f"steps {rec['mean_steps']:6.1f}  "
                f"violations {viol_rate:5.1%}{bad}"
            )

    policy.set_flat(best_params)
    return CEMResult(best_params=best_params, best_return=best_return, history=history)
