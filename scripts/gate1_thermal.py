#!/usr/bin/env python
"""Phase 1 gate -- one thermal agent, one device, tracking under limits.

SPEC.md Phase 1: "One thermal agent controlling heating power, tracking
normalized pressure. Implement limits: Greenwald density fraction, beta limit,
q constraint. Gate: tracks setpoint, respects limits."

PASS requires all three:
  1. the trained policy beats a fixed-power baseline on mean episode return;
  2. tracking error on beta_N is below `--tol`;
  3. the trained policy completes its evaluation episodes with no limit
     violation.

Criterion 1 matters most. A policy that merely sits still can score well if
the reward is badly shaped, so it is compared against the best of several
constant-power baselines rather than against nothing.

Run gate0_env.py and gate0_jit.py FIRST. This script takes much longer and
its result is meaningless if the environment is not sound.

    uv run python scripts/gate1_thermal.py --device iter_like
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.agents.cem import rollout, train_cem  # noqa: E402
from hfmarl.agents.policy import make_policy  # noqa: E402
from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.task import get as get_task  # noqa: E402
from hfmarl.envs.torax_env import ToraxDeviceEnv  # noqa: E402
from hfmarl.util.report import record  # noqa: E402


class ConstantPolicy:
    """Fixed command every step -- the baseline a real controller must beat."""

    def __init__(self, level: float, act_dim: int):
        self.level, self.act_dim = level, act_dim

    def act(self, obs):
        return np.full(self.act_dim, self.level, dtype=float)


def evaluate(env, policy, episodes: int, max_steps: int):
    """Mean return, mean |beta_N - target|, and whether any limit was violated."""
    returns, errors, violated = [], [], []
    for _ in range(episodes):
        total, _, viol = rollout(env, policy, max_steps)
        returns.append(total)
        violated.append(bool(viol))
        # Error against the target IN FORCE at each step -- the setpoint
        # moves for every task above `easy`, so a single final target would
        # mis-score the whole episode.
        errs = [
            abs(s.scalars["beta_N"] - s.target)
            for s in env.trajectory
            if s.ok and "beta_N" in s.scalars and np.isfinite(s.target)
        ]
        if errs:
            errors.append(float(np.mean(errs)))
    return (
        float(np.mean(returns)) if returns else float("nan"),
        float(np.mean(errors)) if errors else float("nan"),
        any(violated),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--task", default="easy",
                    help="difficulty preset; see hfmarl/envs/task.py and "
                         "scripts/gate_headroom.py")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--tol", type=float, default=None,
                    help="max acceptable mean |beta_N - target|; "
                         "defaults to the task's own tolerance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    device = get(args.device)
    task = get_task(args.task)
    tol = args.tol if args.tol is not None else task.tolerance * 2.0
    max_steps = task.steps_per_shot + 2

    env = ToraxDeviceEnv(device, task=task, clusters=("thermal",), seed=args.seed)
    say(f"device       : {device.name}")
    say(f"task         : {task.name} (difficulty {task.expected_difficulty}/5)")
    say(f"actuators    : {env.bank.names}")
    say(f"setpoint     : {task.setpoint.kind}, base {task.setpoint.base}, "
        f"amplitude {task.setpoint.amplitude}")
    say(f"episode      : {task.episode_length}s in {task.delta_t_a}s steps "
        f"({task.steps_per_shot} actions)")
    say(f"transport    : {task.transport_model}")
    say(f"seed         : {args.seed}")
    for w in env.task_warnings:
        say(f"task warning : {w}")
    say()

    # --- baselines -----------------------------------------------------
    say("constant-power baselines:")
    baselines = {}
    for level in (-1.0, -0.5, 0.0, 0.5, 1.0):
        r, e, v = evaluate(env, ConstantPolicy(level, env.n_actions), 1, max_steps)
        baselines[level] = r
        say(f"  level {level:+.2f}  return {r:10.2f}  |beta err| {e:7.3f}"
            f"  {'VIOLATED' if v else 'ok'}")
    best_level = max(baselines, key=lambda k: baselines[k])
    best_baseline = baselines[best_level]
    say(f"  best baseline: level {best_level:+.2f} at {best_baseline:.2f}")
    say()

    # --- train ---------------------------------------------------------
    policy = make_policy(obs_dim=env._observe().shape[0], act_dim=env.n_actions,
                         seed=args.seed)
    say(f"policy: {policy.n_params} params, {policy.payload_bytes()} bytes "
        f"({policy.payload_bytes() / 1024:.2f} KiB) per federated update")
    say()
    say("training (CEM):")
    result = train_cem(
        env, policy,
        iterations=args.iterations, population=args.population,
        seed=args.seed, max_steps=max_steps, verbose=True,
    )
    for h in result.history:
        lines.append(
            f"  iter {h['iteration']:3d}  mean {h['mean_return']:10.2f}  "
            f"best {h['max_return']:10.2f}  steps {h['mean_steps']:6.1f}  "
            f"violations {h['violation_rate']:5.1%}"
        )
    say()

    # --- evaluate ------------------------------------------------------
    ret, err, violated = evaluate(env, policy, args.eval_episodes, max_steps)
    say("trained policy:")
    say(f"  mean return        : {ret:10.2f}   (best baseline {best_baseline:.2f})")
    say(f"  mean |beta_N - tgt|: {err:10.3f}   (tolerance {tol})")
    say(f"  limit violations   : {'YES' if violated else 'none'}")
    say()

    beats = ret > best_baseline
    tracks = np.isfinite(err) and err < tol
    safe = not violated
    say(f"  beats baseline  : {'PASS' if beats else 'FAIL'}")
    say(f"  tracks setpoint : {'PASS' if tracks else 'FAIL'}")
    say(f"  respects limits : {'PASS' if safe else 'FAIL'}")
    say()

    if beats and tracks and safe:
        say("PASS: Phase 1 gate met. Proceed to Phase 2 (thermal cluster).")
        status, rc = "PASSED", 0
    else:
        say("FAIL: Phase 1 gate not met. Report this rather than tuning until")
        say("it passes -- SPEC.md Phase 2 explicitly asks for negative results")
        say("to be reported. Things worth checking first:")
        if not tracks:
            say(f"  * is the {task.setpoint.kind} setpoint (base "
                f"{task.setpoint.base}) reachable on this device with these")
            say("    actuator limits? Try --task trivial and a constant-power sweep.")
        if not beats:
            say("  * is the reward dominated by the limit penalty, making every")
            say("    policy look alike? Try an easier --task.")
        if not safe:
            say("  * is the episode long enough for the limit to be avoidable,")
            say("    or does the plasma cross it regardless of the action?")
        status, rc = "FAILED", 1

    if not args.no_record:
        p = record(f"Phase 1 -- thermal tracking on {device.name} ({status})",
                   "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
