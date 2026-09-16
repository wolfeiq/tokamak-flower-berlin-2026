#!/usr/bin/env python
"""Headroom gate -- is the task hard enough for federation to show anything?

    "If isolated training reaches threshold in a few hundred shots, the gain is
     too small to matter. Design the task hard enough that isolated learning is
     genuinely slow -- otherwise there's no headroom for federation to show
     anything."

THIS IS A BLOCKING GATE. Run it before the Phase 5 matrix, not after. The
matrix is 5 conditions x 4 devices x N seeds; if isolated converges in 200
shots, every one of those runs was wasted before it started.

The gate trains ISOLATED ONLY, at increasing task difficulty, and reports which
preset puts isolated shots-to-threshold inside the measurable band:

    too easy -> no headroom, any speedup is within campaign noise
    too hard -> isolated never converges, so there is no denominator
    too slow -> headroom fine, but the matrix does not finish

    uv run python scripts/gate_headroom.py --device iter_like
    uv run python scripts/gate_headroom.py --tasks moderate,hard --seeds 3

Note this is expensive by construction -- it is measuring how long slow
learning takes. Start with --shot-budget small to sanity-check the plumbing,
then run it properly.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.agents.cem import rollout  # noqa: E402
from hfmarl.agents.policy import make_policy  # noqa: E402
from hfmarl.devices.registry import DEVICES, get as get_device  # noqa: E402
from hfmarl.envs.task import get as get_task  # noqa: E402
from hfmarl.envs.torax_env import ToraxDeviceEnv  # noqa: E402
from hfmarl.experiments.headroom import assess  # noqa: E402
from hfmarl.metrics.curves import (  # noqa: E402
    asymptotic_performance,
    shots_to_competence,
    shots_to_threshold,
    threshold_from_reference,
)
from hfmarl.metrics.log import RunLog, ShotRecord  # noqa: E402
from hfmarl.util.report import record  # noqa: E402


def train_isolated(env, shots: int, seed: int, verbose_every: int = 100) -> RunLog:
    """Train one isolated agent, logging every shot.

    Uses a simple hill-climbing search rather than CEM: CEM evaluates a whole
    population per iteration, which makes "shots consumed" jump in blocks and
    blurs exactly the quantity this gate measures. Hill climbing consumes one
    shot per step, so the shot axis is exact.
    """
    log = RunLog(condition="isolated", device=env.device.name, seed=seed)
    rng = np.random.default_rng(seed)
    policy = make_policy(env._observe().shape[0], env.n_actions, seed=seed)
    best = policy.get_flat().copy()
    best_return = -np.inf
    sigma = 0.3
    max_steps = env.task.steps_per_shot + 2

    for shot in range(shots):
        cand = best + rng.normal(0, sigma, best.size) if shot else best
        policy.set_flat(cand)
        t0 = time.perf_counter()
        total, steps, viol = rollout(env, policy, max_steps)
        wall = time.perf_counter() - t0

        errs = [
            abs(s.scalars.get("beta_N", np.nan) - s.target)
            for s in env.trajectory
            if s.ok and "beta_N" in s.scalars
        ]
        log.add(ShotRecord(
            shot=shot, reward=float(total), steps=steps, violations=viol,
            terminated_early=steps < max_steps - 2,
            beta_error=float(np.nanmean(errs)) if errs else float("nan"),
            wall_seconds=wall,
        ))

        if total > best_return:
            best_return, best = total, cand.copy()
            sigma = min(sigma * 1.05, 0.5)
        else:
            sigma = max(sigma * 0.99, 0.02)

        if verbose_every and (shot + 1) % verbose_every == 0:
            recent = log.rewards()[-verbose_every:]
            print(f"    shot {shot + 1:5d}  recent mean {recent.mean():9.2f}  "
                  f"best {best_return:9.2f}  sigma {sigma:.3f}")
    return log


def last_improvement(rewards) -> int:
    """1-based shot at which the best-so-far last improved.

    If this is far beyond the reported shots-to-threshold, the crossing
    happened while learning was still running and the threshold was too low
    to mean 'converged'.
    """
    rw = np.asarray(rewards, dtype=float)
    if rw.size == 0:
        return 0
    running = np.maximum.accumulate(rw)
    improved = np.flatnonzero(np.diff(running, prepend=-np.inf) > 0)
    return int(improved[-1]) + 1 if improved.size else 0


def _emit(args, device, status: str, lines: list[str]) -> None:
    """Append this run to FINDINGS.md unless --no-record."""
    if args.no_record:
        return
    path = record(f"Headroom gate -- {device.name} ({status})",
                  "```\n" + "\n".join(lines) + "\n```")
    print(f"\n(recorded to {path})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--tasks", default="easy,moderate,hard",
                    help="comma-separated presets, easiest first")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shot-budget", type=int, default=2000)
    ap.add_argument("--threshold-fraction", type=float, default=0.8,
                    help="fraction of the best plateau observed, used as threshold")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--n-conditions", type=int, default=5)
    ap.add_argument("--n-devices", type=int, default=4)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    device = get_device(args.device)
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in task_names:
        get_task(t)  # validate early

    say(f"device      : {device.name}")
    say(f"tasks       : {task_names}")
    say(f"seeds       : {args.seeds}   shot budget: {args.shot_budget}")
    say(f"matrix size : {args.n_conditions} conditions x {args.n_devices} devices "
        f"x {args.seeds} seeds")
    say()

    verdicts = {}
    for task_name in task_names:
        task = get_task(task_name)
        say(f"=== {task_name}  (difficulty {task.expected_difficulty}/5, "
            f"{task.steps_per_shot} steps/shot, {task.transport_model}, "
            f"setpoint {task.setpoint.kind}) ===")
        runs = []
        for seed in range(args.seeds):
            say(f"  seed {seed}:")
            env = ToraxDeviceEnv(device, task=task, seed=seed)
            if env.task_warnings:
                for w in env.task_warnings:
                    say(f"    task warning: {w}")
            t0 = time.perf_counter()
            runs.append(train_isolated(env, args.shot_budget, seed))
            say(f"    {time.perf_counter() - t0:.1f}s")

        sec_per_shot = float(np.mean([
            np.mean([s.wall_seconds for s in r.shots]) for r in runs
        ]))
        # The rate goes out BEFORE anything that can fail. It is the whole
        # reason RUNBOOK 6 says to run a small pilot first, and a pilot that
        # dies before printing it has told you nothing about what the real run
        # costs.
        matrix_h = (args.n_conditions * args.n_devices * args.seeds
                    * args.shot_budget * sec_per_shot / 3600.0)
        say()
        say(f"  {sec_per_shot:.2f} s/shot  ->  {matrix_h:.1f} h serial for "
            f"{args.n_conditions}x{args.n_devices}x{args.seeds} runs "
            f"x {args.shot_budget} shots")

        # THE VERDICT RUNS ON THE ABSOLUTE CRITERION. See
        # `shots_to_competence` for why the plateau-relative one cannot rank
        # tasks against each other.
        tol = float(env.task.tolerance)
        comp = shots_to_competence(runs, tol, window=args.window)
        say()
        say(f"  competence criterion: |beta_N - target| <= {tol:.3f} "
            f"(the task tolerance), smoothed over {args.window}")

        if comp.n_reached == 0:
            # No crossing anywhere. Budget and difficulty are
            # indistinguishable from here, so neither is claimed.
            verdicts[task_name] = None
            say(f"  NO CROSSING: {comp.summary()}")
            say("  A censored run cannot separate 'too hard' from 'too")
            say("  short'. Scale the budget from the rate above and re-run.")
            say()
            continue

        verdict = assess(
            comp, seconds_per_shot=sec_per_shot, shot_budget=args.shot_budget,
            n_conditions=args.n_conditions, n_devices=args.n_devices,
            n_seeds=args.seeds,
        )
        verdicts[task_name] = verdict
        for ln in verdict.summary().splitlines():
            say(f"  {ln}")
        say()

        # Everything below is diagnostic: it is what makes a misleading
        # number visible instead of plausible.
        plateau, _ = asymptotic_performance(runs)
        starts = [float(np.median(r.rewards()[: args.window])) for r in runs]
        start = float(np.median(starts))
        say(f"  curve: start {start:.3f} -> plateau {plateau:.3f} "
            f"(range {plateau - start:+.3f})")
        say(f"  best-so-far last improved at shot "
            f"{max(last_improvement(r.rewards()) for r in runs)}")
        try:
            threshold = threshold_from_reference(runs, args.threshold_fraction)
        except ValueError as exc:
            say(f"  plateau-relative threshold: undefined ({exc})")
        else:
            rel = shots_to_threshold(runs, threshold, window=args.window)
            say(f"  plateau-relative threshold {threshold:.3f}: {rel.summary()}")
            say("    ^ informational only -- each task anchors this to its "
                "own curve, so it cannot rank tasks against each other.")
        say()

    # --- overall -------------------------------------------------------
    say("=" * 66)
    passing = [t for t, v in verdicts.items() if v is not None and v.passes]
    say(f"{'task':12s}{'isolated shots':>18s}{'reach':>8s}{'matrix h':>10s}  verdict")
    for t, v in verdicts.items():
        if v is None:
            say(f"{t:12s}{'no crossing':>18s}{'-':>8s}{'-':>10s}  "
                "INCONCLUSIVE (budget)")
            continue
        med = f"{v.isolated.median:.0f}" if v.isolated.n_reached else "never"
        say(f"{t:12s}{med:>18s}{v.isolated.reach_rate:>7.0%}"
            f"{v.matrix_hours:>10.1f}  "
            f"{'PASS' if v.passes else 'FAIL (' + v.binding_constraint + ')'}")
    say()

    if passing:
        chosen = passing[0]
        say(f"PASS: use task '{chosen}' for the Phase 5 matrix.")
        say(f"  Isolated needs ~{verdicts[chosen].isolated.median:.0f} shots, which")
        say("  leaves room for a federation speedup that is worth reporting.")
        say(f"  Estimated serial matrix cost: {verdicts[chosen].matrix_hours:.1f} h "
            "(parallelise across processes).")
        status, rc = "PASSED", 0
    else:
        decided = [(t, v) for t, v in verdicts.items() if v is not None]
        if not decided:
            # Not a FAIL. A FAIL says the task is wrong; this says the run
            # was too short to say anything, and the two send you to
            # opposite places.
            say("INCONCLUSIVE: no run brought the tracking error inside")
            say("tolerance, so nothing here is a verdict on the TASK -- a")
            say("censored run cannot separate too-hard from too-short.")
            say("Re-run at a budget scaled from the measured s/shot above.")
            say("DO NOT run the Phase 5 matrix yet.")
            status, rc = "INCONCLUSIVE", 1
            _emit(args, device, status, lines)
            return rc

        say("FAIL: no task in this sweep is usable. DO NOT run the Phase 5 matrix.")
        say()
        hardest_name, hardest = decided[-1]
        for ln in hardest.advice():
            say(f"  {ln}")
        if hardest.binding_constraint == "headroom":
            nxt = get_task(hardest_name).harder().name
            if nxt != hardest_name:
                say()
                say(f"  Next preset up is '{nxt}' -- rerun with --tasks {nxt}")
            else:
                say()
                say("  Already at the hardest preset. Define a harder TaskSpec in")
                say("  hfmarl/envs/task.py, or accept that this task cannot")
                say("  demonstrate the claim and say so.")
        status, rc = "FAILED", 1

    _emit(args, device, status, lines)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
