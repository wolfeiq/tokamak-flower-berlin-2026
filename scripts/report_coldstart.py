#!/usr/bin/env python
"""Read saved cold-start folds and print the comparison, without re-running.

    uv run python scripts/report_coldstart.py --since 20260916-05

WHY THIS EXISTS AS A SCRIPT
---------------------------
Every number in a cold-start fold is recomputable from `runs.json`, and twice
today a claim of mine died to a rescore of exactly that kind: the plateau I
reported as evidence turned out to be measured on candidate shots, and a
"replication" turned out to have changed two settings at once. Recomputing
from artifacts costs seconds; re-running a fold costs two hours. So the
analysis lives here rather than inside the experiment, and the experiment's
printed table is a convenience rather than the record.

WHAT IT REPORTS, AND WHY EACH ONE
---------------------------------
  * REACH RATE at the fixed budget, censored runs retained. PROTOCOL.md 4:
    a ratio between the seeds that happened to succeed compares different
    populations, and with few seeds it reports a precise-looking interval
    that has nothing to do with the effect.
  * shots to competence, median over the seeds that reached it, with the
    censored count alongside so the median is never read alone.
  * CLOSEST evaluated error, in units of that device's tolerance. On a device
    where no arm passes, this is the only thing that still ranks them -- but
    it is a MINIMUM over evaluations, so it ranks arms against each other and
    never against the conventional row, whose error is a median over twenty
    shots. Setting a best-of-N against a median flatters the best-of-N.
  * adjacent-rung comparisons only, with their guards printed. The ladder
    exists so a gap is attributable to one ingredient; comparing everything
    to scratch throws that away.
  * the conventional reference for the joiner, BOTH ways round -- tuned on
    the joiner, and with the gain chosen on the source devices. The second
    is the one a cold-start comparison is against, because a machine with no
    shots cannot tune.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.metrics.curves import (  # noqa: E402
    best_evaluated_error,
    binomial_exact_interval,
    paired_joint_reach,
    shots_to_competence_evaluated,
    shots_to_joint_competence_evaluated,
    speedup_from_results,
)
from hfmarl.metrics.log import RunLog  # noqa: E402

ARMS = ["scratch", "single_source", "handover_merge",
        "federated_uniform", "federated_similarity"]

def conventional_reference(task: str, root: str = "results/conventional"):
    """The classical baseline for this task, read from its own artifacts.

    Two numbers per device, never one: the ceiling reached by tuning the gain
    on the device itself, and what the same controller scores when the gain
    is chosen on the other devices, as PROTOCOL.md 2 requires of every arm.

    READ, NOT TRANSCRIBED. The first version of this file carried the four
    `moderate` rows as a literal dict copied out of a run's output. That is a
    number with no provenance and a date it does not carry: re-measure the
    sweep with a different gain grid or a different evaluation count and the
    constant keeps reporting the old answer, in a file whose whole purpose is
    to recompute things from artifacts rather than trust a remembered value.

    Returns {} when no sweep for this task has been run, and the caller says
    so rather than printing a reference it does not have.
    """
    best: dict[str, dict] = {}
    newest = None
    for d in sorted(Path(root).glob("*")):
        man_p, sum_p = d / "manifest.json", d / "summary.json"
        if not (man_p.exists() and sum_p.exists()):
            continue
        man = json.loads(man_p.read_text(encoding="utf-8"))
        if man.get("task") != task:
            continue
        summ = json.loads(sum_p.read_text(encoding="utf-8"))
        rows = {r["device"]: r for r in summ.get("rows", [])}
        transfer = {r["device"]: r for r in summ.get("transfer", [])}
        if not transfer:
            continue  # a sweep without the transfer column cannot answer this
        newest = d.name
        best = {
            dev: {
                "tuned": rows.get(dev, {}).get("joint_success", float("nan")),
                "transferred": t["joint_transferred"],
                "own_kp": t["own_kp"],
                "transferred_kp": t["transferred_kp"],
            }
            for dev, t in transfer.items()
        }
    if best:
        best["_source"] = newest
    return best


def load(d: Path):
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((d / "runs.json").read_text(encoding="utf-8"))
    runs: dict[str, dict[str, list[RunLog]]] = {}
    for key, rows in payload.items():
        # keys are "<joiner>:<arm>" in the current writer; older runs used
        # the condition name, which is why the arm is taken from the key.
        joiner, _, arm = key.rpartition(":")
        runs.setdefault(joiner or man["joiners"][0], {})[arm] = [
            RunLog.from_dict(r) for r in rows]
    return man, runs


def report(d: Path, conv_root: str = "results/conventional") -> None:
    man, runs = load(d)
    task = man.get("task")
    conv_table = conventional_reference(task, conv_root) if task else {}
    budget = man.get("join")
    tols = man.get("tolerances", {})
    print(f"\n{'=' * 78}")
    print(f"{d.name}   task={task}  seeds={man.get('seeds')}  "
          f"budget={budget}  eval_every={man.get('eval_every')}  "
          f"commit={man.get('commit')}")
    if man.get("status") == "running":
        print("  RUNNING / INCOMPLETE: completed seeds only; no final conclusion.")
    if "joint_competence" not in man.get("metric", ""):
        print("  Legacy run: joint endpoint rescored, but historical handover "
              "eligibility and training bugs may remain in these observations.")
    print("  A nonsignificant comparison is inconclusive, not equivalence; "
          "the minimum detectable effect is not an upper bound on the effect.")
    print("  Paired p-values below are unadjusted across ladder comparisons.")

    for joiner, arms in runs.items():
        tol = float(tols.get(joiner, float("nan")))
        print(f"\n  joiner {joiner}   tolerance {tol:.4f}")
        conv = conv_table.get(joiner)
        if conv:
            print(f"    conventional reference ({conv_table.get('_source')}): "
                  f"tuned here {conv['tuned']:.0%} (kp {conv['own_kp']}), "
                  f"gain from the sources {conv['transferred']:.0%} "
                  f"(kp {conv['transferred_kp']})")
            print("      (conventional values are per-shot success; the reach "
                  "column below is per-seed competence, a different denominator)")
        else:
            print(f"    conventional reference: none measured for task "
                  f"{task!r} with a transfer column -- run "
                  f"scripts/exp_conventional.py --task {task} --kp-sweep ...")
        print(f"    {'arm':24s}{'joint reach':>12s}{'shots (median)':>16s}"
              f"{'censored':>10s}{'track reach':>12s}{'closest |err|':>15s}{'x tol':>8s}")
        print("      (joint = completed + contained + tracked on two consecutive "
              "evaluations; track reach is the legacy smoothed diagnostic)")
        print("      (closest is a MINIMUM over evaluations -- compare arms to "
              "each other, not to the conventional row above)")
        results = {}
        for arm in ARMS:
            logs = arms.get(arm)
            if not logs:
                continue
            r = shots_to_joint_competence_evaluated(
                logs, tol, expected_steps=man.get("expected_steps"))
            results[arm] = r
            tracking = shots_to_competence_evaluated(logs, tol)
            closest, _ = best_evaluated_error(logs)
            med = f"{r.median:.0f}" if r.n_reached else "never"
            ratio = closest / tol if tol and np.isfinite(closest) else float("nan")
            reached = f"{r.n_reached}/{r.n_total}"
            print(f"    {arm:24s}{reached:>12s}{med:>16s}"
                  f"{len(r.censored_at):>10d}{tracking.reach_rate:>11.0%}"
                  f"{closest:>15.4f}{ratio:>8.2f}")
            lo, hi = binomial_exact_interval(r.n_reached, r.n_total)
            print(f"      reach {r.reach_rate:.0%} "
                  f"[exact 95% CI {lo:.0%}, {hi:.0%}]")
            lengths = sorted({len(log) for log in logs})
            if budget is not None and lengths != [budget]:
                print(f"      ! observed shot budgets {lengths} differ from "
                      f"the declared fixed budget {budget}")
            if man.get("seeds") is not None and len(logs) != man["seeds"]:
                print(f"      ! only {len(logs)} of {man['seeds']} declared seeds "
                      "are present; this is incomplete coverage")

        comparisons = [("scratch", "federated_uniform")]
        comparisons += list(zip(ARMS, ARMS[1:]))
        for lower, upper in comparisons:
            if lower not in results or upper not in results:
                continue
            try:
                paired = paired_joint_reach(
                    arms[lower], arms[upper], tol,
                    expected_steps=man.get("expected_steps"))
                print(f"    {upper} vs {lower}: {paired.summary()}")
            except ValueError as exc:
                print(f"    {upper} vs {lower}: paired comparison unavailable "
                      f"({exc})")
            sp = speedup_from_results(results[lower], results[upper])
            print(f"      reaching-seed timing diagnostic: {sp.summary()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/coldstart")
    ap.add_argument("--since", default="",
                    help="only directories whose name sorts at or after this")
    ap.add_argument("--last", type=int, default=0,
                    help="only the N most recent")
    args = ap.parse_args()

    dirs = sorted(p for p in Path(args.root).glob("*")
                  if (p / "runs.json").exists() and p.name >= args.since)
    if args.last:
        dirs = dirs[-args.last:]
    if not dirs:
        print(f"no completed folds under {args.root} matching {args.since!r}")
        return 1
    for d in dirs:
        try:
            report(d)
        except Exception as exc:  # a half-written fold must not stop the rest
            print(f"\n{d.name}: could not read ({type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
