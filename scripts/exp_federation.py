#!/usr/bin/env python
"""Federation vs no federation -- the comparison the whole project is for.

    uv run python scripts/exp_federation.py --shots 300 --seeds 2 --plot

Runs the conditions of METRICS.md "The five conditions" on the same devices,
the same seeds and the same per-device shot budget, and reports shots to
competence, the speedup ratio over isolated, final performance and violations.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
It can show whether federating helps at all (isolated vs fedbuff_uniform) and
whether weighting peers by dimensionless distance helps beyond that
(fedbuff_uniform vs fedbuff_similarity -- the gap that IS SPEC.md §4b).

It cannot currently run the role-blind negative control: with the hierarchy
removed there is one cluster, so role-blind and role-matched aggregation are
the same computation. The runner refuses rather than reporting a duplicate as
a control.

And a speedup reported here is bounded above by what the headroom gate allows.
If isolated reaches competence in a couple of hundred shots, a ratio computed
from it is a ratio between two small numbers, and the gate says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get as get_device  # noqa: E402
from hfmarl.envs.task import get as get_task  # noqa: E402
from hfmarl.experiments.conditions import (  # noqa: E402
    CONDITION_ORDER,
    ISOLATED,
    LABELS,
)
from hfmarl.experiments.runner import (  # noqa: E402
    RoleControlUnavailable,
    derangements,
    run_condition,
)
from hfmarl.metrics.curves import (  # noqa: E402
    ThresholdResult,
    asymptotic_performance,
    learning_curve,
    shots_to_competence,
    speedup_from_results,
    violation_rate,
)
from hfmarl.util.report import record  # noqa: E402

DEFAULT_CONDITIONS = "isolated,fedbuff_uniform,fedbuff_similarity"


def plot(curves: dict, comp: dict, out: Path, task_name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "isolated": "#8b9096",
        "fedbuff_uniform": "#3d6ba5",
        "fedbuff_similarity": "#2e8b3d",
        "fedavg_naive": "#c8342f",
        "centralised": "#d98a1f",
    }
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.7, 1.0]})

    for name, (x, mean, sem) in curves.items():
        c = colours.get(name, "#555555")
        ax.plot(x, mean, color=c, lw=1.8, label=LABELS.get(name, name))
        ax.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("shots fired on this device")
    ax.set_ylabel("episode return (trailing mean)")
    ax.set_title(f"Learning curves — task '{task_name}'", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    names = [n for n in CONDITION_ORDER if n in comp]
    vals = [comp[n].median if comp[n].n_reached else np.nan for n in names]
    errs = np.array([
        [max(0.0, comp[n].median - comp[n].ci()[0]) if comp[n].n_reached > 1 else 0.0
         for n in names],
        [max(0.0, comp[n].ci()[1] - comp[n].median) if comp[n].n_reached > 1 else 0.0
         for n in names],
    ])
    axb.bar(range(len(names)), vals, yerr=errs, capsize=4,
            color=[colours.get(n, "#555") for n in names], alpha=0.85)
    for i, n in enumerate(names):
        if not comp[n].n_reached:
            axb.text(i, 0, "never\nreached", ha="center", va="bottom",
                     fontsize=7, color="#c8342f")
        else:
            axb.text(i, vals[i], f" {vals[i]:.0f}", ha="center", va="bottom",
                     fontsize=8)
    axb.set_xticks(range(len(names)))
    axb.set_xticklabels([n.replace("fedbuff_", "fb\n") for n in names],
                        fontsize=7.5)
    axb.set_ylabel("shots to competence")
    axb.set_title("|β_N − target| inside tolerance, held", fontsize=10)
    axb.grid(axis="y", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--devices", default=",".join(sorted(DEVICES)))
    ap.add_argument("--task", default="moderate")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shots", type=int, default=400,
                    help="per-device shot budget, adoption shots included")
    ap.add_argument("--local-shots", type=int, default=25,
                    help="shots between exchanges; small keeps clients from "
                         "drifting apart, which is what makes averaging valid")
    ap.add_argument("--stagger", action="store_true",
                    help="give devices different publish periods, so buffered "
                         "updates are genuinely stale and FedBuff's staleness "
                         "term does something")
    ap.add_argument("--no-safety", action="store_true",
                    help="ablation: switch off the physics gates "
                         "(regime and violation rejection), keeping "
                         "every generic robustness layer. This is the "
                         "arm that isolates what the physics buys.")
    ap.add_argument("--scramble", type=int, default=0,
                    help="permutation control: run N DERANGEMENTS of the "
                         "device-to-coordinates assignment, so each "
                         "device publishes under a peer's measured "
                         "region. One permutation is an anecdote; the "
                         "null distribution is the control.")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    device_names = [d.strip() for d in args.devices.split(",") if d.strip()]
    task = get_task(args.task)

    stagger = None
    if args.stagger:
        # Different campaign cadences. Nothing tunes these; they just have to
        # differ, or every buffered update has tau = 0 and the staleness term
        # is multiplying by one.
        stagger = {n: p for n, p in zip(device_names, (1, 2, 1, 3))}

    say(f"task        : {args.task} (difficulty {task.expected_difficulty}/5, "
        f"{task.steps_per_shot} steps/shot)")
    say(f"devices     : {device_names}")
    say(f"conditions  : {conditions}")
    say(f"seeds       : {args.seeds}   shots/device: {args.shots}   "
        f"local: {args.local_shots}")
    say(f"stagger     : {stagger or 'none (every device publishes every round)'}")
    say(f"physics gates: {'OFF (ablation)' if args.no_safety else 'on'}")
    scrambles = derangements(device_names, args.scramble) if args.scramble else []
    say(f"scramble    : {len(scrambles)} derangement(s)" if scrambles
        else "scramble    : none (true coordinates)")
    say()

    logs: dict[str, list] = {}
    meta: dict[str, dict] = {}
    t_start = time.perf_counter()

    for cond in conditions:
        say(f"=== {cond} ===")
        runs = []
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            try:
                r, m, _ = run_condition(
                    cond, device_names, args.task, seed=seed, shots=args.shots,
                    local_shots=args.local_shots, stagger=stagger,
                    use_safety=not args.no_safety,
                    scramble=scrambles[seed % len(scrambles)] if scrambles else None,
                    verbose_every=0, say=say,
                )
            except RoleControlUnavailable as exc:
                say(f"  SKIPPED: {exc}")
                runs = []
                break
            runs.extend(r)
            meta[cond] = m
            say(f"  seed {seed}: {time.perf_counter() - t0:.1f}s  "
                f"({m['rounds']} rounds)")
        if runs:
            logs[cond] = runs
        say()

    if ISOLATED not in logs:
        say("No isolated baseline: every ratio below would have no denominator.")
        return 2

    # --- metrics ---------------------------------------------------------
    # Tolerance is a FRACTION of each device's measured beta_N band until
    # AUDIT #1, a second time, in the script the audit did not review.
    # `resolve_for` binds the tolerance to ONE device, and this took
    # `device_names[0]` and scored all four against it. On `moderate` that
    # is diiid_like's 0.2401 applied to sparc_like, whose own tolerance is
    # 0.0091 -- a bar 26x looser than the task it trained on. The fix landed
    # in exp_coldstart.py and stopped there; this file kept the defect and
    # produced a 300-shot, 2-seed result with it.
    tolerances = {d: float(task.resolve_for(get_device(d)).tolerance)
                  for d in device_names}

    def competence(runs):
        # Per-device scoring; pooled only after each device has been
        # judged against its own criterion.
        values, censored = [], []
        for r in runs:
            one = shots_to_competence([r], tolerances[r.device],
                                      window=args.window)
            values += one.values
            censored += one.censored_at
        return ThresholdResult(values=values, censored_at=censored,
                               threshold=float('nan'), window=args.window)

    comp = {c: competence(runs) for c, runs in logs.items()}

    say("=" * 74)
    say("competence: |beta_N - target| inside EACH DEVICE'S OWN tolerance, "
        f"trailing mean over {args.window}, held 5 shots")
    say("  " + "  ".join(f"{d} {t:.4f}" for d, t in tolerances.items()))
    say()
    say(f"{'condition':26s}{'shots to competence':>24s}{'reach':>8s}"
        f"{'plateau':>10s}{'viol':>7s}")
    for c in CONDITION_ORDER:
        if c not in logs:
            continue
        plateau, _ = asymptotic_performance(logs[c])
        med = f"{comp[c].median:.0f}" if comp[c].n_reached else "never"
        say(f"{c:26s}{med:>24s}{comp[c].reach_rate:>7.0%}"
            f"{plateau:>10.3f}{violation_rate(logs[c]):>7.1%}")
    say()

    for c in conditions:
        if c == ISOLATED or c not in logs:
            continue
        sp = speedup_from_results(comp[ISOLATED], comp[c])
        say(f"{c} vs isolated: {sp.summary()}")
        for w in sp.warnings:
            say(f"  ! {w}")
        say()

    # --- outputs ---------------------------------------------------------
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "federation"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.shots}x{args.seeds}"

    payload = {
        "task": args.task, "devices": device_names, "seeds": args.seeds,
        "shots": args.shots, "local_shots": args.local_shots,
        "stagger": stagger, "tolerances": tolerances,
        "meta": meta,
        "competence": {
            c: {"median": None if not r.n_reached else r.median,
                "values": r.values, "censored_at": r.censored_at,
                "reach_rate": r.reach_rate}
            for c, r in comp.items()
        },
        "plateau": {c: asymptotic_performance(r)[0] for c, r in logs.items()},
        "violation_rate": {c: violation_rate(r) for c, r in logs.items()},
    }
    jpath = outdir / f"{stem}.json"
    jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    say(f"wrote {jpath}")

    if args.plot:
        curves = {c: learning_curve(r, window=args.window)
                  for c, r in logs.items()}
        p = plot(curves, comp, outdir / f"{stem}.png", args.task)
        say(f"wrote {p}")

    say(f"total {time.perf_counter() - t_start:.0f}s")

    if not args.no_record:
        path = record(f"Federation vs isolated -- {args.task}, "
                      f"{args.shots} shots x {args.seeds} seeds",
                      "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {path})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
