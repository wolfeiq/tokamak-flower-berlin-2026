#!/usr/bin/env python
"""A new reactor with no data joins a federation that has been running.

    uv run python scripts/exp_coldstart.py --seeds 3 --pretrain 100 --join 100 --plot

METRICS.md metric 4, and the strongest practical number in the set. Three
machines train; a fourth arrives with zero shots of its own and is handed a
controller aggregated from them, weighted toward its own operating point. The
question is how many of ITS shots it needs before it tracks.

WHY THIS RATHER THAN A SPEEDUP RATIO
------------------------------------
A speedup on shots-to-competence needs the isolated baseline to be slow enough
that the gap clears campaign noise, and `gate_headroom.py` currently says it is
not. Cold start does not have that problem: it measures the first shots on a
machine that has none, where the gap is largest, and where the operational
meaning is direct -- at 20-40 shots/day, 40 shots against 240 is a week of
commissioning against a month.

WHAT IT CAN FALSIFY
-------------------
Similarity theory predicts WHO benefits, not just that someone does. A joiner
close to the federation in (rho*, nu*, beta_N, q95) should inherit more than a
distant one: sparc_like sits 0.595 from iter_like, tcv_like 1.669. Leave-one-out
over both tests that prediction. If the distant machine gains as much as the
near one, the physics weighting is decoration and SPEC.md 4b earns nothing --
which is a result worth reporting, not a failure to hide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, encoded_states, get as get_device  # noqa: E402
from hfmarl.envs.task import get as get_task  # noqa: E402
from hfmarl.experiments.runner import (  # noqa: E402
    derangements,
    run_cold_start,
)
from hfmarl.federation.similarity import similarity_distance  # noqa: E402
from hfmarl.metrics.curves import (  # noqa: E402
    asymptotic_performance,
    asymptotic_performance_evaluated,
    best_evaluated_error,
    binomial_exact_interval,
    shots_to_competence,
    shots_to_joint_competence_evaluated,
    speedup_from_results,
)
from hfmarl.util.artifacts import _jsonable, new_run_dir, save_run  # noqa: E402
from hfmarl.util.report import record  # noqa: E402
from hfmarl.metrics.log import RunLog  # noqa: E402

# PROTOCOL.md 3. Each rung adds exactly one ingredient to the one above,
# so a gap between adjacent rungs is attributable to that ingredient. The
# earlier three-arm set could not separate pretraining from multiple
# sources from repeated exchange, because every federated arm had all three.
ARMS = [
    ("scratch", dict(arm="scratch")),
    ("single_source", dict(arm="single_source")),
    ("handover_merge", dict(arm="handover_merge")),
    ("federated_uniform", dict(arm="federated_uniform")),
    ("federated_similarity", dict(arm="federated_similarity")),
]
COLOURS = {
    "scratch": "#8b9096",
    "single_source": "#9a6fb0",
    "handover_merge": "#d98a1f",
    "federated_uniform": "#3d6ba5",
    "federated_similarity": "#2e8b3d",
}
LABEL = {
    "scratch": "its own shots only",
    "single_source": "one pretrained source",
    "handover_merge": "all sources, merged once",
    "federated_uniform": "federated, uniform weights",
    "federated_similarity": "federated, similarity weights",
}


def snapshot_source(outdir: Path) -> dict[str, str]:
    """Save the exact code, including uncommitted fixes, used by this run."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "hfmarl").rglob("*.py"))
    paths += [Path(__file__).resolve(), root / "scripts/report_coldstart.py",
              root / "PROTOCOL.md", root / "pyproject.toml"]
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        target = outdir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_checkpoint(directory: Path, settings: dict):
    """Refuse to combine different experiments or code in a resumed run."""
    checkpoint = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    manifest = checkpoint["manifest"]
    for key, value in settings.items():
        if manifest.get(key) != value:
            raise ValueError(f"cannot resume: {key} differs from saved settings")
    root = Path(__file__).resolve().parents[1]
    for relative, digest in manifest.get("source_sha256", {}).items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"cannot resume: source changed: {relative}")
    raw = checkpoint["runs"]
    runs = {key: [RunLog.from_dict(row) for row in rows]
            for key, rows in raw.items()}
    metadata = checkpoint["run_metadata"]
    return manifest, runs, metadata


def save_checkpoint(directory: Path, manifest: dict, runs: dict, metadata: dict):
    """Replace one complete checkpoint atomically, then write analysis views."""
    payload = {"manifest": manifest,
               "runs": {key: [run.to_dict() for run in logs]
                        for key, logs in runs.items()},
               "run_metadata": metadata}
    temporary = directory / "checkpoint.json.tmp"
    temporary.write_text(json.dumps(_jsonable(payload), allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "checkpoint.json")
    save_run(directory, manifest=manifest, runs=runs,
             summary={"status": manifest["status"], "run_metadata": metadata})


def mean_distance(joiner: str, incumbents: list[str]) -> float:
    s = encoded_states()
    return float(np.mean([similarity_distance(s[joiner], s[i])
                          for i in incumbents]))


def plot(results: dict, out: Path, task_name: str, join_shots: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joiners = list(results)
    fig, axes = plt.subplots(2, len(joiners), figsize=(6.4 * len(joiners), 8.8),
                             squeeze=False)
    for column, joiner in enumerate(joiners):
        ax, curve = axes[:, column]
        arms = [a for a, _ in ARMS if a in results[joiner]["competence"]]
        vals, err_lo, err_hi = [], [], []
        for a in arms:
            r = results[joiner]["competence"][a]
            vals.append(r.reach_rate)
            lo, hi = binomial_exact_interval(r.n_reached, r.n_total)
            err_lo.append(r.reach_rate - lo)
            err_hi.append(hi - r.reach_rate)
            x = np.arange(join_shots + 1)
            y = np.array([sum(v <= shot for v in r.values) / r.n_total for shot in x])
            curve.step(x, y, where="post", color=COLOURS[a], label=a)
        ax.bar(range(len(arms)), vals, yerr=[err_lo, err_hi], capsize=4,
               color=[COLOURS[a] for a in arms], alpha=0.88)
        for i, a in enumerate(arms):
            r = results[joiner]["competence"][a]
            txt = f"{r.n_reached}/{r.n_total}"
            ax.text(i, 1.06, txt, ha="center", va="bottom", fontsize=9,
                    color="#c8342f" if not r.n_reached else "#2b2b2b")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([LABEL[a].replace(" (", "\n(").replace(", ", ",\n")
                            for a in arms], fontsize=7.6)
        ax.set_ylabel("fraction reaching joint competence (exact 95% CI)")
        ax.set_ylim(-0.03, 1.18)
        ax.set_title(
            f"{joiner} joins  —  mean dimensionless distance "
            f"{results[joiner]['distance']:.2f}", fontsize=10)
        ax.grid(axis="y", alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
            curve.spines[side].set_visible(False)
        curve.set(xlabel="new-device shots (plus 5 cached calibration shots)",
                  ylabel="fraction of all seeds reaching competence",
                  xlim=(0, join_shots), ylim=(-0.03, 1.03))
        curve.grid(alpha=0.18)
        curve.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        f"Cold start on '{task_name}' — completed + safe + tracking, "
        "two consecutive incumbent evaluations\n"
        f"Budget {join_shots} new-device shots; source costs separate; all seeds retained",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joiners", default="sparc_like,tcv_like",
                    help="devices to hold out, one fold each. The default pair "
                         "is the nearest and the furthest in dimensionless "
                         "space, which is what tests the prediction.")
    ap.add_argument("--arms", default=",".join(a for a, _ in ARMS),
                    help="comma-separated arms; defaults to the full comparison ladder")
    ap.add_argument("--resume", type=Path,
                    help="resume per-seed checkpoints using identical settings and code")
    ap.add_argument("--task", default="moderate")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pretrain", type=int, default=100,
                    help="shots per incumbent before the joiner arrives")
    ap.add_argument("--join", type=int, default=100,
                    help="shot budget for the joiner itself")
    ap.add_argument("--local-shots", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="shots between evaluations of the incumbent. "
                         "Competence can only be resolved to this "
                         "granularity: at 10, four different arms all "
                         "reported exactly 41 shots and the comparison "
                         "between them carried no information. The cost is "
                         "budget -- at 5 a fifth of the joiner's shots are "
                         "evaluations, and they are charged like any other.")
    ap.add_argument("--scramble", type=int, default=0,
                    help="permutation control: run N DERANGEMENTS of the "
                         "device-to-coordinates assignment, cycling over "
                         "seeds, so every source publishes under a peer's "
                         "measured region. If the benefit ordering survives "
                         "this, it was never the physics. One permutation is "
                         "an anecdote; the null distribution is the control.")
    ap.add_argument("--rule", default="geomedian",
                    choices=["geomedian", "mean", "median"],
                    help="how buffered updates are combined. mean is the "
                         "unprotected baseline: one bad client moves it "
                         "arbitrarily far.")
    ap.add_argument("--no-align", action="store_true",
                    help="ablation: skip permuting hidden units into the "
                         "target basis before merging, so the merge "
                         "averages coordinates that do not correspond.")
    ap.add_argument("--clip-factor", type=float, default=2.0,
                    help="centred-clipping radius, in multiples of the "
                         "median deviation. 0 disables it.")
    ap.add_argument("--select", action="store_true",
                    help="refuse an aggregate that scores worse than the "
                         "client already had. The robust rules bound how "
                         "much a bad peer can move the merge; this refuses "
                         "the merge outright, using the evaluation shot "
                         "that is fired and paid for either way.")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arm_names or len(set(arm_names)) != len(arm_names):
        ap.error("--arms must contain distinct arm names")
    unknown = set(arm_names) - {a for a, _ in ARMS}
    if unknown:
        ap.error(f"unknown arms: {sorted(unknown)}")
    arms = [(a, kw) for a, kw in ARMS if a in arm_names]
    if min(args.seeds, args.pretrain, args.join, args.local_shots, args.eval_every) < 1:
        ap.error("seeds and shot/evaluation budgets must be positive")

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    def quiet(*a, **k):
        pass

    joiners = [j.strip() for j in args.joiners.split(",") if j.strip()]
    if not joiners or len(set(joiners)) != len(joiners) or set(joiners) - set(DEVICES):
        ap.error("--joiners must contain distinct known devices")
    task = get_task(args.task)
    # AUDIT #1. Tolerance is a FRACTION of each device's own measured
    # beta_N band until resolve_for binds it, so it is PER DEVICE.
    # Resolving it once from joiners[0] scored every joiner against the
    # first one's criterion: on `easy` that made tcv_like face a bar 24.2x
    # stricter than the task it had trained on, and simply reordering
    # --joiners changed the verdict.
    tolerances = {j: float(task.resolve_for(get_device(j)).tolerance)
                  for j in joiners}

    say(f"task        : {args.task} ({task.steps_per_shot} steps/shot)")
    say("tolerances  : " + "  ".join(f"{k} {v:.4f}"
                                     for k, v in tolerances.items()))
    say(f"joiners     : {joiners}   (leave-one-out)")
    say(f"seeds       : {args.seeds}   pretrain/incumbent: {args.pretrain}   "
        f"joiner budget: {args.join}")
    # Printed and recorded: two runs differing only in the merge rule are
    # the ablation, and a FINDINGS block that does not say which rule
    # produced it cannot be paired with its counterpart.
    clip_label = args.clip_factor if args.clip_factor else "off"
    say(f"merge rule  : {args.rule}   align: {not args.no_align}   "
        f"clip: {clip_label}   reject-if-worse: {args.select}")
    # THE CONTROL HAS TO BE IN THE HEADER. A scrambled fold whose output
    # looks like a real one is worse than no control: it is a wrong number
    # with a real number's provenance.
    scrambles = (derangements(sorted(DEVICES), args.scramble)
                 if args.scramble else [])
    if scrambles:
        say(f"SCRAMBLE    : {len(scrambles)} derangement(s) cycling over "
            "seeds -- coordinates are PERMUTED, this is the negative control")
        for i, d in enumerate(scrambles):
            say("              " + f"[{i}] " + "  ".join(
                f"{k}->{v}" for k, v in sorted(d.items())))
    else:
        say("scramble    : none (true coordinates)")
    say()

    results: dict[str, dict] = {}
    t_start = time.perf_counter()
    settings = {
        "experiment": "coldstart", "task": args.task, "joiners": joiners,
        "arms": [a for a, _ in arms], "seeds": args.seeds,
        "pretrain": args.pretrain, "join": args.join,
        "local_shots": args.local_shots, "window": args.window,
        "eval_every": args.eval_every, "scramble": args.scramble,
        "rule": args.rule, "align": not args.no_align,
        "clip_factor": args.clip_factor or None, "select": args.select,
        "tolerances": tolerances, "expected_steps": task.steps_per_shot,
        "metric": "shots_to_joint_competence_evaluated (two consecutive incumbent evaluations)",
        "upstream_calibration_shots_per_joiner": 5,
    }
    if args.resume:
        outdir = args.resume.resolve()
        manifest, checkpoint_runs, run_metadata = load_checkpoint(outdir, settings)
    else:
        outdir = new_run_dir("coldstart")
        manifest = dict(settings, argv=sys.argv[1:], status="running",
                        source_sha256=snapshot_source(outdir))
        checkpoint_runs, run_metadata = {}, {}
    say(f"checkpoints : {outdir}")
    say("budgets     : joiner shots include probes, training and evaluations; "
        "add 5 upstream calibration shots once per joiner. Source shots are separate.")
    manifest["status"] = "running"
    save_checkpoint(outdir, manifest, checkpoint_runs, run_metadata)

    for joiner in joiners:
        incumbents = [d for d in sorted(DEVICES) if d != joiner]
        dist = mean_distance(joiner, incumbents)
        say(f"=== {joiner} joins {incumbents} "
            f"(mean distance {dist:.3f}) ===")

        logs: dict[str, list] = {a: list(checkpoint_runs.get(f"{joiner}:{a}", []))
                                 for a, _ in arms}
        received: dict[str, dict] = {}
        decisions: dict[str, list] = {}
        empty_aggregates: dict[str, list] = {}
        for arm, kwargs in arms:
            t0 = time.perf_counter()
            for seed in range(args.seeds):
                key = f"{joiner}:{arm}"
                existing = next((log for log in logs[arm] if log.seed == seed), None)
                if existing is not None:
                    meta = run_metadata[key][str(seed)]
                else:
                    log, meta = run_cold_start(
                        joiner, incumbents, args.task, seed=seed,
                        pretrain_shots=args.pretrain, join_shots=args.join,
                        local_shots=args.local_shots, eval_every=args.eval_every,
                        rule=args.rule,
                        align=not args.no_align,
                        clip_factor=args.clip_factor or None,
                        accept_if_better=args.select,
                        scramble=(scrambles[seed % len(scrambles)]
                                  if scrambles else None),
                        say=quiet, **kwargs,
                    )
                    logs[arm].append(log)
                    checkpoint_runs[key] = logs[arm]
                    run_metadata.setdefault(key, {})[str(seed)] = meta
                    save_checkpoint(outdir, manifest, checkpoint_runs, run_metadata)
                    say(f"    {arm} seed {seed + 1}/{args.seeds} saved "
                        f"({len(log)} joiner shots)")
                if meta["received_weights"]:
                    received[arm] = meta["received_weights"]
                decisions.setdefault(arm, []).append(meta["handover_adopted"])
                empty_aggregates.setdefault(arm, []).append(
                    (meta.get("aggregate_rejected_all", False),
                     meta.get("aggregate_error", ""),
                     int(meta.get("rejected_rounds", 0)),
                     int(meta.get("error_rounds", 0)),
                     int(meta.get("source_rejections", 0)),
                     int(meta.get("uniform_fallback_rounds", 0)),
                     int(meta.get("source_acceptance_shots", 0)),
                     int(meta.get("joiner_acceptance_shots", 0))))
            say(f"  {arm:26s} {time.perf_counter() - t0:6.1f}s")

        # Both conventions, side by side. The candidate-stream number is
        # kept because every earlier run reported it and a metric
        # change that quietly replaces the old number is not a
        # comparison.
        tol = tolerances[joiner]
        comp = {a: shots_to_joint_competence_evaluated(
                    logs[a], tol, expected_steps=task.steps_per_shot)
                for a, _ in arms}
        comp_cand = {a: shots_to_competence(logs[a], tol, window=args.window)
                     for a, _ in arms}
        results[joiner] = {
            "distance": dist, "incumbents": incumbents, "tolerance": tol,
            "competence": comp, "logs": logs, "received": received,
        }

        say()
        say(f"  {'arm':28s}{'joint competence':>16s}{'reach':>8s}"
            f"{'on candidates':>16s}{'plateau(ev)':>13s}{'plateau(all)':>14s}"
            f"{'closest':>12s}")
        for arm, _ in arms:
            r, rc = comp[arm], comp_cand[arm]
            med = f"{r.median:.0f}" if r.n_reached else "never"
            medc = f"{rc.median:.0f}" if rc.n_reached else "never"
            # Both plateaus, because they disagree. The all-shots number
            # averages candidates the search was still probing with, so it
            # measures how hard an arm explored as much as how good it ended
            # up; the evaluation-only number is the controller alone.
            plateau, _ = asymptotic_performance(logs[arm])
            plateau_ev, _ = asymptotic_performance_evaluated(logs[arm])
            # How close it got, for the folds where nothing passes. On a
            # device with a tight tolerance -- sparc_like's is 0.0091 --
            # every arm can report "never" and the fold says nothing; the
            # best evaluated error still ranks them, in the tolerance's own
            # units.
            closest, _ = best_evaluated_error(logs[arm])
            say(f"  {arm:28s}{med:>16s}{r.reach_rate:>7.0%}"
                f"{medc:>16s}{plateau_ev:>13.3f}{plateau:>14.3f}"
                f"{closest:>12.4f}")
        say()
        # Adjacent rungs. The ladder exists so a gap is attributable to the
        # ONE ingredient that differs, and comparing everything against
        # scratch throws that structure away -- it would report the same
        # number for "federation helped" as for "a warm start helped".
        names = [a for a, _ in ARMS]
        for lower, upper in zip(names, names[1:]):
            if lower not in comp or upper not in comp:
                continue
            sp = speedup_from_results(comp[lower], comp[upper])
            say(f"  {upper} vs {lower}: {sp.summary()}")
            for w in sp.warnings:
                say(f"    ! {w}")
        # Whether each arm actually took what it was handed. With
        # reject-if-worse on, an arm that refuses every handover is scratch
        # under another name, and the table would not otherwise say so.
        for arm, _ in arms:
            took = decisions.get(arm)
            if took and any(t is not None for t in took):
                n_yes = sum(1 for t in took if t)
                say(f"  {arm:24s} adopted the handover in {n_yes}/{len(took)}"
                    " seeds")
        # An arm whose aggregate came back empty on every seed IS scratch,
        # and the only difference is the label on the row. The server has
        # always known; nothing used to ask it.
        for arm, _ in arms:
            rows = empty_aggregates.get(arm) or []
            n_empty = sum(1 for r in rows if r[0])
            errs = sorted({r[1] for r in rows if r[1]})
            rej = sum(r[2] for r in rows)
            errr = sum(r[3] for r in rows)
            src = sum(r[4] for r in rows)
            unif = sum(r[5] for r in rows)
            src_acc = sum(r[6] for r in rows)
            join_acc = sum(r[7] for r in rows)
            if n_empty:
                say(f"  ! {arm:24s} handover aggregate REJECTED EVERY CLIENT "
                    f"in {n_empty}/{len(rows)} seeds -- inherited nothing")
            if rej:
                say(f"  ! {arm:24s} {rej} pretraining round(s) rejected every "
                    f"client across {len(rows)} seeds")
            if errr:
                say(f"  ! {arm:24s} {errr} round(s) failed to aggregate at all")
            if src:
                say(f"    {arm:24s} sources refused the aggregate {src} "
                    "time(s) (reject-if-worse)")
            if unif:
                say(f"  ! {arm:24s} similarity kernel underflowed in {unif} "
                    "round(s) -- weights were uniform, so this arm was "
                    "federated_uniform there")
            if src_acc or join_acc:
                # SELECTION IS NOT FREE, and it is not paid evenly. An arm
                # that never exchanges never measures its incumbent to
                # refuse something, so `scratch` gets one more training shot
                # out of the joiner's budget than every inheriting arm.
                say(f"    {arm:24s} spent {join_acc} joiner and {src_acc} "
                    f"source shot(s) measuring the bar for reject-if-worse")
            for e in errs:
                say(f"  ! {arm:24s} aggregation error: {e}")
        say()
        # EVERY arm's weights, not just the similarity one. The ladder bug --
        # handover_merge merging with similarity weighting and no safety
        # penalty while the rung above it merged uniformly with one -- was
        # invisible in this output and had to be dug out of `runs.json`
        # afterwards. Three rows next to each other would have shown it
        # immediately: 0.558/0.232/0.210 against 0.377/0.377/0.245 is not two
        # settings of the same rule.
        for arm, _ in arms:
            w = received.get(arm)
            if w:
                say(f"  {arm:24s} inherited from: " + "  ".join(
                    f"{k} {v:.3f}"
                    for k, v in sorted(w.items(), key=lambda kv: -kv[1])))
        say()

    # --- does distance predict the benefit? ------------------------------
    say("=" * 74)
    say("The prediction similarity theory makes: a closer joiner inherits more.")
    say()
    say(f"{'joiner':14s}{'distance':>10s}{'scratch':>10s}{'similarity':>12s}"
        f"{'gain':>10s}")
    for joiner, r in results.items():
        if not {"scratch", "federated_similarity"} <= r["competence"].keys():
            continue
        sc = r["competence"]["scratch"]
        si = r["competence"]["federated_similarity"]
        a = f"{sc.median:.0f}" if sc.n_reached else "never"
        b = f"{si.median:.0f}" if si.n_reached else "never"
        comparison = speedup_from_results(sc, si)
        gain = comparison.summary()
        say(f"{joiner:14s}{r['distance']:>10.3f}{a:>10s}{b:>12s}{gain:>10s}")
    say()

    # AUDIT #10. A uniquely named directory, with the raw per-shot
    # records beside the summary. The previous stem omitted the joiner
    # set, the clipping factor, the local-shot interval and the window,
    # so two different runs wrote to one filename; and summaries alone
    # could not be rescored when a metric defect was found.
    stem = "coldstart"
    payload = {
        "task": args.task, "seeds": args.seeds, "pretrain": args.pretrain,
        "join": args.join, "local_shots": args.local_shots,
        "tolerances": tolerances,
        "rule": args.rule, "align": not args.no_align,
        "clip_factor": args.clip_factor or None, "select": args.select,
        "results": {
            j: {
                "distance": r["distance"],
                "tolerance": r["tolerance"],
                "incumbents": r["incumbents"],
                "received": r["received"],
                "competence": {
                    a: {"median": None if not c.n_reached else c.median,
                        "values": c.values, "censored_at": c.censored_at,
                        "reach_rate": c.reach_rate}
                    for a, c in r["competence"].items()
                },
                "plateau": {a: asymptotic_performance(r["logs"][a])[0]
                            for a, _ in arms},
                "plateau_evaluated": {
                    a: asymptotic_performance_evaluated(r["logs"][a])[0]
                    for a, _ in arms},
                "closest_evaluated_error": {
                    a: best_evaluated_error(r["logs"][a])[0]
                    for a, _ in arms},
            }
            for j, r in results.items()
        },
    }
    raw = {f"{joiner}:{arm}": logs
           for joiner, r in results.items()
           for arm, logs in r["logs"].items()}
    payload.update(status="complete", run_metadata=run_metadata)
    manifest["status"] = "complete"
    save_checkpoint(outdir, manifest, raw, run_metadata)
    save_run(outdir, manifest=manifest, runs=raw, summary=payload)
    say(f"wrote {outdir}")

    if args.plot:
        say(f"wrote {plot(results, outdir / f'{stem}.png', args.task, args.join)}")

    say(f"total {time.perf_counter() - t_start:.0f}s")

    if not args.no_record:
        path = record(
            f"Cold start -- {args.task}, {args.join} joiner shots x "
            f"{args.seeds} seeds, merge rule {args.rule}",
            "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {path})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
