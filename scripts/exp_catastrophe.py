#!/usr/bin/env python
"""Can a device avoid a regime it has never personally entered?

    uv run python scripts/exp_catastrophe.py --seeds 3 --plot

SPEC.md 1 states the claim in two halves and then says which one matters:

    "Federating role-matched control agents across research tokamaks makes them
     learn faster AND avoid catastrophic regimes they have never personally
     entered. The second half is the one that matters. Large tokamaks cannot
     generate training disruptions, because they cannot tolerate them. A device
     must therefore learn its limits from devices that have already crossed
     them."

Nothing had ever run it. This does.

THE DESIGN
----------
Peers train ACROSS the full beta_N band, including the high region where limits
bite -- small machines can afford that. The target trains only in a restricted
safe region, never approaching the limit, exactly as a machine that cannot
afford to. Then the target's FROZEN policy is evaluated in the region it has
never entered, and violations are counted.

WHY THIS CLAIM IS NOT BLOCKED BY THE HEADROOM GATE
---------------------------------------------------
Everything else here has been bounded by whether a speedup is bigger than
campaign noise. This is not a speed claim. Whether isolated training converges
in 80 shots or 8000 is irrelevant to whether the resulting controller is SAFE
somewhere it has never been. The gate does not gate this.

The bar is deliberately strict: `CatastropheResult.claim_holds` requires the
federated violation rate to be exactly zero against a non-zero isolated rate. A
merely-lower rate reports PARTIAL and says so.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import (  # noqa: E402
    BETA_N_MARGIN_AT_FULL_COMMAND,
    DEVICES,
    can_violate,
    get as get_device,
)
from hfmarl.envs.task import SetpointSchedule, get as get_task  # noqa: E402
from hfmarl.experiments.runner import run_catastrophe  # noqa: E402
from hfmarl.metrics.curves import (  # noqa: E402
    catastrophe_test,
    completion_rate,
    violation_rate,
)
from hfmarl.util.artifacts import new_run_dir, save_run  # noqa: E402
from hfmarl.util.report import record  # noqa: E402


def build_tasks(base_name: str, safe: float, unseen: float, sweep_lo: float,
                sweep_hi: float):
    """Three band-relative variants of one preset.

    Fractions of each device's MEASURED beta_N band, so "the top of the band"
    means the same thing on a 1.4 MW machine and a 53 MW one. That is the
    only reason a regime held out on one device is comparable to a regime
    explored on another.
    """
    base = get_task(base_name)
    # `name` stays the preset's, because BETA_N_BANDS is keyed by it and
    # the band is a property of the device and the TRANSPORT MODEL, not of
    # the setpoint. These variants change only where in that band the
    # target sits, so reusing the measured band is correct -- renaming them
    # would lose it and every fraction would resolve against nothing.
    peer = replace(
        base,
        setpoint=SetpointSchedule("ramp", base=sweep_lo,
                                  amplitude=sweep_hi - sweep_lo),
    )
    safe_task = replace(
        base, setpoint=SetpointSchedule("constant", base=safe),
    )
    unseen_task = replace(
        base, setpoint=SetpointSchedule("constant", base=unseen),
    )
    return peer, safe_task, unseen_task


def plot(res, iso_logs, fed_logs, out: Path, target: str, unseen: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                  gridspec_kw={"width_ratios": [1.0, 1.3]})

    rates = [res.isolated_rate, res.federated_rate]
    colours = ["#c8342f", "#2e8b3d" if res.federated_rate == 0 else "#d98a1f"]
    ax.bar([0, 1], rates, color=colours, alpha=0.88, width=0.6)
    for i, r in enumerate(rates):
        ax.text(i, r, f" {r:.1%}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"isolated\n(never saw {unseen:.0%} of band)",
                        "federated\n(inherited from peers that did)"],
                       fontsize=8.5)
    ax.set_ylabel("fraction of shots crossing a limit")
    ax.set_ylim(0, max(max(rates) * 1.25, 0.02))
    ax.set_title(f"{target} in a regime it never entered", fontsize=10.5)
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for logs, colour, label in ((iso_logs, "#c8342f", "isolated"),
                                (fed_logs, "#2e8b3d", "federated")):
        curves = [r.cumulative_violations() for r in logs if len(r)]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        stack = np.stack([c[:n] for c in curves]).astype(float)
        x = np.arange(1, n + 1)
        m = stack.mean(axis=0)
        axc.plot(x, m, color=colour, lw=1.9, label=label)
        if stack.shape[0] > 1:
            sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
            axc.fill_between(x, m - sem, m + sem, color=colour, alpha=0.18, lw=0)
    axc.set_xlabel("evaluation shot in the unseen regime")
    axc.set_ylabel("cumulative shots that violated")
    axc.set_title("Violations accumulate, or they do not", fontsize=10.5)
    axc.legend(fontsize=8.5, frameon=False)
    axc.grid(alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)

    fig.suptitle(
        "SPEC.md §1, second half — learning a limit from machines that crossed it",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tcv_like",
                    help="the machine that cannot afford to find its own "
                         "limits. Must be one whose limits are REACHABLE -- "
                         "see registry.can_violate.")
    ap.add_argument("--task", default="easy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--peer-shots", type=int, default=150)
    ap.add_argument("--target-shots", type=int, default=150)
    ap.add_argument("--eval-shots", type=int, default=60)
    ap.add_argument("--local-shots", type=int, default=25)
    ap.add_argument("--safe", type=float, default=0.20,
                    help="target's training setpoint, as a fraction of its band")
    ap.add_argument("--unseen", type=float, default=0.84,
                    help="evaluation setpoint: as high in the band as "
                         "setpoint+tolerance allows, i.e. against the limit")
    ap.add_argument("--sweep", default="0.20,0.84",
                    help="lo,hi fractions the PEERS ramp across, so they meet "
                         "the limit the target must learn about")
    ap.add_argument("--rule", default="geomedian")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    # THE TARGET MUST BE ABLE TO HAVE A CATASTROPHE. Measured by driving each
    # device from zero to full command: iter_like and sparc_like never come
    # within four soft-to-hard bands of the beta limit, so on either of them
    # "avoided a regime it had never entered" is satisfied by a controller
    # that does nothing, and the experiment reports a perfect score for a
    # question it never asked. This default used to be iter_like.
    if not can_violate(args.target):
        print(f"{args.target} cannot cross any limit under the thermal "
              f"cluster (worst beta_N margin at full command: "
              f"{BETA_N_MARGIN_AT_FULL_COMMAND[args.target]:+.2f}). "
              f"There is no catastrophe to avoid, so this experiment would "
              f"report safety it did not earn. Use a device with a reachable "
              f"limit: "
              f"{[d for d in sorted(BETA_N_MARGIN_AT_FULL_COMMAND) if can_violate(d)]}",
              flush=True)
        return 2

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    def quiet(*a, **k):
        pass

    peers = [d for d in sorted(DEVICES) if d != args.target]
    lo, hi = (float(x) for x in args.sweep.split(","))
    peer_task, safe_task, unseen_task = build_tasks(
        args.task, args.safe, args.unseen, lo, hi)

    say(f"target      : {args.target}  (trains only at {args.safe:.0%} of its band)")
    say(f"peers       : {peers}  (ramp across {lo:.0%}-{hi:.0%} of theirs)")
    say(f"unseen      : evaluation at {args.unseen:.0%} of the target's band, "
        "frozen policy, no learning")
    say(f"task        : {args.task}   seeds: {args.seeds}")
    say(f"budget      : peers {args.peer_shots}/device, target "
        f"{args.target_shots}, eval {args.eval_shots}")
    say(f"merge rule  : {args.rule}   reject-if-worse: {args.select}")

    # AUDIT #4. The withheld region, as an absolute beta_N. Everything at
    # or above this is what the target must never have seen -- and it is
    # now enforced by an interlock and CHECKED afterwards, rather than
    # assumed because the setpoint was lower.
    dev = get_device(args.target)
    resolved_unseen = unseen_task.resolve_for(dev)
    withheld = float(resolved_unseen.setpoint.base - resolved_unseen.tolerance)
    tol = float(safe_task.resolve_for(dev).tolerance)
    expected_steps = int(unseen_task.steps_per_shot)
    say(f"withheld    : beta_N >= {withheld:.3f} (enforced by interlock)")
    say(f"endpoint    : completed {expected_steps} steps + no violation + "
        f"|beta_N - target| <= {tol:.3f}")
    say()

    iso_train, iso_eval, fed_train, fed_eval = [], [], [], []
    metas: list[dict] = []
    received: dict = {}
    t_start = time.perf_counter()

    for federated in (False, True):
        label = "federated" if federated else "isolated"
        t0 = time.perf_counter()
        for seed in range(args.seeds):
            tr, ev, meta = run_catastrophe(
                args.target, peers, peer_task, safe_task, unseen_task,
                seed=seed, peer_shots=args.peer_shots,
                target_shots=args.target_shots, eval_shots=args.eval_shots,
                local_shots=args.local_shots, federated=federated,
                withheld_beta_N=withheld,
                rule=args.rule, accept_if_better=args.select, say=quiet,
            )
            (fed_train if federated else iso_train).append(tr)
            (fed_eval if federated else iso_eval).append(ev)
            if meta["received_weights"]:
                received = meta["received_weights"]
            metas.append(meta)
        say(f"  {label:10s} {time.perf_counter() - t0:7.1f}s")

    say()
    say("=" * 74)
    say("SAFE-REGION TRAINING (where the target actually lived)")
    say(f"  isolated  violation rate {violation_rate(iso_train):.1%}")
    say(f"  federated violation rate {violation_rate(fed_train):.1%}")
    say()
    breached = [m for m in metas if m.get("holdout_breached")]
    caps = sorted({m.get("action_cap") for m in metas})
    peak = max((m.get("max_beta_N_in_training", float("nan"))
                for m in metas), default=float("nan"))
    say(f"  interlock cap {caps}, peak beta_N seen in training {peak:.3f}")
    if breached:
        say(f"  HOLDOUT BREACHED in {len(breached)}/{len(metas)} runs -- the "
            "target entered the region it is supposed never to have seen, "
            "so this is not a held-out-region result.")
    else:
        say("  holdout intact at action-step granularity (TORAX substeps "
            "inside an action window are not visible here)")
    say()
    say("THE UNSEEN REGIME (frozen policy, never trained here)")
    res = catastrophe_test(iso_eval, fed_eval, tolerance=tol,
                           expected_steps=expected_steps)
    for ln in res.summary().splitlines():
        say("  " + ln)
    say()
    if received:
        say("  the target inherited from: " + "  ".join(
            f"{k} {v:.3f}" for k, v in sorted(received.items(),
                                              key=lambda kv: -kv[1])))
    say(f"  completion rate: isolated "
        f"{completion_rate(iso_eval, expected_steps):.1%}, federated "
        f"{completion_rate(fed_eval, expected_steps):.1%}")
    say(f"  total {time.perf_counter() - t_start:.0f}s")

    # AUDIT #10: a uniquely named directory with the raw per-shot records,
    # so a metric defect found later can be rescored instead of rerun.
    outdir = new_run_dir("catastrophe")
    summary = {
        "target": args.target, "peers": peers, "task": args.task,
        "safe": args.safe, "unseen": args.unseen, "sweep": [lo, hi],
        "withheld_beta_N": withheld, "tolerance": tol,
        "expected_steps": expected_steps,
        "seeds": args.seeds, "peer_shots": args.peer_shots,
        "target_shots": args.target_shots, "eval_shots": args.eval_shots,
        "rule": args.rule, "select": args.select,
        "isolated_rate": res.isolated_rate,
        "federated_rate": res.federated_rate,
        "isolated_success": res.isolated_success,
        "federated_success": res.federated_success,
        "claim_holds": res.claim_holds,
        "train_violation_isolated": violation_rate(iso_train),
        "train_violation_federated": violation_rate(fed_train),
        "completion_isolated": completion_rate(iso_eval, expected_steps),
        "completion_federated": completion_rate(fed_eval, expected_steps),
        "holdout_breached_runs": len(breached),
        "runs_total": len(metas),
        "action_caps": caps,
        "peak_beta_N_in_training": peak,
        "received_weights": received,
    }
    save_run(outdir, manifest={
        "experiment": "catastrophe", "argv": sys.argv[1:],
        "target": args.target, "peers": peers, "task": args.task,
        "withheld_beta_N": withheld, "tolerance": tol,
        "eval_disturbance": "set by run_catastrophe (independent per shot)",
        "endpoint": "completed + contained + tracking",
        "per_run_meta": metas,
    }, runs={
        "isolated_train": iso_train, "isolated_eval": iso_eval,
        "federated_train": fed_train, "federated_eval": fed_eval,
    }, summary=summary)
    say(f"wrote {outdir}")

    if args.plot:
        p_png = plot(res, iso_eval, fed_eval, outdir / "catastrophe.png",
                     args.target, args.unseen)
        say(f"wrote {p_png}")

    if not args.no_record:
        path = record(
            f"Catastrophe transfer -- {args.target}, unseen at "
            f"{args.unseen:.0%} of band, {args.seeds} seeds",
            "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {path})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
