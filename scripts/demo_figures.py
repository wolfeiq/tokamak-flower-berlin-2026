#!/usr/bin/env python
"""Generate all six deliverable figures from SYNTHETIC data. Needs no TORAX.

Purpose: see what the deliverables look like, and check the plotting code, long
before any real run exists. The synthetic data is generated with the structure
the claim PREDICTS -- federation faster, role-blind FedAvg at or below isolated,
violations flattening earlier, benefit decaying with dimensionless distance.

    ==> These figures are NOT results. They are a rendering test. <==

Every output is stamped SYNTHETIC in the corner so a stray PNG can never be
mistaken for a finding.

    uv run python scripts/demo_figures.py --out results/figures_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import all_devices, encoded_states  # noqa: E402
from hfmarl.experiments.conditions import CONDITION_ORDER  # noqa: E402
from hfmarl.federation.similarity import similarity_distance  # noqa: E402
from hfmarl.metrics import plots  # noqa: E402
from hfmarl.metrics.curves import shots_to_threshold, speedup_ratio  # noqa: E402
from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord  # noqa: E402

# Learning rate per condition (higher = slower). The structure the claim
# predicts: role-blind FedAvg is WORSE than isolated; centralised is best.
RATES = {
    "isolated": 260.0,
    "fedbuff_uniform": 150.0,
    "fedbuff_similarity": 95.0,
    "fedavg_naive": 330.0,
    "centralised": 70.0,
}
# Smaller, data-poorer devices gain more from federation -- the prediction the
# grouped-bar figure exists to test.
DEVICE_HARDNESS = {"iter_like": 1.0, "sparc_like": 1.15, "diiid_like": 1.3, "tcv_like": 1.6}


def make_run(cond, device, seed, n_shots, rng):
    rate = RATES[cond] * DEVICE_HARDNESS[device]
    # Federation helps the small devices disproportionately.
    if cond in ("fedbuff_uniform", "fedbuff_similarity"):
        rate /= DEVICE_HARDNESS[device] ** 0.8
    viol_tau = rate * (0.35 if cond in ("fedbuff_uniform", "fedbuff_similarity") else 0.9)
    run = RunLog(cond, device, seed)
    for i in range(n_shots):
        reward = -12.0 * np.exp(-i / rate) + rng.normal(0, 0.45)
        p_viol = 0.45 * np.exp(-i / viol_tau)
        run.add(ShotRecord(
            shot=i, reward=float(reward), steps=30,
            violations=("beta_N",) if rng.random() < p_viol else (),
            beta_error=float(abs(reward) * 0.05),
        ))
    return run


def stamp(fig):
    fig.text(0.995, 0.005, "SYNTHETIC — rendering test, not a result",
             ha="right", va="bottom", fontsize=7, color="#B00020", alpha=0.85)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures_demo")
    ap.add_argument("--shots", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    log = ExperimentLog("synthetic-demo")
    devices = [d.name for d in all_devices()]
    for cond in CONDITION_ORDER:
        for dev in devices:
            for seed in range(args.seeds):
                log.add(make_run(cond, dev, seed, args.shots, rng))

    print(log.coverage())
    print()

    iso = log.select("isolated", "iter_like")
    threshold = -1.5
    res = speedup_ratio(iso, log.select("fedbuff_similarity", "iter_like"), threshold)
    print(f"synthetic speedup on iter_like: {res.summary()}")
    print()

    figs = {
        "fig1_learning_curves": plots.plot_learning_curves(log, "iter_like"),
        "fig2_shots_to_threshold": plots.plot_shots_to_threshold(log, threshold),
        "fig3_violations": plots.plot_violations(log, "iter_like"),
    }

    # Figure 4: the catastrophe plot. Isolated trajectories overshoot the beta
    # limit; federated ones approach and turn away.
    limit, soft = 3.0, 2.5
    iso_tr, fed_tr = [], []
    for k in range(6):
        t = np.linspace(0, 1, 40)
        iso_tr.append(2.0 + 1.4 * t + rng.normal(0, 0.04, 40))
        fed_tr.append(2.0 + 0.42 * np.tanh(4 * t) + rng.normal(0, 0.03, 40))
    figs["fig4_catastrophe"] = plots.plot_catastrophe(
        iso_tr, fed_tr, limit_value=limit, soft_value=soft, quantity="beta_N")

    # Figure 5: benefit vs dimensionless distance, using the REAL device set.
    states = encoded_states()
    target = states["iter_like"]
    dists, benefits, labels = [], [], []
    for name, st in states.items():
        if name == "iter_like":
            continue
        d = similarity_distance(st, target)
        dists.append(d)
        benefits.append(float(2.6 * np.exp(-d / 1.1) + rng.normal(0, 0.06)))
        labels.append(name)
    figs["fig5_similarity_ablation"] = plots.plot_similarity_ablation(
        np.array(dists), np.array(benefits),
        errors=np.full(len(dists), 0.12), labels=labels)

    # Figure 6: cold start on the smallest device.
    figs["fig6_cold_start"] = plots.plot_cold_start(log, "tcv_like", threshold)

    # Supplementary: staleness tolerance.
    lags = np.array([0, 2, 5, 10, 20, 40])
    perf = -0.4 - 0.02 * lags + rng.normal(0, 0.03, lags.size)
    figs["figS1_staleness"] = plots.plot_staleness(lags, perf, np.full(lags.size, 0.05))

    for name, fig in figs.items():
        stamp(fig)
        p = out / f"{name}.png"
        fig.savefig(p, dpi=150)
        print(f"  wrote {p}")

    print()
    print("These are SYNTHETIC. They show what the deliverables look like and")
    print("that the plotting code works -- nothing about the real system.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
