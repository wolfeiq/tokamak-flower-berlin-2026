#!/usr/bin/env python
"""What difference could N seeds actually detect?

    uv run python scripts/analyse_power.py [results/coldstart]

Run this BEFORE reading a fold, not after.

Worth knowing BEFORE the folds land, because it decides what a null result
means. If six seeds cannot resolve a difference smaller than the whole budget,
then "all five arms are indistinguishable" says nothing about federation and
everything about the experiment.

Uses the per-seed shots-to-competence from the saved three-seed folds to
estimate the seed-to-seed spread, then asks what separation a two-sample
comparison at n=6 could see.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.metrics.curves import shots_to_competence_evaluated
from hfmarl.metrics.log import RunLog

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/coldstart")

rows = []
for d in sorted(root.glob("*")):
    if not (d / "runs.json").exists():
        continue
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((d / "runs.json").read_text(encoding="utf-8"))
    tols = man.get("tolerances", {})
    if not tols:
        continue
    for key, raw in payload.items():
        joiner, _, arm = key.rpartition(":")
        if not joiner or joiner not in tols:
            continue
        logs = [RunLog.from_dict(r) for r in raw]
        per_seed = []
        for lg in logs:
            r = shots_to_competence_evaluated([lg], float(tols[joiner]))
            per_seed.append(r.values[0] if r.n_reached else np.nan)
        rows.append((d.name, joiner, arm, np.array(per_seed, float),
                     man.get("join")))

print(f"{'fold':26s}{'joiner':11s}{'arm':22s}{'per-seed shots':>22s}"
      f"{'sd':>8s}")
spreads = []
for name, joiner, arm, vals, budget in rows:
    good = vals[np.isfinite(vals)]
    sd = float(np.std(good, ddof=1)) if good.size > 1 else float("nan")
    if np.isfinite(sd):
        spreads.append(sd)
    shown = ",".join("never" if not np.isfinite(v) else f"{v:.0f}"
                     for v in vals)
    print(f"{name[:25]:26s}{joiner:11s}{arm:22s}{shown:>22s}{sd:>8.1f}")

s = np.array(spreads, float)
print()
print(f"pooled sd over {s.size} (fold, arm) cells: median {np.median(s):.1f}, "
      f"range {s.min():.1f}-{s.max():.1f} shots")

# Minimum detectable difference, two-sample, equal n, alpha .05 two-sided,
# power .80: delta = (z_a/2 + z_b) * sd * sqrt(2/n) = 2.80 * sd * sqrt(2/n).
# Normal approximation, so this is optimistic at n=6 -- a t-test needs more.
for n in (3, 6, 12, 24):
    for sd in (np.median(s), s.max()):
        delta = 2.80 * sd * np.sqrt(2.0 / n)
        tag = "median sd" if sd == np.median(s) else "worst sd "
        print(f"  n={n:3d}  {tag}  minimum detectable difference "
              f"{delta:6.1f} shots  ({delta / 120:.0%} of a 120-shot budget)")
