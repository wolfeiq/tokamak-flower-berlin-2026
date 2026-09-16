#!/usr/bin/env python
"""Derive each device's operating point from TORAX instead of asserting it.

    uv run python scripts/measure_operating_points.py --task easy

RUNBOOK 5 has asked for this since the beginning and it had never been done,
because `hfmarl/physics/torax_adapter.py` -- the module `dimensionless.py`
names as the way to encode a simulation state -- did not exist. So
`OPERATING_POINTS` stayed a hand-written table of nominal kinetics, and every
aggregation weight in the project was a Gaussian kernel evaluated at
coordinates no simulation had produced.

WHY A SWEEP AND NOT ONE SHOT
----------------------------
A device does not HAVE an operating point; it has a region it moves through,
and which part of that region depends on what the controller commands. So each
device is driven at several command levels for a whole shot and the state is
encoded at every step after a settling window. Reported: the trajectory mean,
and the spread across the sweep.

The spread is the number that matters for federation. If a device's own
excursion in (rho*, nu*, beta_N, q95) is comparable to the DISTANCE between
devices, then weighting peers by a single static coordinate each is measuring
noise, and SPEC.md 4b needs the distribution rather than the point.

rho* and nu* are averaged in the log, because `similarity_distance` compares
them in the log; averaging linearly here and comparing logarithmically there
would be two metrics wearing one name.
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

from hfmarl.devices.registry import (  # noqa: E402
    DEVICES,
    encoded_states,
    get as get_device,
)
from hfmarl.envs.torax_env import ToraxDeviceEnv  # noqa: E402
from hfmarl.federation.similarity import similarity_distance  # noqa: E402
from hfmarl.physics.torax_adapter import mean_state, state_from_env  # noqa: E402
from hfmarl.util.report import record  # noqa: E402

LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def measure(device_name: str, task: str, settle_fraction: float = 0.4,
            levels=LEVELS):
    """Encoded states visited across a command sweep, one list per level."""
    per_level = {}
    for lv in levels:
        env = ToraxDeviceEnv(get_device(device_name), task=task,
                             strict_task_check=False)
        env.reset()
        states = []
        n = env.task.steps_per_shot
        settle = int(n * settle_fraction)
        for i in range(n):
            _, _, terminated, truncated, _ = env.step(
                np.full(env.n_actions, lv))
            if terminated or truncated:
                break
            if i >= settle:
                # After settling only: the first steps are the plant leaving
                # its initial condition, which is a transient the device does
                # not operate in and which no federation partner shares.
                try:
                    states.append(state_from_env(env))
                except Exception:
                    pass
        per_level[lv] = states
    return per_level


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="easy")
    ap.add_argument("--devices", nargs="*", default=sorted(DEVICES))
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"task    : {args.task}")
    say(f"levels  : {list(LEVELS)}  (fraction of every actuator's envelope)")
    say("states encoded per step after a 40% settling window")
    say()

    nominal = encoded_states()
    measured: dict[str, dict] = {}
    excursions: dict[str, float] = {}
    t0 = time.perf_counter()

    for name in args.devices:
        per_level = measure(name, args.task)
        flat = [s for v in per_level.values() for s in v]
        if not flat:
            say(f"{name}: no usable states")
            continue
        m = mean_state(flat)
        # The device's OWN excursion: how far it moves in similarity space
        # between the weakest and strongest command it can be given. If that
        # is comparable to the distance between devices, a single static
        # coordinate per device cannot carry SPEC.md 4b.
        # The extreme levels that actually COMPLETED a shot. diiid_like and
        # tcv_like violate on step 1 at full command, so their strongest
        # usable level is not the strongest available one -- taking the
        # nominal extremes would report nan and hide the excursion entirely.
        usable = [lv for lv in LEVELS if per_level[lv]]
        excursions[name] = (
            similarity_distance(mean_state(per_level[usable[0]]),
                                mean_state(per_level[usable[-1]]))
            if len(usable) >= 2 else float("nan"))
        measured[name] = {
            "rho_star": m.rho_star, "nu_star": m.nu_star,
            "beta_N": m.beta_N, "q95": m.q95, "mach": m.mach,
        }

        def rng(attr):
            v = np.array([getattr(s, attr) for s in flat], float)
            v = v[np.isfinite(v)]
            return (float(v.min()), float(v.max())) if v.size else (np.nan, np.nan)

        t = nominal[name]
        say(f"{name}")
        say(f"  measured  rho* {m.rho_star:.3e}  nu* {m.nu_star:.4f}  "
            f"beta_N {m.beta_N:6.3f}  q95 {m.q95:5.2f}")
        say(f"  nominal   rho* {t.rho_star:.3e}  nu* {t.nu_star:.4f}  "
            f"beta_N {t.beta_N:6.3f}  q95 {t.q95:5.2f}")
        say(f"  ratio     rho* {m.rho_star / t.rho_star:6.2f}   "
            f"nu* {m.nu_star / t.nu_star:8.2f}   "
            f"beta_N {m.beta_N / t.beta_N:6.2f}")
        for attr in ("rho_star", "nu_star", "beta_N"):
            lo, hi = rng(attr)
            say(f"  swept     {attr:9s} {lo:.4g} .. {hi:.4g}"
                + (f"   ({hi / lo:.1f}x)" if lo > 0 else ""))
        say()

    # --- is a single point per device defensible at all? -----------------
    say("=" * 74)
    say("Is one static coordinate per device enough?")
    say()
    names = [n for n in args.devices if n in measured]
    from hfmarl.physics.dimensionless import DimensionlessState

    states = {n: DimensionlessState(**measured[n]) for n in names}
    say(f"{'device':13s}{'own excursion':>16s}{'nearest peer':>16s}  verdict")
    for n in names:
        others = [similarity_distance(states[n], states[o])
                  for o in names if o != n]
        nearest = min(others) if others else float("nan")
        exc = excursions.get(n, float("nan"))
        verdict = ("a point is defensible" if exc < 0.5 * nearest
                   else "A POINT IS NOT ENOUGH")
        say(f"{n:13s}{exc:>16.3f}{nearest:>16.3f}  {verdict}")
    say()
    say("If a device's own excursion is comparable to its distance from the")
    say("nearest peer, a single point per device cannot support SPEC.md 4b and")
    say("the weighting needs the distribution of visited states instead.")
    say()

    say("MEASURED_OPERATING_POINTS = " + json.dumps(measured, indent=4))
    say(f"total {time.perf_counter() - t0:.0f}s")

    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "operating_points"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}.json").write_text(json.dumps(measured, indent=2),
                                           encoding="utf-8")
    say(f"wrote {out / f'{args.task}.json'}")

    if not args.no_record:
        path = record(f"Operating points measured from TORAX -- {args.task}",
                      "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {path})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
