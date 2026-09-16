#!/usr/bin/env python
"""Gate 0b -- THE gate. Does JIT survive per-step actuator changes?

Everything downstream assumes TORAX can be driven like a Gym environment:
change a heating power, advance, repeat, without paying JAX compilation on
every step. If that is false the project as specified does not work and the
control formulation has to change before any RL code is written.

WHAT IS MEASURED
----------------
N steps, each changing the heating power, each timed separately.

    step 1        pays compilation (expect seconds to minutes)
    steps 2..N    should be fast and near-identical to each other

PASS requires both:
    * median(steps 2..N) < `--ratio` x step 1          (compilation amortised)
    * max/min across steps 2..N < `--spread`           (no sporadic recompiles)

THE NEGATIVE CONTROL -- run it
------------------------------
A test that cannot fail proves nothing. With `--negative-control` the script
also runs a deliberately WRONG version that grows the waveform array by one
breakpoint each step, changing a JAX leaf shape every time. That *must*
recompile, and must therefore be dramatically slower.

    If the negative control is NOT slower, this script is not measuring
    compilation at all and the PASS above is meaningless.

Report both numbers. A pass with no working negative control is not a pass.

    uv run python scripts/gate0_jit.py --negative-control
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.actuators import bank_from_device  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.envs.torax_env import ToraxCore  # noqa: E402
from hfmarl.util.report import peak_rss_mb, record  # noqa: E402


def timed_steps(core, bank, powers, delta_t, grow_breakpoints=False):
    """Run one step per entry in `powers`, returning per-step wall times."""
    import numpy as np
    from torax.experimental import TimeVaryingScalarUpdate

    times: list[float] = []
    core.reset()
    bank.reset()
    for i, frac in enumerate(powers):
        values = bank.from_unit_actions(np.full(len(bank), frac, dtype=float))

        if grow_breakpoints:
            # THE WRONG WAY, on purpose: an array that gets longer every step.
            # Each new shape is a new JAX signature and forces a recompile.
            #
            # Note this touches ONE config path while the correct path updates
            # all of them, so the control does strictly less work. That makes
            # the comparison conservative: if it is still dramatically slower,
            # the slowdown is compilation and not extra updates.
            n = i + 2
            t_arr = np.linspace(core.t, core.t + delta_t, n)
            path = bank.specs[0].torax_path
            v_arr = np.linspace(bank.prev_values[0], values[0], n)
            updates = {
                path: TimeVaryingScalarUpdate(
                    time=t_arr.astype(np.float64), value=v_arr.astype(np.float64)
                )
            }
            bank.prev_values[0] = values[0]
        else:
            # The correct way: exactly two breakpoints, every time.
            updates = bank.build_updates(values, core.t, delta_t)

        core.apply_updates(updates)
        t0 = time.perf_counter()
        ok, done, err = core.advance()
        times.append(time.perf_counter() - t0)
        if not ok:
            print(f"  step {i + 1} failed: {err}")
            break
        if done:
            break
    return times


def summarise(times, label, lines):
    def say(s=""):
        print(s)
        lines.append(s)

    if len(times) < 2:
        say(f"{label}: too few steps ({len(times)}) to judge")
        return None
    first, rest = times[0], times[1:]
    med = statistics.median(rest)
    spread = max(rest) / max(min(rest), 1e-9)
    say(f"{label}")
    say(f"  step 1 (compile)      : {first:9.3f} s")
    say(f"  steps 2..{len(times):<3d} median   : {med:9.3f} s")
    say(f"  steps 2..{len(times):<3d} min/max  : {min(rest):9.3f} / {max(rest):.3f} s")
    say(f"  ratio median/step1    : {med / max(first, 1e-9):9.4f}")
    say(f"  spread max/min        : {spread:9.2f}")
    say()
    return {"first": first, "median": med, "spread": spread, "rest": rest}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--transport", default="constant")
    ap.add_argument("--solver", default="linear")
    ap.add_argument("--ratio", type=float, default=0.10,
                    help="median(2..N) must be below this fraction of step 1")
    ap.add_argument("--spread", type=float, default=3.0,
                    help="max/min across steps 2..N must be below this")
    ap.add_argument("--negative-control", action="store_true",
                    help="also run the deliberately-wrong growing-array version")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s)
        lines.append(s)

    device = get(args.device)
    cfg = build_config(
        device,
        t_final=args.delta_t * (args.steps + 2),
        transport_model=args.transport,
        solver_type=args.solver,
    )
    bank = bank_from_device(device, ("thermal",))
    say(f"device {device.name}, actuators {bank.names}")
    say(f"delta_t_a = {args.delta_t}s, transport = {args.transport}, solver = {args.solver}")
    say()

    # Vary the command every step; a constant command could be optimised away
    # and would not exercise the parameter path at all.
    powers = [(-1.0 + 2.0 * (i % 5) / 4.0) for i in range(args.steps)]

    core = ToraxCore(cfg, args.delta_t)
    core.start()
    good = summarise(
        timed_steps(core, bank, powers, args.delta_t), "CORRECT (2 breakpoints)", lines
    )

    bad = None
    if args.negative_control:
        say("running negative control -- this SHOULD be slow")
        core2 = ToraxCore(cfg, args.delta_t)
        core2.start()
        bad = summarise(
            timed_steps(core2, bank, powers, args.delta_t, grow_breakpoints=True),
            "NEGATIVE CONTROL (growing array, expected to recompile)",
            lines,
        )

    say(f"peak RSS: {peak_rss_mb():.0f} MB")
    say()

    if good is None:
        say("INCONCLUSIVE: not enough successful steps.")
        status, rc = "INCONCLUSIVE", 1
    else:
        ok_ratio = good["median"] < args.ratio * good["first"]
        ok_spread = good["spread"] < args.spread
        say(f"ratio  < {args.ratio}: {'PASS' if ok_ratio else 'FAIL'}")
        say(f"spread < {args.spread}: {'PASS' if ok_spread else 'FAIL'}")

        if bad is not None:
            sensitivity = bad["median"] / max(good["median"], 1e-9)
            say(f"negative control is {sensitivity:.1f}x slower than correct path")
            if sensitivity < 2.0:
                say(
                    "  WARNING: the negative control was NOT meaningfully slower. "
                    "This script may not be measuring compilation at all, and the "
                    "verdict above cannot be trusted. Investigate before proceeding."
                )
        else:
            say("negative control NOT RUN -- rerun with --negative-control to "
                "confirm this test can actually fail.")

        if ok_ratio and ok_spread:
            say()
            say("PASS: JIT survives per-step actuator changes. The Gym wrapper "
                "approach in SPEC.md is viable. Proceed to Phase 1.")
            status, rc = "PASSED", 0
        else:
            say()
            say("FAIL: every step appears to pay compilation.")
            say("Before abandoning the approach, check:")
            say("  * are all updates exactly two breakpoints? (envs/actuators.py)")
            say("  * is any action hitting a JAX_STATIC path? (STATIC_EXACT)")
            say("  * is delta_t_a an integer multiple of numerics.fixed_dt?")
            say("Fallback if genuinely unfixable: episode-level control -- fix a "
                "whole actuator trajectory, run one simulation, treat the episode "
                "as a single action. That changes the RL formulation and must be "
                "decided before Phase 1, not worked around later.")
            status, rc = "FAILED", 1

    if not args.no_record:
        p = record(f"Gate 0b -- JIT survival ({status})", "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
