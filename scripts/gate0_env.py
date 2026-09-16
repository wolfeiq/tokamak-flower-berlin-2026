#!/usr/bin/env python
"""Gate 0a -- does TORAX run here at all?

Pass criteria:
  1. TORAX imports and reports a version.
  2. A config builds and validates for every device in the registry.
  3. One simulation initialises and takes a few steps without error.
  4. Every scalar the limits and observations depend on actually exists in
     PostProcessedOutputs. A missing safety quantity must be a loud failure
     here, not a silent zero during training.

Run this before anything else:  uv run python scripts/gate0_env.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.envs.torax_env import (  # noqa: E402
    SCALAR_FIELDS,
    ToraxCore,
    ToraxUnavailableError,
)
from hfmarl.util.report import peak_rss_mb, record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--transport", default="constant")
    ap.add_argument("--solver", default="linear")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    # --- 1. import -----------------------------------------------------
    try:
        import jax
        import torax
    except ImportError as e:
        print(f"FAIL: cannot import ({e}). See SETUP.md.")
        return 1

    say(f"torax version : {getattr(torax, '__version__', 'unknown')}")
    say(f"jax version   : {jax.__version__}")
    say(f"jax devices   : {jax.devices()}")
    # jax.config.read has moved before. This is the first thing anyone runs, so
    # an accessor change must not be what greets them.
    try:
        x64 = jax.config.read("jax_enable_x64")
    except Exception as e:
        x64 = f"unreadable ({type(e).__name__})"
    say(f"x64 enabled   : {x64}")
    say()

    if getattr(torax, "__version__", "").split(".")[:2] != ["1", "4"]:
        say(
            "WARNING: this repo targets TORAX 1.4.3. Other versions changed the "
            "transport config schema and will likely fail validation below."
        )

    # --- 2. every device config validates ------------------------------
    say("config validation:")
    bad = 0
    for name in sorted(DEVICES):
        try:
            cfg = build_config(
                get(name), transport_model=args.transport, solver_type=args.solver
            )
            torax.ToraxConfig.from_dict(cfg)
            say(f"  {name:14s} OK")
        except Exception as e:
            bad += 1
            say(f"  {name:14s} FAIL: {type(e).__name__}: {e}")
    say()
    if bad:
        say(f"FAIL: {bad} device config(s) invalid. Fix before proceeding.")
        if not args.no_record:
            record("Gate 0a -- environment (FAILED)", "```\n" + "\n".join(lines) + "\n```")
        return 1

    # --- 3. initialise and step ----------------------------------------
    device = get(args.device)
    cfg = build_config(
        device, t_final=args.delta_t * (args.steps + 1),
        transport_model=args.transport, solver_type=args.solver,
    )
    core = ToraxCore(cfg, args.delta_t)

    say(f"running {args.device}, {args.steps} steps of {args.delta_t}s")
    t0 = time.perf_counter()
    try:
        core.start()
    except ToraxUnavailableError as e:
        say(f"FAIL: {e}")
        return 1
    t_build = time.perf_counter() - t0
    say(f"  build + initial state : {t_build:8.2f} s")

    core.reset()
    for i in range(args.steps):
        t0 = time.perf_counter()
        ok, done, err = core.advance()
        dt = time.perf_counter() - t0
        flag = "ok" if ok else f"FAILED ({err})"
        say(f"  step {i + 1:2d}               : {dt:8.2f} s   {flag}")
        if not ok:
            say("FAIL: simulation error. See the TORAX message above.")
            if not args.no_record:
                record("Gate 0a -- environment (FAILED)", "```\n" + "\n".join(lines) + "\n```")
            return 1
        if done:
            break
    say(f"  peak RSS              : {peak_rss_mb():8.0f} MB")
    say()

    # --- 4. required scalars all present -------------------------------
    scalars = core.read_scalars()
    missing = [f for f in SCALAR_FIELDS if f not in scalars]
    say(f"scalars found : {len(scalars)}/{len(SCALAR_FIELDS)}")
    for k in ("beta_N", "q95", "fgw_n_e_line_avg"):
        mark = "OK " if k in scalars else "MISSING"
        val = f"{scalars[k]:.4f}" if k in scalars else "--"
        say(f"  {mark} {k:22s} {val}   <- a limit depends on this")
    if missing:
        say(f"  absent: {missing}")
    say()

    profiles = core.read_profiles()
    say("profiles: " + ", ".join(f"{k}{list(v.shape)}" for k, v in profiles.items()))
    say()

    critical = [k for k in ("beta_N", "q95", "fgw_n_e_line_avg") if k not in scalars]
    if critical:
        say(f"FAIL: limits depend on {critical}, which TORAX did not report.")
        status = "FAILED"
        rc = 1
    else:
        say("PASS: TORAX runs and reports every quantity the limits need.")
        say("Next: scripts/gate0_jit.py")
        status = "PASSED"
        rc = 0

    if not args.no_record:
        p = record(f"Gate 0a -- environment ({status})", "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
