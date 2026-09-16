#!/usr/bin/env python
"""Gate 0c -- CPU vs GPU, float64 vs float32. Where should Phase 5 run?

DO NOT ASSUME THE GPU WINS. Two reasons it may not:

  1. TORAX runs in **float64 by default**. Consumer NVIDIA cards (GeForce,
     and most RTX workstation parts) execute f64 at roughly 1/32 of their f32
     rate. A datacentre card (A100, H100) does not have this penalty.
  2. This is a **1-D problem on ~25 radial cells**. There is very little
     arithmetic per kernel, so per-kernel launch overhead can dominate and the
     GPU can lose outright to a CPU.

The real benefit of the WSL box for this project is more likely RAM and core
count -- letting Phase 5 run several federation clients concurrently -- than
raw per-step speed. This script measures rather than assumes.

Each configuration runs in a SUBPROCESS, because JAX reads its platform and
precision settings once at import and they cannot be changed afterwards.

    uv run python scripts/gate0_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _worker() -> int:
    """Run inside the subprocess: time a few steps and print JSON."""
    import numpy as np

    from hfmarl.devices.registry import get
    from hfmarl.envs.actuators import bank_from_device
    from hfmarl.envs.torax_config import build_config
    from hfmarl.envs.torax_env import ToraxCore

    device_name = os.environ["BENCH_DEVICE"]
    steps = int(os.environ["BENCH_STEPS"])
    delta_t = float(os.environ["BENCH_DT"])

    import jax

    device = get(device_name)
    cfg = build_config(device, t_final=delta_t * (steps + 2))
    core = ToraxCore(cfg, delta_t)
    bank = bank_from_device(device, ("thermal",))

    t0 = time.perf_counter()
    core.start()
    core.reset()
    build_s = time.perf_counter() - t0

    times = []
    for i in range(steps):
        vals = bank.from_unit_actions(np.full(len(bank), (i % 5) / 4.0 * 2 - 1))
        core.apply_updates(bank.build_updates(vals, core.t, delta_t))
        t0 = time.perf_counter()
        ok, done, err = core.advance()
        times.append(time.perf_counter() - t0)
        if not ok:
            print(json.dumps({"error": err}))
            return 1
        if done:
            break

    rest = times[1:] or times
    print(json.dumps({
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.read("jax_enable_x64")),
        "build_s": build_s,
        "first_s": times[0],
        "median_s": float(sorted(rest)[len(rest) // 2]),
    }))
    return 0


def run_case(name: str, env_extra: dict[str, str], args) -> dict | None:
    env = dict(os.environ)
    env.update(env_extra)
    env.update({
        "BENCH_DEVICE": args.device,
        "BENCH_STEPS": str(args.steps),
        "BENCH_DT": str(args.delta_t),
        "HFMARL_BENCH_WORKER": "1",
    })
    print(f"  running {name} ...", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env, capture_output=True, text=True, timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {args.timeout}s")
        return None
    line = next(
        (ln for ln in reversed(r.stdout.splitlines()) if ln.strip().startswith("{")),
        None,
    )
    if line is None:
        print(f"    FAILED (rc={r.returncode})")
        tail = (r.stderr or r.stdout).strip().splitlines()[-4:]
        for t in tail:
            print(f"    | {t}")
        return None
    out = json.loads(line)
    if "error" in out:
        print(f"    simulation error: {out['error']}")
        return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    from hfmarl.util.report import record

    cases = {
        "CPU  f64 (TORAX default)": {"JAX_PLATFORMS": "cpu"},
        "CPU  f32": {"JAX_PLATFORMS": "cpu", "JAX_PRECISION": "f32"},
        "GPU  f64 (TORAX default)": {"JAX_PLATFORMS": "cuda"},
        "GPU  f32": {"JAX_PLATFORMS": "cuda", "JAX_PRECISION": "f32"},
    }

    print(f"benchmarking {args.device}, {args.steps} steps of {args.delta_t}s\n")
    results = {name: run_case(name, env, args) for name, env in cases.items()}

    lines = [
        f"{'configuration':28s}{'build s':>10s}{'step1 s':>10s}{'median s':>11s}  backend",
    ]
    for name, r in results.items():
        if r is None:
            lines.append(f"{name:28s}{'--':>10s}{'--':>10s}{'--':>11s}  unavailable")
        else:
            lines.append(
                f"{name:28s}{r['build_s']:10.2f}{r['first_s']:10.2f}"
                f"{r['median_s']:11.4f}  {r['backend']}"
            )
    ok = {k: v for k, v in results.items() if v}
    body = "\n".join(lines)
    if ok:
        best = min(ok, key=lambda k: ok[k]["median_s"])
        body += f"\n\nfastest steady-state: {best} ({ok[best]['median_s']:.4f} s/step)"
        cpu = next((v for k, v in ok.items() if k.startswith("CPU  f64")), None)
        gpu = next((v for k, v in ok.items() if k.startswith("GPU  f64")), None)
        if cpu and gpu:
            ratio = cpu["median_s"] / max(gpu["median_s"], 1e-9)
            body += f"\nGPU/CPU speedup at f64: {ratio:.2f}x"
            if ratio < 1.2:
                body += (
                    "\n  -> The GPU is not meaningfully faster. Expected for a 1-D "
                    "problem at f64. Run Phase 5 as PARALLEL CPU CLIENTS rather "
                    "than one GPU job; that is where the WSL box's cores and RAM "
                    "pay off."
                )
            else:
                body += "\n  -> GPU is worth using for single-client throughput."
    else:
        body += "\n\nNo configuration completed. See SETUP.md."

    print("\n" + body)
    if not args.no_record:
        p = record("Gate 0c -- CPU vs GPU benchmark", "```\n" + body + "\n```")
        print(f"\n(recorded to {p})")
    return 0 if ok else 1


if __name__ == "__main__":
    if os.environ.get("HFMARL_BENCH_WORKER"):
        sys.exit(_worker())
    sys.exit(main())
