#!/usr/bin/env python
"""Ground-truth recovery: hide a TORAX config, recover it.

This is the credibility experiment. In TORAX you SET the physics, so the truth
is known exactly -- generate trajectories from a known config, hide it, run the
inverse, and see whether the parameters come back. Clean pass/fail, no
ambiguity. Do this before building anything on top of identification.

    uv run python scripts/exp_recover_scalars.py
    uv run python scripts/exp_recover_scalars.py --noise 0.02 --subsample 4

HONESTY KNOBS -- use them
------------------------
Handed full noiseless profiles at every timestep, identification is far easier
than from real diagnostics, and every downstream claim inherits that optimism.
`--noise` and `--subsample` degrade the observations toward something a
diagnostic could actually deliver. Report clean and noisy separately.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, get  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.identification.torax_inverse import (  # noqa: E402
    CANDIDATE_UNKNOWNS,
    RecoveryResult,
    adam,
    build_loss,
    check_gradient_flow,
)
from hfmarl.util.report import record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--params", default="plasma_composition.Z_eff,"
                                        "sources.ei_exchange.Qei_multiplier",
                    help="comma-separated config paths; start with 2 you trust")
    ap.add_argument("--perturb", type=float, default=0.35,
                    help="how far the hidden truth sits from nominal")
    ap.add_argument("--start-offset", type=float, default=-0.2,
                    help="how wrong the initial guess is, relative")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="relative Gaussian noise on the observations")
    ap.add_argument("--subsample", type=int, default=1,
                    help="keep every Nth radial point (diagnostic sparsity)")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    names = [p.strip() for p in args.params.split(",") if p.strip()]
    unknown = [n for n in names if n not in CANDIDATE_UNKNOWNS]
    if unknown:
        say(f"unknown parameter path(s): {unknown}")
        say(f"known: {list(CANDIDATE_UNKNOWNS)}")
        return 1

    rng = np.random.default_rng(args.seed)
    nominal = np.array([CANDIDATE_UNKNOWNS[n][0] for n in names])
    lo = np.array([CANDIDATE_UNKNOWNS[n][1] for n in names])
    hi = np.array([CANDIDATE_UNKNOWNS[n][2] for n in names])

    # The hidden truth. Perturbed away from nominal so recovering "the default"
    # is not mistaken for recovering the answer.
    truth = np.clip(nominal * (1.0 + args.perturb * rng.uniform(-1, 1, len(names))),
                    lo, hi)
    start = np.clip(truth * (1.0 + args.start_offset), lo, hi)

    device = get(args.device)
    cfg = build_config(device, t_final=args.delta_t * (args.steps + 2))

    say(f"device      : {device.name}")
    say(f"parameters  : {names}")
    say(f"hidden truth: {np.array2string(truth, precision=4)}")
    say(f"start guess : {np.array2string(start, precision=4)}")
    say(f"observations: {args.steps} steps, noise {args.noise:.1%}, "
        f"every {args.subsample} radial point(s)")
    say()

    # --- generate the observations from the hidden truth ----------------
    _, simulate = build_loss(cfg, names, {"T_e": np.zeros((args.steps, 25))},
                             args.delta_t, args.steps)
    obs = {k: np.asarray(v, dtype=float) for k, v in simulate(truth).items()}
    for k in ("T_e", "T_i", "n_e"):
        if k not in obs:
            continue
    n_rho_full = next(iter(obs.values())).shape[1]
    rho_indices = np.arange(0, n_rho_full, args.subsample)
    obs = {k: v[:, rho_indices] for k, v in obs.items()}
    if args.noise > 0:
        obs = {k: v * (1 + rng.normal(0, args.noise, v.shape)) for k, v in obs.items()}

    # `simulate` returns the full radial grid, so the loss is told which cells
    # the observations correspond to. Slicing inside `build_loss` is what makes
    # --subsample work at all.
    loss_and_grad_full, _ = build_loss(cfg, names, obs, args.delta_t, args.steps,
                                       rho_indices=rho_indices)

    say("gradient flow at the starting point:")
    flow = check_gradient_flow(loss_and_grad_full, start, names)
    for n, g in flow.items():
        say(f"   {'OK  ' if g > 1e-12 else 'DEAD'} |dL/d({n})| = {g:.4e}")
    if all(g <= 1e-12 for g in flow.values()):
        say("All gradients are zero. Nothing can be recovered; run "
            "scripts/gate_identify.py to find out whether this is "
            "unidentifiability or a tracing bug.")
        return 1
    say()

    say("optimising:")
    est, hist = adam(loss_and_grad_full, start, iterations=args.iterations,
                     lr=args.lr, bounds=(lo, hi), verbose=True)
    for h in hist:
        lines.append(f"    it {h.get('iteration')}  loss {h.get('loss', float('nan')):.6e}")
    say()

    res = RecoveryResult(names=names, truth=truth, estimate=est,
                         history=hist, gradient_flow=flow)
    for ln in res.summary(args.tolerance).splitlines():
        say(ln)

    passed = float(np.max(res.relative_error)) < args.tolerance
    say()
    if passed:
        say("PASS: TORAX parameters recovered from trajectories alone.")
        say("This is the result that makes everything built on identification")
        say("credible. Re-run with --noise 0.02 --subsample 4 before believing it")
        say("applies to anything diagnostic-like.")
    else:
        say("FAIL. Before tuning the optimiser, check in this order:")
        say("  1. scripts/gate_identify.py -- is the parameter identifiable at all?")
        say("  2. is any gradient DEAD above?")
        say("  3. are two parameters degenerate? (CRLB correlation)")
        say("  4. only then: more iterations, different lr")

    if not args.no_record:
        p = record(f"Scalar recovery -- {device.name} "
                   f"({'PASSED' if passed else 'FAILED'})",
                   "```\n" + "\n".join(lines) + "\n```")
        print(f"\n(recorded to {p})")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
