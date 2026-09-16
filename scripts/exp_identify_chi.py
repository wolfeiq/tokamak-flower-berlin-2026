#!/usr/bin/env python
"""chi(rho) identification by classical power balance.

The estimator is an INTEGRAL of the conservation law divided by the measured
gradient. No network, no training loop, ~1% error on clean data.

An inverse-PINN arm exists behind `--pinn` and is kept only as a documented
negative result: it did not beat this estimator anywhere it was tried. See
`docs/pinn_comparison.md`. Do not put it on the critical path.


Two modes:

  --manufactured   (default, NO TORAX NEEDED)
      Method of manufactured solutions: choose chi(rho), build profiles and the
      source that make the conservation law hold exactly, then see whether each
      estimator recovers chi. Verifies the identification MATHS before it ever
      meets simulator output. Runs on any machine in seconds.

  --torax
      Same estimators, fed real TORAX profiles and sources.

WHAT IS BEING TESTED
--------------------
Whether the PINN earns its place. The classical estimator integrates the
balance law and divides by the measured gradient -- no network, no training,
and on clean data it reaches ~0.1%. The PINN should win only where finite
differences break: sparse, noisy diagnostics. If it does not, say so.

    uv run python scripts/exp_identify_chi.py
    uv run python scripts/exp_identify_chi.py --noise 0.03 --points 40
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.identification.closure import (  # noqa: E402
    chi_from_power_balance,
    heat_flux_from_balance,
    manufactured_case,
    relative_error_summary,
    relative_error,
)
from hfmarl.identification.pinn import identify_chi_pinn  # noqa: E402
from hfmarl.util.report import record  # noqa: E402


def bump_chi(rho, base=0.5, height=2.0, centre=0.5, width=0.15):
    """Anomalous transport at mid-radius -- the case the method exists for."""
    return base + height * np.exp(-(((rho - centre) / width) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torax", action="store_true",
                    help="use real TORAX output instead of a manufactured case")
    ap.add_argument("--device", default="iter_like")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--points", type=int, default=60,
                    help="number of radial observation points (diagnostic sparsity)")
    ap.add_argument("--pinn", action="store_true",
                    help="also run the inverse-PINN arm (a negative "
                         "result; see docs/pinn_comparison.md)")
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--physics-weight", type=float, default=3.0)
    ap.add_argument("--no-jax", action="store_true",
                    help="force the slow finite-difference gradient")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    lines: list[str] = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    rng = np.random.default_rng(args.seed)

    if args.torax:
        say("TORAX mode is not wired up yet -- run gate_identify.py first to")
        say("confirm prescribed chi(rho) is supported, then extract profiles and")
        say("sources from a run and feed them to the same two estimators below.")
        say("The estimators themselves are simulator-agnostic by construction.")
        return 2

    # --- manufactured case ---------------------------------------------
    case = manufactured_case(bump_chi, n_rho=201)
    rho_f, n_f, T_f, S_f, chi_f = (case["rho"], case["n_e"], case["T_e"],
                                   case["source"], case["chi_true"])
    idx = np.unique(np.linspace(5, len(rho_f) - 6, args.points).astype(int))
    rho, n_e = rho_f[idx], n_f[idx]
    T_e = T_f[idx] * (1 + rng.normal(0, args.noise, idx.size)) if args.noise else T_f[idx]
    S = S_f[idx] * (1 + rng.normal(0, args.noise, idx.size)) if args.noise else S_f[idx]
    chi_true = chi_f[idx]

    say(f"manufactured case: chi(rho) = {bump_chi.__doc__.splitlines()[0]}")
    say(f"  true chi range   : {chi_true.min():.3f} .. {chi_true.max():.3f}")
    say(f"  observation pts  : {len(rho)}")
    say(f"  noise            : {args.noise:.1%}")
    say()

    # --- estimator 1: classical power balance --------------------------
    t0 = time.perf_counter()
    est = chi_from_power_balance(rho, n_e, T_e, S)
    t_base = time.perf_counter() - t0
    err_base = relative_error(est.chi, chi_true)
    stats = relative_error_summary(est.chi, chi_true)
    say("1. classical power balance (integral form, finite differences)")
    say(f"   mean relative error : {err_base:.4f}")
    # chi is a ratio, so one near-zero denominator can set the mean on its own.
    # A mean far above the median is that, not a systematically bad fit.
    say(f"   median / max        : {stats['median']:.4f} / {stats['max']:.4f}"
        f"   (over {stats['n']} points)")
    say(f"   {est.masked_summary()}")
    say(f"   time                : {t_base:.2f} s")
    say()

    if not args.pinn:
        say("PASS" if err_base < 0.05 else "CHECK: error above 5%")
        say()
        say("The inverse-PINN arm is off by default -- it did not beat this")
        say("estimator anywhere tried. Pass --pinn to reproduce that comparison.")
        if not args.no_record:
            record(f"chi(rho) identification -- manufactured, "
                   f"{args.noise:.0%} noise, {len(rho)} pts",
                   "```\n" + "\n".join(lines) + "\n```")
        return 0

    # --- estimator 2: inverse PINN (opt-in; the losing arm) -------------
    q = heat_flux_from_balance(rho, n_e, T_e, S)[0]
    t0 = time.perf_counter()
    res = identify_chi_pinn(rho, T_e, n_e, rho * q, iterations=args.iterations,
                            lr=args.lr, physics_weight=args.physics_weight,
                            seed=args.seed, use_jax=not args.no_jax)
    t_pinn = time.perf_counter() - t0
    err_pinn = relative_error(res.chi, chi_true)
    say("2. inverse PINN (integral-form residual, smooth T and chi)")
    say(f"   mean relative error : {err_pinn:.4f}")
    say(f"   final loss          : {res.final_loss:.4e} "
        f"(data {res.data_loss:.3e}, physics {res.physics_loss:.3e})")
    say(f"   recovered chi range : {res.chi.min():.3f} .. {res.chi.max():.3f}")
    say(f"   time                : {t_pinn:.1f} s")
    say()

    # --- verdict --------------------------------------------------------
    say(f"{'estimator':34s}{'rel err':>10s}{'time s':>10s}")
    say(f"{'classical power balance':34s}{err_base:>10.4f}{t_base:>10.2f}")
    say(f"{'inverse PINN':34s}{err_pinn:>10.4f}{t_pinn:>10.1f}")
    say()
    if err_pinn < err_base:
        say(f"PINN wins by {err_base / max(err_pinn, 1e-12):.1f}x at "
            f"{args.noise:.0%} noise / {len(rho)} points.")
    else:
        say(f"CLASSICAL WINS ({err_base:.4f} vs {err_pinn:.4f}).")
        say("The network has not earned its place under these conditions.")
        say("Before tuning: is the gradient path JAX or finite differences?")
        say("The NumPy fallback is ~100x slower per step and in practice leaves")
        say("the fit undertrained -- check the header of the run above.")
    say()
    say("Sweep --noise and --points: the PINN's case is sparse, noisy data.")
    say("If it never wins anywhere in that sweep, report that and use the")
    say("classical estimator, which is simpler and has no training loop.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path("results/identification")
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(rho, chi_true, "k-", lw=2.5, label="truth")
        ax.plot(rho, est.chi, "o-", ms=3, lw=1.2, color="#0072B2",
                label=f"power balance ({err_base:.3f})")
        ax.plot(res.rho, res.chi, "-", lw=2, color="#D55E00",
                label=f"inverse PINN ({err_pinn:.3f})")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\chi$  [m$^2$/s]")
        ax.set_title(f"chi(rho) identification — {args.noise:.0%} noise, "
                     f"{len(rho)} points", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / f"chi_noise{int(args.noise * 100):02d}_pts{len(rho)}.png"
        fig.savefig(p, dpi=150)
        say(f"plot -> {p}")

    if not args.no_record:
        record(f"chi(rho) identification -- manufactured, "
               f"{args.noise:.0%} noise, {len(rho)} pts",
               "```\n" + "\n".join(lines) + "\n```")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
