#!/usr/bin/env python
"""TORAX visualisations: profiles, time traces, and an actuator-response check.

Three things worth looking at, in increasing order of how much they tell you:

  1. profiles      -- T_i, T_e, n_e, q vs rho at several times. Confirms the
                      plasma looks like a plasma.
  2. traces        -- beta_N, q95, Greenwald fraction and H98 vs time, with the
                      operating limits drawn on. Shows whether the envelope is
                      even reachable on this device.
  3. step response -- the same device driven with a heating step, so you can
                      SEE the thermal lag the controller has to anticipate. If
                      the response is instantaneous the control problem is
                      trivial and the headroom gate will say so; if it never
                      responds, the actuator is not connected.

TORAX also ships its own interactive plotter, `torax.experimental
.create_plotly_figure`, which is better for exploring a single run. This script
is for the comparisons across devices that the project actually needs.

    uv run python scripts/plot_torax.py --device iter_like
    uv run python scripts/plot_torax.py --all-devices
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import DEVICES, all_devices, get  # noqa: E402
from hfmarl.envs.actuators import bank_from_device  # noqa: E402
from hfmarl.envs.limits import DEFAULT_LIMITS  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.envs.torax_env import ToraxCore  # noqa: E402

OUT = Path("results/torax_plots")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_with_step(device, steps, delta_t, step_at, level_before, level_after,
                  transport):
    """Drive one device with a heating step; collect profiles and scalars."""
    cfg = build_config(device, t_final=delta_t * (steps + 2),
                       transport_model=transport)
    core = ToraxCore(cfg, delta_t)
    core.start()
    core.reset()
    bank = bank_from_device(device, ("thermal",))

    rec = {"t": [], "scalars": [], "profiles": [], "power": []}
    for i in range(steps):
        level = level_before if i < step_at else level_after
        vals = bank.from_unit_actions(np.full(len(bank), level))
        core.apply_updates(bank.build_updates(vals, core.t, delta_t))
        ok, done, err = core.advance()
        if not ok:
            print(f"  step {i} failed: {err}")
            break
        rec["t"].append(core.t)
        rec["scalars"].append(core.read_scalars())
        rec["profiles"].append(core.read_profiles())
        rec["power"].append(float(vals.sum()))
        if done:
            break
    return rec


def plot_profiles(rec, device, plt):
    prof = rec["profiles"]
    if not prof:
        return None
    keys = [k for k in ("T_i", "T_e", "n_e", "q_face") if k in prof[0]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.4))
    axes = np.atleast_1d(axes)
    picks = np.linspace(0, len(prof) - 1, min(5, len(prof))).astype(int)
    cmap = plt.get_cmap("viridis")
    for ax, k in zip(axes, keys):
        for j, i in enumerate(picks):
            y = prof[i][k]
            ax.plot(np.linspace(0, 1, len(y)), y,
                    color=cmap(j / max(len(picks) - 1, 1)),
                    label=f"t={rec['t'][i]:.1f}s")
        ax.set_xlabel(r"$\rho$")
        ax.set_title(k, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{device.name} — profiles", fontsize=11)
    fig.tight_layout()
    return fig


def plot_traces(rec, device, plt):
    sc = rec["scalars"]
    if not sc:
        return None
    t = np.array(rec["t"])
    limits = {l.name: l for l in DEFAULT_LIMITS}
    panels = [("beta_N", "beta_N"), ("q95", "q95"),
              ("fgw_n_e_line_avg", "greenwald_fraction"), ("H98", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2))
    for ax, (key, lim_name) in zip(np.atleast_1d(axes), panels):
        y = np.array([s.get(key, np.nan) for s in sc])
        ax.plot(t, y, lw=2, color="#0072B2")
        if lim_name and lim_name in limits:
            lim = limits[lim_name]
            ax.axhline(lim.hard, color="#B00020", lw=1.6)
            ax.axhline(lim.soft, color="#B00020", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlabel("t [s]")
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{device.name} — scalars vs time (red = operating limit)",
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_step_response(rec, device, plt):
    """The control problem, made visible: power in, temperature out."""
    sc, t = rec["scalars"], np.array(rec["t"])
    if not sc:
        return None
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.step(t, np.array(rec["power"]) / 1e6, where="post", lw=2,
             color="#666666", label="total aux power")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("P [MW]", color="#666666")
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    beta = np.array([s.get("beta_N", np.nan) for s in sc])
    ax2.plot(t, beta, lw=2.5, color="#D55E00", label=r"$\beta_N$")
    ax2.set_ylabel(r"$\beta_N$", color="#D55E00")
    ax2.spines[["top"]].set_visible(False)

    ax1.set_title(f"{device.name} — step response: the lag the controller "
                  "must anticipate", fontsize=10)
    ax1.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="iter_like", choices=sorted(DEVICES))
    ap.add_argument("--all-devices", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--delta-t", type=float, default=0.5)
    ap.add_argument("--step-at", type=int, default=8)
    ap.add_argument("--before", type=float, default=-0.6)
    ap.add_argument("--after", type=float, default=0.6)
    ap.add_argument("--transport", default="constant")
    args = ap.parse_args()

    plt = _plt()
    OUT.mkdir(parents=True, exist_ok=True)
    devices = all_devices() if args.all_devices else [get(args.device)]

    for d in devices:
        print(f"running {d.name} ...", flush=True)
        rec = run_with_step(d, args.steps, args.delta_t, args.step_at,
                            args.before, args.after, args.transport)
        if not rec["t"]:
            print(f"  no successful steps for {d.name}; skipping plots")
            continue
        for name, fn in (("profiles", plot_profiles), ("traces", plot_traces),
                         ("step_response", plot_step_response)):
            fig = fn(rec, d, plt)
            if fig is None:
                continue
            p = OUT / f"{d.name}_{name}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  wrote {p}")
    print("\nAlso try TORAX's own interactive plotter for a single run:")
    print("  from torax.experimental import create_plotly_figure")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
