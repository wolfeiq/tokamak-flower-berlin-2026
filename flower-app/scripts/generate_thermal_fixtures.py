#!/usr/bin/env python
"""Run TORAX for each investigation facility and emit profile fixtures.

The companion app solves a REDUCED 1-D transport model by default, because
TORAX cannot be installed everywhere: torax==1.4.3 requires jax>=0.10.0, and
jaxlib publishes no macOS x86_64 wheel past 0.4.38, so Intel Macs cannot run it
at all. Run this wherever TORAX does install (Linux x86_64, Apple Silicon,
Colab) and commit the result; the app prefers these fixtures when present and
reports provenance either way.

    .venv/Scripts/python.exe flower-app/scripts/generate_thermal_fixtures.py --transport qlknn

Writes flower-app/.fusion-state/torax-fixtures.json. These are optional offline
snapshots, not a validated steady-state transport reconstruction. No fixture is
included in the agent bundle; configure FUSION_THERMAL_FIXTURES at the site.
Note `constant` transport makes every device look alike in chi and defeats the
point of comparing facilities; `qlknn` is the physically meaningful choice and
is slower to compile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_ROOT))

from hfmarl.devices.registry import DEVICES
from hfmarl.envs.actuators import bank_from_device
from hfmarl.envs.torax_config import build_config
from hfmarl.envs.torax_env import ToraxCore

from fusion_agent.thermal.torax_extract import (
    electron_source,
    fraction_to_action,
)

# Must match SITES in prepare_thermal_investigation.py.
SITES = {"A": "diiid_like", "B": "sparc_like", "C": "tcv_like"}
# C requests weak heating. Whether its profile passes the analogy criterion must
# be computed, not assumed to match the reduced-model demonstration.
POWER_FRACTION = {"A": 1.0, "B": 1.0, "C": 0.001}


def electron_heating(core) -> np.ndarray:
    """Absorbed electron heating density, summed over sources.

    TORAX's source container has moved between versions, so this looks the
    profile up rather than assuming one attribute path, and says what it found
    when it fails. The investigation is ABOUT the difference between commanded
    and delivered power, so a silently wrong source would invalidate the whole
    demonstration rather than merely degrade it.
    """
    return electron_source(core.state)


def run_site(site: str, transport: str, steps: int, delta_t: float) -> dict:
    device = DEVICES[SITES[site]]
    cfg = build_config(device, t_final=delta_t * (steps + 2), transport_model=transport)
    core = ToraxCore(cfg, delta_t)
    core.start()
    core.reset()
    bank = bank_from_device(device, ("thermal",))
    level = fraction_to_action(POWER_FRACTION[site])
    for i in range(steps):
        values = bank.from_unit_actions(np.full(len(bank), level))
        core.apply_updates(bank.build_updates(values, core.t, delta_t))
        ok, done, err = core.advance()
        if not ok:
            raise RuntimeError(f"{site}: TORAX step {i} failed: {err}")
        if done:
            break
    profiles = core.read_profiles()
    geo = core.state.geometry
    rho_norm = np.asarray(geo.rho_norm, dtype=float)
    source = electron_heating(core)
    arrays = [np.asarray(profiles[k], dtype=float) for k in ("n_e", "T_e")]
    if any(
        a.shape != rho_norm.shape or not np.all(np.isfinite(a))
        for a in [source, *arrays]
    ):
        raise ValueError("Snapshot arrays do not share the same finite cell grid")
    return {
        "device": device.name,
        "provenance": f"torax-1.4.3-{transport}",
        "rho": rho_norm.tolist(),
        "r": (rho_norm * device.a_minor).tolist(),
        "n_e": np.asarray(profiles["n_e"], dtype=float).tolist(),
        "T_e_keV": np.asarray(profiles["T_e"], dtype=float).tolist(),
        "source": source.tolist(),
        "source_basis": "net-electron-source-including-exchange",
        "time_seconds": float(core.t),
        "steady_state_verified": False,
        "R_major": device.R_major,
        "a_minor": device.a_minor,
        "B_0": device.B_0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", default="qlknn", choices=["qlknn", "constant"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--delta-t", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=APP_ROOT / ".fusion-state/torax-fixtures.json",
    )
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.delta_t) or args.delta_t <= 0:
        parser.error("steps and delta-t must be positive")

    fixtures = {}
    for site, device_name in SITES.items():
        print(f"running {site} = {device_name} ...", flush=True)
        fixtures[site] = run_site(site, args.transport, args.steps, args.delta_t)
        T = np.asarray(fixtures[site]["T_e_keV"])
        print(f"  T_axis={T[0]:.3f} keV  T_edge={T[-1]:.3f} keV")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixtures, indent=2, allow_nan=False))
    print(f"\nWrote {args.out}")
    print(
        "Keep at the site; set FUSION_THERMAL_FIXTURES to this path. Not bundled or uploaded."
    )


if __name__ == "__main__":
    main()
