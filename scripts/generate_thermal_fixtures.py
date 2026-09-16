#!/usr/bin/env python
"""Run TORAX for each investigation facility and emit profile fixtures.

The companion app solves a REDUCED 1-D transport model by default, because
TORAX cannot be installed everywhere: torax==1.4.3 requires jax>=0.10.0, and
jaxlib publishes no macOS x86_64 wheel past 0.4.38, so Intel Macs cannot run it
at all. Run this wherever TORAX does install (Linux x86_64, Apple Silicon,
Colab) and commit the result; the app prefers these fixtures when present and
reports provenance either way.

    uv run python scripts/generate_thermal_fixtures.py --transport qlknn

Writes apps/thermal-investigation/thermal_investigation/_fixtures.json.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hfmarl.devices.registry import DEVICES  # noqa: E402
from hfmarl.envs.actuators import bank_from_device  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.envs.torax_env import ToraxCore  # noqa: E402

# Must match SITES in prepare_thermal_investigation.py.
SITES = {"A": "diiid_like", "B": "sparc_like", "C": "tcv_like"}
# C is the unidentifiable case: barely heated, so its profile never develops a
# gradient steep enough to identify transport from.
POWER_FRACTION = {"A": 1.0, "B": 1.0, "C": 0.001}


def electron_heating(core) -> np.ndarray:
    """Absorbed electron heating density, summed over sources.

    TORAX's source container has moved between versions, so this looks the
    profile up rather than assuming one attribute path, and says what it found
    when it fails. The investigation is ABOUT the difference between commanded
    and delivered power, so a silently wrong source would invalidate the whole
    demonstration rather than merely degrade it.
    """
    sources = getattr(core.state, "core_sources", None)
    if sources is None:
        raise RuntimeError("core_sources missing from TORAX state; cannot obtain source")
    total = None
    seen = []
    for name in dir(sources):
        if name.startswith("_"):
            continue
        value = getattr(sources, name, None)
        profile = getattr(value, "value", value)
        if not isinstance(profile, (np.ndarray, list)):
            continue
        arr = np.asarray(profile, dtype=float).squeeze()
        seen.append(f"{name}{arr.shape}")
        if arr.ndim != 1 or "temp_el" not in name and "T_e" not in name and "el" not in name:
            continue
        total = arr if total is None else total + arr
    if total is None:
        raise RuntimeError(
            "No electron heating profile found on core_sources. Saw: " + ", ".join(seen)
        )
    return total


def run_site(site: str, transport: str, steps: int, delta_t: float) -> dict:
    device = DEVICES[SITES[site]]
    cfg = build_config(device, t_final=delta_t * (steps + 2), transport_model=transport)
    core = ToraxCore(cfg, delta_t)
    core.start()
    core.reset()
    bank = bank_from_device(device, ("thermal",))
    level = POWER_FRACTION[site]
    for i in range(steps):
        values = bank.from_unit_actions(np.full(len(bank), level))
        core.apply_updates(bank.build_updates(values, core.t, delta_t))
        ok, done, err = core.advance()
        if not ok:
            raise RuntimeError(f"{site}: TORAX step {i} failed: {err}")
        if done:
            break
    profiles = core.read_profiles()
    geo = getattr(core, "geo", None) or getattr(core.state, "geometry", None)
    rho_norm = np.asarray(getattr(geo, "rho_norm", getattr(geo, "rho_face_norm", None)), dtype=float)
    return {
        "device": device.name,
        "provenance": f"torax-1.4.3-{transport}",
        "rho": rho_norm.tolist(),
        "r": (rho_norm * device.a_minor).tolist(),
        "n_e": np.asarray(profiles["n_e"], dtype=float).tolist(),
        "T_e_keV": np.asarray(profiles["T_e"], dtype=float).tolist(),
        "source": electron_heating(core).tolist(),
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
        default=ROOT / "apps/thermal-investigation/thermal_investigation/_fixtures.json",
    )
    args = parser.parse_args()

    fixtures = {}
    for site in SITES:
        print(f"running {site} = {SITES[site]} ...", flush=True)
        fixtures[site] = run_site(site, args.transport, args.steps, args.delta_t)
        T = np.asarray(fixtures[site]["T_e_keV"])
        print(f"  T_axis={T[0]:.3f} keV  T_edge={T[-1]:.3f} keV")
    args.out.write_text(json.dumps(fixtures, indent=2))
    print(f"\nWrote {args.out}")
    print("Commit this file; the app prefers it over the reduced solve.")


if __name__ == "__main__":
    main()
