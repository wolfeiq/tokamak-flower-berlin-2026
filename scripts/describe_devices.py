#!/usr/bin/env python
"""Characterise the device set in dimensionless space. Needs no TORAX.

Run this BEFORE Phase 5 and paste the output into FINDINGS.md. Phase 6 is
impossible if the devices do not overlap in dimensionless space, and the
similarity bandwidth has two silent failure modes (every peer ignored, or all
peers weighted equally) that would quietly destroy the experiment.

    uv run python scripts/describe_devices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfmarl.devices.registry import all_devices, encoded_states  # noqa: E402
from hfmarl.federation.similarity import describe_device_set  # noqa: E402


def main() -> int:
    print(f"{'device':14s}{'R/a':>7s}{'B_0':>7s}{'q_cyl':>8s}{'n_GW[1e20]':>12s}")
    for d in all_devices():
        print(
            f"{d.name:14s}{d.aspect_ratio:7.2f}{d.B_0:7.2f}"
            f"{d.q_cylindrical:8.2f}{d.greenwald_density / 1e20:12.2f}"
        )
    states = encoded_states()
    print()
    print(describe_device_set(states))
    print()
    print("notes:")
    for d in all_devices():
        if d.note:
            print(f"  {d.name}: {d.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
