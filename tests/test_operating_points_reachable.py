"""The similarity coordinates must be points the plant can actually occupy.

WHAT THIS CATCHES
-----------------
`OPERATING_POINTS` is a hand-written table of nominal kinetics. `BETA_N_BANDS`
is measured: TORAX driven from zero to full command, recorded by
`scripts/gate_authority.py`. Every aggregation weight in the project is a
Gaussian kernel over distance in (rho*, nu*, beta_N, q95) computed from the
first, and nothing had ever compared it against the second.

They disagree. Two of four devices are assigned an operating beta_N roughly
1.8x above anything their simulated plant can produce at full power, and a
third is assigned one at 17% of what its plant reaches. So the physics
weighting of SPEC.md 4b -- the project's central claim -- has been computed at
coordinates that the devices cannot occupy.

This is xfail rather than a failure: the finding is recorded, the numbers are
pinned, and the test turns green the moment RUNBOOK 5's "re-derive the
operating points from TORAX output" is done. Deleting it would delete the only
automatic check that the two ever agree.
"""

from __future__ import annotations

import pytest

from hfmarl.devices.registry import BETA_N_BANDS, DEVICES, encoded_states

TASK = "easy"


def test_every_device_has_a_measured_band_for_the_reference_task():
    for name in sorted(DEVICES):
        assert TASK in BETA_N_BANDS[name], f"{name} has no measured {TASK} band"


@pytest.mark.parametrize("name", sorted(DEVICES))
@pytest.mark.xfail(
    reason="OPERATING_POINTS is the hand-written nominal table; RUNBOOK 5 "
           "asks for it to be re-derived from TORAX output and it has not "
           "been. iter_like and sparc_like sit ~1.8x above their reachable "
           "beta_N, tcv_like at ~0.17x of its.",
    strict=False,
)
def test_nominal_operating_point_is_inside_the_measured_band(name):
    lo, hi = BETA_N_BANDS[name][TASK]
    beta = encoded_states()[name].beta_N
    assert lo <= beta <= hi, (
        f"{name}: the similarity weighting places it at beta_N {beta:.3f}, "
        f"but its measured reachable band on '{TASK}' is {lo:.3f}..{hi:.3f}. "
        "Every aggregation weight involving this device is computed at a point "
        "its plant cannot occupy."
    )


def test_the_disagreement_is_recorded_rather_than_forgotten():
    """Pins the size of the gap, so a silent change to either table shows up.

    If someone re-derives OPERATING_POINTS properly, this fails and gets
    deleted along with the xfail above -- which is the intended ending.
    """
    ratios = {}
    for name in sorted(DEVICES):
        _, hi = BETA_N_BANDS[name][TASK]
        ratios[name] = encoded_states()[name].beta_N / hi

    assert ratios["iter_like"] == pytest.approx(1.79, abs=0.05)
    assert ratios["sparc_like"] == pytest.approx(1.90, abs=0.05)
    assert ratios["diiid_like"] < 1.0
    assert ratios["tcv_like"] == pytest.approx(0.16, abs=0.03)
