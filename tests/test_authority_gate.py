"""Decision rules of the actuator-authority gate (scripts/gate_authority.py).

The gate itself needs TORAX; `assess` does not, so the logic that decides
PASS/FAIL is testable here. These encode the three silent failures the gate
exists to catch -- see the script's module docstring.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gate_authority import assess  # noqa: E402

from hfmarl.envs.task import PRESETS  # noqa: E402


# A real device name, because `assess` resolves the task against it: the
# presets carry fractions of that device measured band, not beta_N.
DEVICE = "diiid_like"


def _rows(betas, violated_at_zero=None):
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(betas) == len(levels)
    return [
        {
            "level": lv,
            "beta_N": b,
            "violated_at": violated_at_zero if lv == 0.0 else None,
            "violations": ["beta_N"] if (lv == 0.0 and violated_at_zero) else [],
        }
        for lv, b in zip(levels, betas)
    ]


def _resolved(task="easy"):
    from hfmarl.devices.registry import get as get_device

    return PRESETS[task].resolve_for(get_device(DEVICE))


def test_setpoint_inside_the_band_passes():
    ok, reasons = assess(DEVICE, _rows([0.50, 1.27, 2.14, 2.60, 3.19]), "easy")
    assert ok, reasons


def test_setpoint_above_full_power_fails():
    """The rail case: full command lands below the target."""
    t = _resolved().setpoint.base
    ok, reasons = assess(
        DEVICE, _rows([0.1 * t, 0.2 * t, 0.4 * t, 0.6 * t, 0.9 * t]), "easy")
    assert not ok
    assert any("saturates at maximum power" in r for r in reasons)


def test_setpoint_below_zero_power_fails():
    """The other rail: zero command already sits above the target."""
    t = _resolved().setpoint.base
    ok, reasons = assess(
        DEVICE, _rows([1.1 * t, 1.5 * t, 2.0 * t, 2.5 * t, 3.0 * t]), "easy")
    assert not ok
    assert any("ZERO command already gives" in r for r in reasons)


def test_negligible_authority_fails_even_without_saturation():
    """The band brackets the target but is narrower than the tolerance."""
    spec = _resolved()
    tol, mid = spec.tolerance, spec.setpoint.base
    band = [mid - tol / 4, mid - tol / 8, mid, mid + tol / 8, mid + tol / 4]
    ok, reasons = assess(DEVICE, _rows(band), "easy")
    assert not ok
    assert any("actuator authority" in r for r in reasons)


def test_immediate_violation_at_zero_command_fails():
    """A device that violates on step 1 regardless of action has no gradient."""
    ok, reasons = assess(DEVICE,
                         _rows([0.50, 1.27, 2.14, 2.60, 3.19], violated_at_zero=1),
                         "easy")
    assert not ok
    assert any("step 1" in r for r in reasons)


def test_all_nonfinite_is_a_failure_not_a_pass():
    ok, reasons = assess(DEVICE, _rows([float("nan")] * 5), "easy")
    assert not ok
    assert any("no finite beta_N" in r for r in reasons)


@pytest.mark.parametrize("task", sorted(PRESETS))
def test_assess_runs_for_every_preset(task):
    """Every preset must be assessable; a schedule the gate cannot read is a bug."""
    ok, _ = assess(DEVICE, _rows([0.1, 1.0, 2.0, 3.0, 4.0]), task)
    assert isinstance(ok, bool)


def test_every_preset_resolves_inside_every_device_band():
    """The regression the whole normalisation exists to prevent.

    Before it, one absolute beta_N = 2.0 sat outside three of the four bands,
    and the tolerance was 7.5% of DIII-D control authority against 197% of
    SPARC. Both have to hold for every preset on every device, or
    shots-to-threshold measures device calibration instead of learning.
    """
    import numpy as np

    from hfmarl.devices.registry import all_devices, beta_N_band
    from hfmarl.envs.limits import DEFAULT_LIMITS

    soft = next(l.soft for l in DEFAULT_LIMITS if l.name == "beta_N")
    for name in sorted(PRESETS):
        for d in all_devices():
            lo, hi = beta_N_band(d.name, task=name, soft_limit=soft)
            spec = PRESETS[name].resolve_for(d)
            traj = spec.setpoint.trajectory(
                np.linspace(0.0, spec.episode_length, 200), spec.episode_length)
            assert lo < traj.min() - spec.tolerance, (name, d.name, "below band")
            assert traj.max() + spec.tolerance < hi, (name, d.name, "above band")
            # Equal difficulty: the tolerance is the same share of authority.
            share = spec.tolerance / (hi - lo)
            assert share == pytest.approx(PRESETS[name].tolerance, rel=1e-9)
