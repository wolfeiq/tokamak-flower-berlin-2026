"""The classical baseline, and the contract that keeps it honest.

PROTOCOL.md §3: if a PI controller built from the same calibration sweep
reaches the same endpoint as the federated policy, this is a control problem
with an RL solution bolted on. That makes this the most important arm to get
RIGHT rather than merely to include -- a baseline that is accidentally weak
manufactures a win for everything compared against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.agents.conventional import ConventionalController

# A plausible measured map: command -1..1 moves beta_N 0.46..0.71.
LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]
BETAS = [0.46, 0.52, 0.58, 0.64, 0.71]


def obs(beta, target):
    """The first three entries of `_observe`; the rest are unused here."""
    return np.array([beta, beta - target, target, 4.0, 0.5, 0.8, 1.0, 0.1, 0.5])


def ctrl():
    return ConventionalController.from_sweep(LEVELS, BETAS)


# -- the feedforward, which is what makes it a fair baseline -------------


def test_on_target_it_commands_the_holding_level():
    """With zero error the output is the inverted static map -- the part a
    classical designer gets free from a sweep. Omitting the feedforward is the
    commonest way to make a proposed method look good."""
    c = ctrl()
    u = c.act(obs(0.58, 0.58))[0]
    assert abs(u - 0.0) < 0.05, "0.58 is the measured beta at command 0"


def test_it_pushes_the_right_way():
    c = ctrl()
    below = c.act(obs(0.50, 0.58))[0]
    c.reset()
    above = c.act(obs(0.66, 0.58))[0]
    assert below > 0 > above


def test_the_feedforward_tracks_a_moving_target():
    """A different target must move the baseline command, or the controller is
    a constant and every comparison against it is meaningless."""
    c = ctrl()
    low = c.act(obs(0.50, 0.50))[0]
    c.reset()
    high = c.act(obs(0.68, 0.68))[0]
    assert high > low


# -- closed loop on the fitted map ---------------------------------------


def test_it_converges_on_its_own_model():
    """Not a claim about TORAX -- a check that the gains are sane. If it cannot
    regulate the linear map it was built from, any failure on the real plant
    would be uninterpretable."""
    c = ctrl()
    beta, target = 0.46, 0.66
    for _ in range(40):
        u = float(c.act(obs(beta, target))[0])
        beta = c.slope * u + c.intercept
    assert abs(beta - target) < 0.01


def test_the_integral_does_not_wind_up_forever():
    """The command saturates at the envelope, so an unbounded integral only
    delays the reversal when the error finally changes sign."""
    c = ctrl()
    for _ in range(500):
        c.act(obs(0.10, 0.90))  # far below target, saturated the whole time
    assert np.isfinite(c.integral)
    assert abs(c.integral) <= abs(2.0 / c.ki) + 1e-9


def test_a_reset_clears_the_integral_between_shots():
    """Carrying it across discharges would let this arm learn from shot to
    shot, which is precisely what it must not do."""
    c = ctrl()
    for _ in range(20):
        c.act(obs(0.10, 0.90))
    assert c.integral != 0.0
    c.reset()
    assert c.integral == 0.0


# -- the contracts that stop it lying ------------------------------------


def test_it_refuses_a_weight_vector_rather_than_ignoring_one():
    """Silently accepting weights would make a constant controller look like a
    trained baseline in every table it appears in."""
    with pytest.raises(TypeError, match="no learned parameters"):
        ctrl().set_flat(np.zeros(194))


def test_a_flat_calibration_sweep_is_refused():
    """Zero gain means the actuators do not move beta_N on this device. A
    controller built on it would command infinite power, and the honest report
    is that no controller can track here."""
    with pytest.raises(ValueError, match="flat"):
        ConventionalController.from_sweep([-1, 0, 1], [0.5, 0.5, 0.5])


def test_too_few_calibration_points_is_refused():
    with pytest.raises(ValueError, match="two usable points"):
        ConventionalController.from_sweep([0.0], [0.5])


def test_nonfinite_sweep_points_are_dropped_not_fitted():
    """A failed calibration shot returns NaN. Fitting through it would produce
    gains from noise while looking like a successful calibration."""
    c = ConventionalController.from_sweep(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        [0.46, float("nan"), 0.58, 0.64, 0.71])
    assert np.isfinite(c.slope) and c.slope > 0


def test_the_command_stays_inside_the_actuator_envelope():
    c = ctrl()
    for beta, target in ((0.0, 5.0), (5.0, 0.0), (0.58, 0.58)):
        u = c.act(obs(beta, target))[0]
        assert -1.0 <= u <= 1.0


# -- zero observed failures is not zero failure probability ---------------


def test_zero_failures_does_not_certify_zero_failure_rate():
    """AUDIT. The script reported 100.0% from ten evaluation shots as though it
    settled the matter. Zero failures in 10 is consistent with a true failure
    rate up to 25.9%, which on a safety claim is the whole claim."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "exp_conventional", root / "scripts" / "exp_conventional.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.failure_upper_bound(10, 0) == pytest.approx(0.259, abs=0.005)
    assert mod.failure_upper_bound(100, 0) == pytest.approx(0.030, abs=0.005)
    # More evidence must never loosen the bound.
    bounds = [mod.failure_upper_bound(n, 0) for n in (10, 20, 50, 100, 300)]
    assert all(a > b for a, b in zip(bounds, bounds[1:]))
    # Observed failures must widen it beyond the zero-failure case.
    assert mod.failure_upper_bound(100, 2) > mod.failure_upper_bound(100, 0)
