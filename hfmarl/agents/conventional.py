"""A classical controller built from design information and a calibration sweep.

THE BASELINE THAT IS EASIEST TO SKIP AND MOST DAMAGING TO OMIT
--------------------------------------------------------------
Every arm in this study is a learned controller compared against another
learned controller. None of them establishes that learning was needed. If a
proportional-integral controller built from the same calibration shots reaches
the same endpoint, then this is a control-engineering problem with an RL
solution bolted on, and no amount of federation makes that finding go away.

PROTOCOL.md §3 lists it first for that reason.

WHAT IT IS ALLOWED TO USE
-------------------------
Design information (free) plus the command-to-beta_N static map, which costs
calibration shots and is charged. Nothing else: no target training shots, no
tuning on the held-out device. The gains come from the sweep by a stated rule,
not from a search -- a baseline tuned on the device it is evaluated on would be
a stronger claim than any arm it is compared against is allowed to make.

DESIGN
------
    u(t) = ff(target(t)) + Kp e(t) + Ki integral(e)

`ff` inverts the measured static map, so the controller starts each step at the
command that would hold the current target in steady state. That is the part a
classical designer gets for free from a sweep, and omitting it would make the
baseline artificially weak -- the commonest way to make a proposed method look
good.

Kp is derived from the sweep's own gain: a unit of command produces
`dbeta/du`, so `Kp = kp_fraction / (dbeta/du)` closes a fraction of the error
per action window. Ki is a fixed fraction of Kp. Both are stated constants
chosen once, on source devices, and recorded here rather than fitted per device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Chosen once, on source devices, and held fixed for every held-out run.
# Deliberately mild: this baseline should be a fair classical controller, not a
# straw man and not a hand-tuned champion.
KP_FRACTION = 0.6  # close 60% of the remaining error per action window
KI_FRACTION = 0.15  # integral gain as a fraction of Kp


@dataclass
class ConventionalController:
    """PI plus static feedforward, in the policy interface `fire_shot` expects.

    `set_flat` and `get_flat` exist because every other controller here is a
    parameter vector and the shot-firing path is shared. This one has no
    parameters to set, and says so rather than silently accepting a vector and
    ignoring it -- an arm that quietly discards the weights handed to it would
    look like a working baseline while being a constant.
    """

    slope: float  # dbeta_N / du, from the calibration sweep
    intercept: float  # beta_N at u = 0
    kp: float
    ki: float
    integral: float = 0.0
    n_actions: int = 2
    history: list = field(default_factory=list)

    @classmethod
    def from_sweep(cls, levels, betas, n_actions: int = 2,
                   kp_fraction: float = KP_FRACTION,
                   ki_fraction: float = KI_FRACTION) -> "ConventionalController":
        """Fit the static map beta_N = slope * u + intercept and set gains.

        Raises rather than guessing when the sweep is degenerate: a flat map
        means the actuators do not move beta_N on this device, and a controller
        built on a zero gain would command infinite power. That is a fact about
        the device worth surfacing, not a numerical edge case to paper over.
        """
        u = np.asarray(levels, float)
        b = np.asarray(betas, float)
        good = np.isfinite(u) & np.isfinite(b)
        if good.sum() < 2:
            raise ValueError("calibration sweep has fewer than two usable points")
        slope, intercept = np.polyfit(u[good], b[good], 1)
        if abs(slope) < 1e-9:
            raise ValueError(
                f"calibration sweep is flat (dbeta/du = {slope:.3g}): the "
                "actuators do not move beta_N on this device, so no controller "
                "-- classical or learned -- can track a setpoint here")
        return cls(slope=float(slope), intercept=float(intercept),
                   kp=float(kp_fraction / slope),
                   ki=float(ki_fraction * kp_fraction / slope),
                   n_actions=n_actions)

    # -- the policy interface --------------------------------------------

    def set_flat(self, vector) -> None:
        raise TypeError(
            "ConventionalController has no learned parameters. It is the arm "
            "that tests whether learning was needed at all, so handing it a "
            "weight vector is a category error -- and silently ignoring one "
            "would make a constant controller look like a trained baseline.")

    def get_flat(self) -> np.ndarray:
        return np.zeros(0)

    def reset(self) -> None:
        self.integral = 0.0

    def act(self, obs) -> np.ndarray:
        """obs = [beta_N, beta_N - target, target, ...] -- see `_observe`."""
        o = np.asarray(obs, float)
        beta, target = float(o[0]), float(o[2])
        err = target - beta  # positive means we need more power

        self.integral += err
        # Anti-windup: the command saturates at the envelope, so an integral
        # that keeps growing past that only delays the eventual reversal.
        limit = abs(2.0 / self.ki) if self.ki else 0.0
        if limit:
            self.integral = float(np.clip(self.integral, -limit, limit))

        feedforward = (target - self.intercept) / self.slope
        u = feedforward + self.kp * err + self.ki * self.integral
        u = float(np.clip(u, -1.0, 1.0))
        return np.full(self.n_actions, u)
