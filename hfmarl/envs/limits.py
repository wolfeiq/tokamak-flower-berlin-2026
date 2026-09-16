"""Operational limits -- the boundaries of the safe operating envelope.

These three limits are what SPEC.md Phase 1 requires and what the whole
catastrophe claim of Phase 6 is built on. The experiment there is: restrict
device A to a safe envelope, let B and C cross these limits, then test whether
federated A avoids a violation it was never shown.

So the limits must be (a) physically real, (b) computed identically on every
device, and (c) reported as a continuous margin rather than a bare boolean --
a policy cannot learn to avoid a cliff it only sees once it has fallen off.

All three quantities come straight out of TORAX's PostProcessedOutputs; none
is derived or approximated here. That matters: if we recomputed the Greenwald
fraction ourselves we would be grading the policy against a different number
than the simulator is enforcing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Limit:
    """One operational limit, with a soft warning band before the hard edge.

    ``soft`` is where the penalty starts; ``hard`` is where the episode is
    considered to have entered a catastrophic regime. The band between them is
    what gives the policy a gradient to learn from.
    """

    name: str
    soft: float
    hard: float
    upper: bool  # True: violation is exceeding. False: violation is falling below.
    description: str = ""

    def margin(self, value: float) -> float:
        """Normalised distance to the hard limit.

        +1 is comfortably safe, 0 is exactly at the hard limit, negative is a
        violation. Linear in the soft-to-hard band so the gradient is constant
        where it matters.
        """
        if self.upper:
            if self.hard <= self.soft:
                raise ValueError(f"{self.name}: hard must exceed soft for an upper limit")
            return (self.hard - value) / (self.hard - self.soft)
        if self.hard >= self.soft:
            raise ValueError(f"{self.name}: hard must be below soft for a lower limit")
        return (value - self.hard) / (self.soft - self.hard)

    def violated(self, value: float) -> bool:
        return value > self.hard if self.upper else value < self.hard


# Default envelope. Values are standard tokamak operational boundaries, not
# tuned: the Greenwald fraction limit is the empirical density limit
# (Greenwald, Nucl. Fusion 28, 2199, 1988); beta_N = 3.0 sits at the lower end
# of the ideal-MHD no-wall Troyon limit (2.8-4); q95 = 2.0 is the classical
# hard edge below which the plasma is disruptively unstable.
DEFAULT_LIMITS: tuple[Limit, ...] = (
    Limit(
        "greenwald_fraction",
        soft=0.80,
        hard=1.00,
        upper=True,
        description="n_e / n_GW. Above 1 the empirical density limit is crossed "
        "and a radiative collapse or density-limit disruption follows.",
    ),
    Limit(
        "beta_N",
        soft=2.50,
        hard=3.00,
        upper=True,
        description="Troyon-normalised beta. Above the no-wall limit the plasma "
        "is ideal-MHD unstable.",
    ),
    Limit(
        "q95",
        soft=2.50,
        hard=2.00,
        upper=False,
        description="Safety factor at the 95% flux surface. Below 2 the plasma "
        "is disruptively unstable to low-order kink modes.",
    ),
)


@dataclass
class LimitReport:
    """Outcome of evaluating the whole envelope at one instant."""

    margins: dict[str, float] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    @property
    def worst_margin(self) -> float:
        """The binding constraint. This is what the reward penalises."""
        return min(self.margins.values()) if self.margins else 1.0

    @property
    def any_violated(self) -> bool:
        return bool(self.violations)

    def __str__(self) -> str:
        parts = [
            f"{k}={self.values[k]:.3f}(m={self.margins[k]:+.2f})" for k in self.margins
        ]
        s = " ".join(parts)
        return s + (f"  VIOLATED: {', '.join(self.violations)}" if self.violations else "")


class LimitSet:
    """Evaluates an operating envelope against observed plasma quantities."""

    def __init__(self, limits: tuple[Limit, ...] = DEFAULT_LIMITS):
        self.limits = limits

    def evaluate(self, observed: dict[str, float]) -> LimitReport:
        """Score every limit for which a value is present.

        Missing keys are skipped rather than defaulted, because defaulting a
        missing safety quantity to a safe value is exactly the bug that makes
        a controller look safe when it is not. ``gate0_env.py`` verifies every
        name is actually present in the TORAX output.
        """
        report = LimitReport()
        for lim in self.limits:
            if lim.name not in observed:
                continue
            v = float(observed[lim.name])
            report.values[lim.name] = v
            report.margins[lim.name] = lim.margin(v)
            if lim.violated(v):
                report.violations = report.violations + (lim.name,)
        return report

    def restricted(self, fraction: float = 0.6) -> "LimitSet":
        """A tightened envelope, for the safe device in the Phase 6 experiment.

        SPEC.md Phase 6 needs device A confined to a regime where it never
        experiences a limit violation, while B and C explore past the edge.
        This shrinks the soft band toward safety so A's own training
        terminates well before the real limit, leaving the genuine limit
        entirely outside its experience.

        ``fraction`` is how far from soft toward hard A is allowed to go.
        Must be strictly below 1: at exactly 1 the "restricted" envelope equals
        the real one, so device A would experience the very limit Phase 6
        requires it never to have seen, and the experiment would silently test
        nothing.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                f"fraction must be in (0, 1); got {fraction}. At 1.0 the "
                "envelope is unchanged, which defeats the Phase 6 design."
            )
        tightened = tuple(
            Limit(
                name=lim.name,
                soft=lim.soft,
                hard=lim.soft + fraction * (lim.hard - lim.soft),
                upper=lim.upper,
                description=lim.description + " [RESTRICTED ENVELOPE]",
            )
            for lim in self.limits
        )
        return LimitSet(tightened)
