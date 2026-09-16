"""Headroom: is the task hard enough for federation to show anything?

THE FAILURE THIS PREVENTS
-------------------------
"If isolated training reaches threshold in a few hundred shots, the gain is too
small to matter. Design the task hard enough that isolated learning is
genuinely slow -- otherwise there's no headroom for federation to show
anything."

This is not a caveat to note in the discussion section. It is a precondition,
and it is cheap to check and expensive to discover late: a full Phase 5 matrix
is 5 conditions x 4 devices x several seeds, and if isolated converges in 200
shots then every one of those runs was wasted before it started.

So it runs as a BLOCKING GATE before Phase 5, with a go/no-go on a number.

THE TENSION THE GATE HAS TO BALANCE
-----------------------------------
Harder task -> more headroom -> but also more compute. These pull against each
other, and optimising either alone gives a useless experiment:

  * too easy  -> isolated converges fast, no room for a ratio to exist
  * too hard  -> isolated never converges, so there is no denominator at all,
                 and the "ratio" becomes a comparison of failure rates
  * too slow  -> headroom is fine but the matrix does not finish this year

`assess` reports all three and names the binding one.

THE THREAT TO THIS GATE'S VALIDITY -- read before trusting a PASS
------------------------------------------------------------------
Shots-to-threshold is a property of the TASK **and the optimiser**, not of the
task alone. `gate_headroom.py` measures it with simple hill climbing, because
that consumes exactly one shot per step and keeps the shot axis exact (CEM
evaluates a whole population per iteration, so "shots consumed" jumps in blocks
and blurs the quantity being measured).

But a weak optimiser needs more shots. So this gate is BIASED TOWARD PASSING:
a task that hill climbing takes 3000 shots to crack might take PPO 300, at
which point the headroom it certified does not exist.

Two consequences, neither optional:

  * A PASS here is provisional. Re-measure the isolated baseline with whatever
    algorithm the Phase 5 matrix actually uses, and if it converges far faster,
    the task must be made harder before the matrix means anything.
  * A FAIL here is trustworthy in the other direction. If even hill climbing
    converges in a few hundred shots, the task is definitively too easy and no
    stronger algorithm will change that.

The reverse bias does not exist: nothing here can make a task look harder than
it is for a better learner.
"""

from __future__ import annotations

from dataclasses import dataclass

from hfmarl.metrics.curves import ThresholdResult

# Below this, isolated learning is too fast for a speedup ratio to mean much:
# the difference between 150 and 90 shots is within campaign-to-campaign noise
# and no fusion audience would act on it.
MIN_ISOLATED_SHOTS = 1000

# Above this, isolated is unlikely to converge reliably inside any affordable
# budget, and the comparison degenerates into "which condition failed less".
MAX_ISOLATED_SHOTS = 20000

# A matrix taking longer than this is not a research loop any more.
MAX_MATRIX_HOURS = 72.0


@dataclass
class HeadroomVerdict:
    """Whether the task admits a measurable federation benefit, affordably."""

    isolated: ThresholdResult
    seconds_per_shot: float
    n_conditions: int
    n_devices: int
    n_seeds: int
    shot_budget: int

    # -- derived ---------------------------------------------------------

    @property
    def has_headroom(self) -> bool:
        """Isolated is slow enough that a speedup could be worth reporting."""
        return (self.isolated.n_reached > 0
                and self.isolated.median >= MIN_ISOLATED_SHOTS)

    @property
    def is_measurable(self) -> bool:
        """Isolated converges reliably enough to be a denominator."""
        return self.isolated.reach_rate >= 0.8 and self.isolated.median <= MAX_ISOLATED_SHOTS

    @property
    def matrix_hours(self) -> float:
        """Serial wall-clock for the full Phase 5 matrix."""
        runs = self.n_conditions * self.n_devices * self.n_seeds
        return runs * self.shot_budget * self.seconds_per_shot / 3600.0

    @property
    def is_affordable(self) -> bool:
        return self.matrix_hours <= MAX_MATRIX_HOURS

    @property
    def passes(self) -> bool:
        return self.has_headroom and self.is_measurable and self.is_affordable

    @property
    def binding_constraint(self) -> str:
        if not self.is_measurable:
            return "measurability"
        if not self.has_headroom:
            return "headroom"
        if not self.is_affordable:
            return "cost"
        return "none"

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "isolated shots-to-threshold: " + self.isolated.summary(),
            f"  reached by {self.isolated.n_reached}/{self.isolated.n_total} seeds",
            "",
            f"headroom     (median >= {MIN_ISOLATED_SHOTS} shots)      : "
            f"{'PASS' if self.has_headroom else 'FAIL'}",
            f"measurable   (>=80% seeds converge, <= {MAX_ISOLATED_SHOTS}): "
            f"{'PASS' if self.is_measurable else 'FAIL'}",
            f"affordable   (matrix <= {MAX_MATRIX_HOURS:.0f} h serial)      : "
            f"{'PASS' if self.is_affordable else 'FAIL'}"
            f"  [{self.matrix_hours:.1f} h at {self.seconds_per_shot:.2f} s/shot]",
            "",
        ]
        if self.passes:
            lines.append("PASS: the task admits a measurable federation benefit.")
            lines.append("Proceed to the Phase 5 matrix.")
            return "\n".join(lines)

        lines.append(f"FAIL: binding constraint is {self.binding_constraint.upper()}.")
        lines.append("")
        lines.extend(self.advice())
        return "\n".join(lines)

    def advice(self) -> list[str]:
        """What to change, specifically, given which constraint binds."""
        if not self.is_measurable and self.isolated.reach_rate < 0.8:
            return [
                "Isolated training does not converge reliably, so there is no",
                "denominator for a speedup ratio. The task is TOO HARD. Make it",
                "easier, in this order:",
                "  * widen the tracking tolerance, or lower the threshold",
                "    (--threshold-fraction) toward what isolated can actually reach",
                "  * use a constant setpoint rather than a moving one",
                "  * shorten the episode so credit assignment is easier",
                "  * transport_model='constant' instead of 'qlknn'",
                "Then re-run this gate.",
            ]
        if not self.has_headroom:
            return [
                f"Isolated converges in ~{self.isolated.median:.0f} shots, below the",
                f"{MIN_ISOLATED_SHOTS}-shot floor. Any federation speedup here is",
                "within campaign noise and no fusion audience would act on it.",
                "THIS KILLS THE CLAIM AS CURRENTLY SPECIFIED. Make the task harder:",
                "  * moving setpoint (ramp or step schedule) instead of constant",
                "  * tighter tracking tolerance",
                "  * transport_model='qlknn' -- stiff, nonlinear, genuinely hard",
                "  * enable fusion alpha heating (self-heating nonlinearity)",
                "  * more actuators to coordinate (add the particle cluster)",
                "  * harsher initial conditions, or per-shot disturbances",
                "  * longer episodes, so limits are reachable and must be planned for",
                "Re-run this gate after each change. Do NOT proceed to Phase 5",
                "until it passes -- the whole matrix would measure nothing.",
            ]
        return [
            f"The matrix would take {self.matrix_hours:.0f} h serially, over the",
            f"{MAX_MATRIX_HOURS:.0f} h budget. Options, cheapest first:",
            "  * run devices and conditions as PARALLEL PROCESSES -- this",
            "    workload is embarrassingly parallel and the WSL box's cores are",
            "    the resource that actually helps (see gate0_bench.py)",
            "  * reduce seeds to 3 (the floor for a trend claim), not fewer",
            "  * shorten episodes, or increase delta_t_a so a shot is fewer steps",
            "  * cut the shot budget to just above the isolated convergence point",
            "Reducing DEVICES is the last resort: device heterogeneity is the",
            "experimental variable, and dropping the small machine removes the",
            "case federation is supposed to help most.",
        ]


def assess(
    isolated: ThresholdResult,
    seconds_per_shot: float,
    shot_budget: int,
    n_conditions: int = 5,
    n_devices: int = 4,
    n_seeds: int = 5,
) -> HeadroomVerdict:
    return HeadroomVerdict(
        isolated=isolated,
        seconds_per_shot=seconds_per_shot,
        n_conditions=n_conditions,
        n_devices=n_devices,
        n_seeds=n_seeds,
        shot_budget=shot_budget,
    )
