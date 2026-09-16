"""Task difficulty -- the knob the headroom gate turns.

"Design the task hard enough that isolated learning is genuinely slow --
otherwise there's no headroom for federation to show anything."

That makes difficulty an experimental parameter, not a constant, so it lives
here as an explicit object rather than being spread across default arguments.
`scripts/gate_headroom.py` sweeps these presets and reports which one puts
isolated shots-to-threshold inside the measurable band.

WHAT ACTUALLY MAKES THIS TASK HARD
----------------------------------
In rough order of how much difficulty they add per unit of extra compute:

  1. A MOVING setpoint. A constant target is a regulation problem solvable by
     a near-constant action; a ramp or step schedule demands anticipation,
     because the plasma's thermal response lags the actuator.
  2. Stiff transport (`qlknn` instead of `constant`). The real nonlinearity --
     transport coefficients depend on the gradients the controller is shaping,
     so the plant changes underneath the policy.
  3. Tight tolerance. Narrows the reward's basin without changing the physics.
  4. Per-shot disturbances. Forces a policy that is robust rather than one
     that has memorised a single trajectory.
  5. Proximity to limits. A target near the beta limit means the safe and
     optimal actions differ, which is the whole point of the limit machinery.

Pure NumPy and fully testable: none of this needs TORAX.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SetpointSchedule:
    """Target beta_N as a function of time within an episode.

    `kind` is one of:
        constant -- hold `base`. Easiest; a regulation problem.
        ramp     -- linear from `base` to `base + amplitude`. Requires
                    anticipation, because the thermal response lags.
        steps    -- piecewise-constant, `n_steps` levels. Hardest of the three:
                    each transition is a step disturbance the policy must
                    reject without overshooting into a limit.
        sine     -- smooth oscillation; tests tracking bandwidth.
    """

    kind: str = "constant"
    base: float = 2.0
    amplitude: float = 0.0
    n_steps: int = 3
    period: float = 5.0

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "ramp", "steps", "sine"):
            raise ValueError(f"unknown setpoint kind {self.kind!r}")
        if self.base <= 0:
            raise ValueError("base setpoint must be positive")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

    def target(self, t: float, t_final: float) -> float:
        """Target at time `t` in an episode of length `t_final`."""
        if t_final <= 0:
            return self.base
        frac = float(np.clip(t / t_final, 0.0, 1.0))
        if self.kind == "constant":
            return self.base
        if self.kind == "ramp":
            return self.base + self.amplitude * frac
        if self.kind == "steps":
            # Level index in [0, n_steps-1]; alternate up/down so the policy
            # cannot simply learn a monotone drift.
            idx = min(int(frac * self.n_steps), self.n_steps - 1)
            sign = 1.0 if idx % 2 == 0 else -1.0
            return self.base + self.amplitude * sign * (idx / max(self.n_steps - 1, 1))
        # `period` is an absolute timescale in seconds, deliberately: a
        # tracking-bandwidth test is about a real frequency, so a longer
        # episode means more cycles rather than slower ones. Unlike ramp/steps
        # this does NOT rescale with t_final -- but t is still clamped to the
        # episode, so queries outside it behave like the other schedules.
        t_clamped = float(np.clip(t, 0.0, t_final))
        return self.base + self.amplitude * float(
            np.sin(2 * np.pi * t_clamped / self.period)
        )

    def trajectory(self, times: np.ndarray, t_final: float) -> np.ndarray:
        return np.array([self.target(float(t), t_final) for t in times], dtype=float)


@dataclass(frozen=True)
class TaskSpec:
    """Everything that sets how hard the control problem is.

    `expected_difficulty` is a label for reporting only. The headroom gate
    measures difficulty rather than trusting it -- the ordering below is a
    hypothesis about these presets, and it is exactly the kind of thing that
    turns out wrong.
    """

    name: str
    setpoint: SetpointSchedule
    tolerance: float  # beta_N error treated as "on target"
    transport_model: str  # constant (smooth) | qlknn (stiff, realistic)
    enable_fusion: bool
    episode_length: float  # seconds of plasma
    delta_t_a: float  # action window, seconds
    disturbance_std: float = 0.0  # per-shot variation in initial conditions
    clusters: tuple[str, ...] = ("thermal",)
    expected_difficulty: int = 1  # 1 (easiest) .. 5 (hardest)
    # How `tolerance` is interpreted:
    #   band_fraction -- a fraction of the device's reachable beta_N
    #       band. MEASURED TO BE BROKEN: the bands differ by ~600x
    #       across this device set, and achievable error does not scale
    #       with band width because it is set by disturbance and lag,
    #       which are absolute. sparc_like was asked to hold beta_N
    #       inside 15% of everything it can reach while diiid_like got a
    #       tolerance 25x looser than it needed.
    #   floor_multiple -- a multiple of the MEASURED error floor: what a
    #       perfect-knowledge non-anticipating controller achieves on
    #       this device and this setpoint schedule. Equal difficulty by
    #       construction rather than by assumption.
    tolerance_mode: str = "band_fraction"
    # "band_fraction": `setpoint.base`, `setpoint.amplitude` and `tolerance` are
    #     fractions of the device's MEASURED reachable beta_N band, resolved to
    #     absolute values by `resolve_for`. This is the default because the
    #     alternative silently makes the task a different difficulty on every
    #     device -- see `devices/registry.BETA_N_BANDS`.
    # "absolute": the fields are beta_N directly. Correct for a single-device
    #     study, or when a specific beta_N is the object of interest; wrong for
    #     anything that compares devices.
    setpoint_mode: str = "band_fraction"

    def __post_init__(self) -> None:
        if self.setpoint_mode not in ("band_fraction", "absolute"):
            raise ValueError(f"unknown setpoint_mode {self.setpoint_mode!r}")
        if self.setpoint_mode == "band_fraction":
            if not 0.0 < self.setpoint.base < 1.0:
                raise ValueError(
                    f"{self.name}: in band_fraction mode setpoint.base is a "
                    f"fraction of the reachable band and must be in (0, 1); "
                    f"got {self.setpoint.base}"
                )
            if not 0.0 < self.tolerance < 1.0:
                raise ValueError(
                    f"{self.name}: in band_fraction mode tolerance is a "
                    f"fraction of the band WIDTH and must be in (0, 1); got "
                    f"{self.tolerance}"
                )
            # The WHOLE schedule must stay inside the band, not just its base.
            # `base` alone passing while `base + amplitude` runs past 1.0 is the
            # same unreachable-setpoint bug one level up: the excursions would
            # be the part the policy cannot track, and only on the devices with
            # the narrowest bands. Checked here so it fails at import, not after
            # a training run.
            traj = self.setpoint.trajectory(
                np.linspace(0.0, self.episode_length, 200), self.episode_length
            )
            span = (float(traj.min()) - self.tolerance,
                    float(traj.max()) + self.tolerance)
            if not (0.0 < span[0] and span[1] < 1.0):
                raise ValueError(
                    f"{self.name}: the setpoint schedule plus tolerance spans "
                    f"{span[0]:.3f}..{span[1]:.3f} of the reachable band, which "
                    "leaves it. Every point a policy is asked to track must be "
                    "inside (0, 1); reduce base, amplitude or tolerance."
                )

    @property
    def steps_per_shot(self) -> int:
        return int(round(self.episode_length / self.delta_t_a))

    def resolve_for(self, device, limits=None) -> "TaskSpec":
        """Bind fractional setpoint/tolerance to one device's measured band.

        Returns an ``absolute``-mode copy, so the result is an ordinary TaskSpec
        that every downstream consumer (the reward, the gates, the plots) reads
        in beta_N without knowing about normalisation. Already-absolute specs
        are returned unchanged, so this is safe to call unconditionally.
        """
        if self.setpoint_mode == "absolute":
            return self

        from hfmarl.devices.registry import beta_N_band
        from hfmarl.envs.limits import DEFAULT_LIMITS, LimitSet

        limits = limits if limits is not None else LimitSet(DEFAULT_LIMITS)
        beta = next((l for l in limits.limits if l.name == "beta_N"), None)
        lo, hi = beta_N_band(
            device.name if hasattr(device, "name") else str(device),
            task=self.name,
            soft_limit=beta.soft if beta is not None else None,
        )
        width = hi - lo

        return replace(
            self,
            setpoint=replace(
                self.setpoint,
                base=lo + self.setpoint.base * width,
                amplitude=self.setpoint.amplitude * width,
            ),
            tolerance=self._resolve_tolerance(device, width),
            setpoint_mode="absolute",
            name=self.name,
        )

    def _resolve_tolerance(self, device, width: float) -> float:
        """Absolute beta_N tolerance for this device."""
        if self.tolerance_mode == "band_fraction":
            return self.tolerance * width
        if self.tolerance_mode == "floor_multiple":
            from hfmarl.devices.registry import tracking_floor

            name = device.name if hasattr(device, "name") else str(device)
            return self.tolerance * tracking_floor(name, self.name)
        raise ValueError(
            f"unknown tolerance_mode {self.tolerance_mode!r}; expected "
            "band_fraction or floor_multiple")

    def config_overrides(self) -> dict:
        return {
            "transport_model": self.transport_model,
            "enable_fusion": self.enable_fusion,
        }

    def harder(self) -> "TaskSpec":
        """The next preset up, or self if already hardest.

        Used by the headroom gate to escalate automatically rather than making
        someone guess which knob to turn next.
        """
        order = DIFFICULTY_ORDER
        try:
            i = order.index(self.name)
        except ValueError:
            return self
        return PRESETS[order[min(i + 1, len(order) - 1)]]

    def easier(self) -> "TaskSpec":
        order = DIFFICULTY_ORDER
        try:
            i = order.index(self.name)
        except ValueError:
            return self
        return PRESETS[order[max(i - 1, 0)]]

    def for_band_measurement(self) -> "TaskSpec":
        """An absolute copy whose setpoint is a placeholder.

        BREAKS A CIRCULAR DEPENDENCY. `resolve_for` needs the device's
        measured beta_N band, the band is produced by
        `scripts/gate_authority.py`, and that script builds an env -- which
        resolves the task. So a NEW task could never be measured: the error
        told you to run the tool that could not run.

        A command sweep does not use the setpoint. It drives each actuator
        level and reads back beta_N; the target and tolerance are never
        consulted. So for that one purpose the setpoint can be anything, and
        marking the spec absolute makes `resolve_for` a no-op.

        Not for anything else: a task returned from here has a meaningless
        target, and scoring a controller against it would be scoring it
        against 1.0 beta_N for no reason.
        """
        return replace(
            self,
            setpoint=replace(self.setpoint, base=1.0, amplitude=0.0),
            tolerance=1.0,
            setpoint_mode="absolute",
            name=self.name,
        )

    def with_tolerance(self, tol: float) -> "TaskSpec":
        return replace(self, tolerance=tol, name=f"{self.name}_tol{tol:g}")


# ---------------------------------------------------------------------------
# Presets, easiest to hardest.
#
# The headroom gate sweeps these. Expect the first one or two to FAIL headroom
# (isolated converges too fast to be interesting) and the last to risk failing
# measurability (isolated never converges). The usable task is in between, and
# which one it is depends on the device -- so the gate measures per device.
#
# EVERY NUMBER BELOW IS A FRACTION, not a beta_N. `setpoint.base` and
# `setpoint.amplitude` are fractions of the device's measured reachable band and
# `tolerance` a fraction of that band's WIDTH; `TaskSpec.resolve_for` turns them
# into beta_N for one device. Absolute values here would make every preset a
# different difficulty on each device -- 0.15 in beta_N is 7.5% of DIII-D's
# control authority and 197% of SPARC's. See `devices/registry.BETA_N_BANDS`.
#
# base = 0.5 puts the target mid-band, so the policy has authority on BOTH
# sides of it. That matters more than it looks: a target at the top of the band
# is reachable only at saturation, which is a bang-bang problem with no
# interior optimum, and reads as a learning failure rather than a task one.
# ---------------------------------------------------------------------------

PRESETS: dict[str, TaskSpec] = {
    "trivial": TaskSpec(
        name="trivial",
        setpoint=SetpointSchedule("constant", base=0.5),
        tolerance=0.30,
        transport_model="constant",
        enable_fusion=False,
        episode_length=10.0,
        delta_t_a=1.0,
        expected_difficulty=1,
    ),
    "easy": TaskSpec(
        name="easy",
        setpoint=SetpointSchedule("constant", base=0.5),
        tolerance=0.15,
        transport_model="constant",
        enable_fusion=True,
        episode_length=10.0,
        delta_t_a=0.5,
        expected_difficulty=2,
    ),
    "moderate": TaskSpec(
        name="moderate",
        setpoint=SetpointSchedule("ramp", base=0.30, amplitude=0.45),
        tolerance=0.12,
        transport_model="constant",
        enable_fusion=True,
        episode_length=15.0,
        delta_t_a=0.5,
        disturbance_std=0.05,
        expected_difficulty=3,
    ),
    "hard": TaskSpec(
        name="hard",
        setpoint=SetpointSchedule("steps", base=0.55, amplitude=0.30, n_steps=4),
        tolerance=0.10,
        transport_model="qlknn",
        enable_fusion=True,
        episode_length=20.0,
        delta_t_a=0.5,
        disturbance_std=0.10,
        expected_difficulty=4,
    ),
    "brutal": TaskSpec(
        name="brutal",
        # Target near the TOP of the reachable band, so the optimal and the
        # safe action genuinely differ and the limits must be planned for
        # rather than merely avoided. `resolve_for` caps the band at the beta_N
        # soft edge, so this cannot become unsatisfiable the way an absolute
        # setpoint above the limit would -- see `validate_against_limits`.
        setpoint=SetpointSchedule("steps", base=0.52, amplitude=0.36, n_steps=5),
        tolerance=0.08,
        transport_model="qlknn",
        enable_fusion=True,
        episode_length=25.0,
        delta_t_a=0.5,
        disturbance_std=0.15,
        clusters=("thermal", "particle"),
        expected_difficulty=5,
    ),
    # NOT a rung on the difficulty ladder -- a different axis, and the only
    # one grounded in a measurement rather than in turning knobs up.
    #
    # Driving each device from zero to full command showed that beta_N is the
    # only limit anything can cross, and that only diiid_like (above ~60%
    # command) and tcv_like (above ~40%) can cross it. `brink` puts the
    # setpoint ramp just under that crossing, so tracking it quickly and
    # staying inside Troyon pull in opposite directions: too little gain and
    # the ramp is not followed inside tolerance, too much and the overshoot
    # crosses the limit. A fixed-gain PI has ONE knob for both.
    #
    # Everything else is copied from `moderate` deliberately -- constant
    # transport, thermal cluster, 15 s, fusion on -- because `hard` and
    # `brutal` turned out to be ill-posed rather than hard (band-fraction
    # tolerances below actuator resolution under qlknn), and the point here is
    # to change exactly one thing.
    #
    # On iter_like and sparc_like this preset is just `moderate` with a higher
    # setpoint: their limits are unreachable, so there is no brink to sit on.
    "brink": TaskSpec(
        name="brink",
        setpoint=SetpointSchedule("ramp", base=0.45, amplitude=0.45),
        tolerance=0.08,
        transport_model="constant",
        enable_fusion=True,
        episode_length=15.0,
        delta_t_a=0.5,
        disturbance_std=0.05,
        expected_difficulty=4,
    ),
}

DIFFICULTY_ORDER: tuple[str, ...] = ("trivial", "easy", "moderate", "hard", "brutal")

# Where to start a headroom sweep. Anything below this is almost certainly too
# easy to show a federation benefit worth reporting.
DEFAULT_TASK = "moderate"


def get(name: str) -> TaskSpec:
    if name not in PRESETS:
        # sorted(PRESETS), not DIFFICULTY_ORDER: `brink` is a preset but not a
        # rung on the ladder, and an error that omits an existing task reads
        # as "no such task" when the task is right there.
        raise KeyError(f"unknown task {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_against_limits(task: TaskSpec, limits=None) -> list[str]:
    """Check the setpoint schedule is actually achievable inside the envelope.

    A setpoint above the beta limit makes tracking and safety strictly
    contradictory: every policy must either miss the target or cross the line,
    so the Phase 1 gate can never pass and the reward is measuring the
    trade-off rather than control quality. That is an UNSATISFIABLE task, not
    a hard one, and it is an easy mistake to make while turning the difficulty
    up in response to a headroom failure.

    Returns a list of problems; empty means fine.
    """
    from hfmarl.envs.limits import DEFAULT_LIMITS, LimitSet

    limits = limits or LimitSet(DEFAULT_LIMITS)
    beta = next((l for l in limits.limits if l.name == "beta_N"), None)
    if beta is None:
        return []

    times = np.linspace(0.0, task.episode_length, 200)
    traj = task.setpoint.trajectory(times, task.episode_length)
    hi, lo = float(traj.max()), float(traj.min())

    problems: list[str] = []
    if hi > beta.hard:
        problems.append(
            f"setpoint reaches beta_N = {hi:.2f}, above the HARD limit "
            f"{beta.hard:.2f}. Tracking would require a violation; no policy "
            "can satisfy both objectives."
        )
    elif hi > beta.soft:
        problems.append(
            f"setpoint reaches beta_N = {hi:.2f}, past the soft edge "
            f"{beta.soft:.2f}. Permanently inside the penalty band, so the "
            "tracking signal is masked by the limit penalty. Intentional only "
            "if the trade-off itself is the object of study."
        )
    if hi + task.tolerance > beta.hard:
        problems.append(
            f"setpoint plus tolerance ({hi + task.tolerance:.2f}) exceeds the "
            f"hard limit {beta.hard:.2f}; the on-target band straddles the line."
        )
    if lo <= 0:
        problems.append(f"setpoint falls to {lo:.2f}; beta_N must stay positive.")
    return problems
