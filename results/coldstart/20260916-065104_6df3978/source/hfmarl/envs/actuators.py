"""Actuator commands and the no-recompilation contract.

THE ONE THING TO GET RIGHT
--------------------------
TORAX stays JIT-compiled across steps only if the JAX pytree *structure* and
every leaf *shape* stay constant. Runtime parameters are traced leaves (TORAX
config models register as pytrees and only ``JAX_STATIC``-annotated fields are
compile-time constants), so changing a heating power is free -- but changing
the number of time-breakpoints in the waveform is not, because that changes an
array shape and triggers a full recompile.

The fix, which Gym-TORAX (MIT, arXiv:2510.11283) discovered and which we adopt
with attribution, is to make every actuator update a **two-breakpoint linear
ramp**:

    time  = [t, t + dt]
    value = [previous_value, new_value]

Two breakpoints, always, for every actuator, on every step. The shape never
changes, so the compiled function is reused. As a bonus the plasma sees a
continuous piecewise-linear actuator trajectory rather than a step
discontinuity, which is both more physical and kinder to the solver.

Build a growing waveform array instead and every step pays the compile cost
(tens of seconds). That is the failure mode ``scripts/gate0_jit.py`` exists to
detect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Config paths that TORAX marks JAX_STATIC. Changing any of these mid-episode
# forces a recompile, so none may ever be an action. Verified against TORAX
# 1.4.3 docs/configuration.rst "_trigger_recompilation" plus the source
# annotations; see TORAX_NOTES.md.
STATIC_EXACT: frozenset[str] = frozenset(
    {
        "geometry.n_rho",
        "geometry.hires_factor",
        "numerics.evolve_ion_heat",
        "numerics.evolve_electron_heat",
        "numerics.evolve_current",
        "numerics.evolve_density",
        "numerics.exact_t_final",
        "numerics.adaptive_dt",
        "numerics.enable_fast_ions",
        "transport.model_name",
        "solver.solver_type",
        "time_step_calculator.calculator_type",
    }
)

# Suffixes that are static wherever they appear (e.g. sources.ecrh.mode).
STATIC_SUFFIXES: tuple[str, ...] = (
    ".mode",
    ".is_explicit",
    ".interpolation_mode",
    ".is_bool_param",
)


class StaticParameterError(ValueError):
    """Raised when something that would force a JIT recompile is used as an action."""


def assert_not_static(path: str) -> None:
    """Reject config paths that cannot be changed without recompiling.

    Called in the ``ActuatorSpec`` constructor so the error surfaces when the
    action space is *defined*, not sixty seconds into training when the
    recompile shows up as mysterious slowness.
    """
    if path in STATIC_EXACT or any(path.endswith(s) for s in STATIC_SUFFIXES):
        raise StaticParameterError(
            f"{path!r} is a JAX_STATIC TORAX parameter. Using it as an action "
            "forces a full JIT recompile on every step, which makes training "
            "unusably slow. See TORAX_NOTES.md 'What must never be an action'."
        )


@dataclass(frozen=True)
class ActuatorSpec:
    """One controllable TORAX parameter with its engineering envelope."""

    name: str
    torax_path: str
    lo: float
    hi: float
    cluster: str
    units: str = ""
    scale: float = 1.0  # multiplied in before the value reaches TORAX
    # Where the actuator sits at the start of a shot. Must match the value
    # `build_config` writes at `torax_path`; see devices/registry.Actuator.
    initial: float | None = None  # None => lo

    def __post_init__(self) -> None:
        assert_not_static(self.torax_path)
        if self.hi < self.lo:
            raise ValueError(f"{self.name}: hi ({self.hi}) < lo ({self.lo})")

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.lo, self.hi))

    @property
    def start_value(self) -> float:
        return self.lo if self.initial is None else self.clip(self.initial)

    def from_unit(self, u: float) -> float:
        """Map a policy output in [-1, 1] onto the physical envelope.

        Policies emit bounded, zero-centred actions; actuator ranges are
        wildly different in magnitude (watts vs amps vs particles per second).
        Doing the mapping here keeps every network's output space identical,
        which is a precondition for federating their weights at all.
        """
        u = float(np.clip(u, -1.0, 1.0))
        return self.lo + 0.5 * (u + 1.0) * (self.hi - self.lo)

    def to_unit(self, value: float) -> float:
        if self.hi == self.lo:
            return 0.0
        return float(2.0 * (value - self.lo) / (self.hi - self.lo) - 1.0)


class ActuatorBank:
    """A device's actuators, and the previous command sent to each.

    Holds the ``prev_values`` that make each update a ramp from where the
    actuator actually was rather than from zero.
    """

    def __init__(self, specs: list[ActuatorSpec] | tuple[ActuatorSpec, ...]):
        if not specs:
            raise ValueError("ActuatorBank needs at least one actuator")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate actuator names: {names}")
        self.specs: tuple[ActuatorSpec, ...] = tuple(specs)
        # Start each actuator where the TORAX config actually has it, not at
        # the bottom of its envelope -- otherwise the first ramp is a step
        # change from a state the plasma was never in.
        self.prev_values: np.ndarray = np.array(
            [s.start_value for s in self.specs], float
        )

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def reset(self) -> None:
        self.prev_values = np.array([s.start_value for s in self.specs], float)

    def from_unit_actions(self, actions: np.ndarray) -> np.ndarray:
        """Policy outputs in [-1,1] -> SI values, clipped to the envelope."""
        actions = np.asarray(actions, float).ravel()
        if actions.shape != (len(self.specs),):
            raise ValueError(
                f"expected {len(self.specs)} actions, got {actions.shape}"
            )
        return np.array(
            [s.clip(s.from_unit(a)) for s, a in zip(self.specs, actions)], float
        )

    def build_updates(self, values: np.ndarray, t: float, dt: float) -> dict:
        """Build the two-breakpoint ramp updates for one action window.

        Returns a mapping of TORAX dot-path -> ``TimeVaryingScalarUpdate``,
        ready for ``RuntimeParamsProvider.update_provider_from_mapping``.

        Two actuators may legitimately share a ``torax_path`` (in TORAX 1.4.3
        there is only one generic auxiliary heat source, so aux_heat and icrh
        collide on ``sources.generic_heat.P_total``). Silently letting one
        overwrite the other would mean half the commanded power vanishes with
        no error, so collisions are summed and the caller is expected to know
        -- see the note in devices/registry.py.

        This import is deferred because TORAX takes seconds to import and
        pulls in JAX; the rest of this module must stay usable without it.
        """
        from torax.experimental import TimeVaryingScalarUpdate

        values = np.asarray(values, float).ravel()
        if values.shape != (len(self.specs),):
            raise ValueError(f"expected {len(self.specs)} values, got {values.shape}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        ramps: dict[str, tuple[float, float]] = {}
        for spec, prev, new in zip(self.specs, self.prev_values, values):
            p, n = prev * spec.scale, spec.clip(new) * spec.scale
            if spec.torax_path in ramps:
                op, on = ramps[spec.torax_path]
                ramps[spec.torax_path] = (op + p, on + n)
            else:
                ramps[spec.torax_path] = (p, n)

        updates = {
            path: TimeVaryingScalarUpdate(
                # float64 because TORAX runs in double precision by default.
                time=np.array([t, t + dt], dtype=np.float64),
                value=np.array([p, n], dtype=np.float64),
            )
            for path, (p, n) in ramps.items()
        }

        # The command just issued is where the next ramp starts from.
        self.prev_values = np.array(
            [s.clip(v) for s, v in zip(self.specs, values)], float
        )
        return updates


def bank_from_device(
    device,
    clusters: tuple[str, ...] | None = None,
    include_unavailable: bool = False,
) -> ActuatorBank:
    """Build an ActuatorBank from a Device, optionally limited to some clusters.

    Phase 1 uses ``clusters=("thermal",)``; Phase 3 uses all of them.

    Actuators marked ``available=False`` are excluded by default. In TORAX
    1.4.3 that means ICRH, which has no independent source and would collide
    with NBI on ``sources.generic_heat.P_total`` -- see
    ``devices/registry._standard_actuators``.
    """
    specs = [
        ActuatorSpec(
            name=a.name,
            torax_path=a.torax_path,
            lo=a.lo,
            hi=a.hi,
            cluster=a.cluster,
            units=a.units,
            initial=a.initial,
        )
        for a in device.actuators
        if (clusters is None or a.cluster in clusters)
        and (include_unavailable or a.available)
    ]
    if not specs:
        raise ValueError(f"{device.name} has no actuators in clusters {clusters}")
    return ActuatorBank(specs)
