"""Device definitions for the federation.

Each device is a federation client. Heterogeneity between them is the
experimental variable, not an accident -- see SPEC.md Phase 0.

GEOMETRY CAVEAT (important, read before trusting any result)
------------------------------------------------------------
All devices use TORAX's ``circular`` geometry, because it is the only
geometry that is pure-parameter -- every other option (CHEASE, EQDSK, FBT,
IMAS) needs an equilibrium file, and TORAX ships only ITER-hybrid and STEP.
There is no SPARC equilibrium in existence that we can legally use.

TORAX's own docstring for ``_build_circular_geometry`` says "used for testing
only". It assumes r/a_minor = rho_norm and supports no shaping beyond
elongation -- no triangularity, no divertor, no real flux surfaces.

Consequences you must not forget:
  * "SPARC-like" here means *a compact high-field circular plasma with
    SPARC's R, a, B_0*. It is NOT SPARC. Say so in any write-up.
  * TORAX refuses edge/SOL models on circular geometry
    (``_check_edge_with_circular_geometry`` raises). The exhaust cluster
    (SPEC.md Phase 7) is therefore unreachable from this file. It needs real
    equilibria first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Actuator:
    """One controllable input, with its device-specific engineering limit."""

    name: str
    torax_path: str  # dot-path into the TORAX config, e.g. sources.ecrh.P_total
    lo: float  # SI, minimum commandable value
    hi: float  # SI, maximum commandable value
    cluster: str  # thermal | particle | current -- sets the federation channel
    units: str
    # Value the actuator holds at the start of a shot. MUST equal whatever
    # `torax_config.build_config` writes at `torax_path`, or step 1 issues a
    # ramp from a value the plasma was never at. For Ip that meant a 90% current
    # drop on the first action. `tests/test_devices.py` enforces the match.
    initial: float | None = None  # None => lo
    # False => excluded from the default action space. Used for actuators that
    # have no independent TORAX source and would silently collide with another
    # actuator's config path. See the ICRH note below.
    available: bool = True

    def clip(self, value: float) -> float:
        return float(min(max(value, self.lo), self.hi))

    @property
    def start_value(self) -> float:
        return self.lo if self.initial is None else self.initial


@dataclass(frozen=True)
class Device:
    """A tokamak, as far as this study is concerned.

    Machine parameters are public nominal values. They define the circular
    geometry and the actuator envelopes; everything else TORAX derives.
    """

    name: str
    R_major: float  # m, major radius
    a_minor: float  # m, minor radius
    B_0: float  # T, vacuum toroidal field on axis
    elongation: float  # kappa at the LCFS
    Ip_nominal: float  # A, flat-top plasma current
    actuators: tuple[Actuator, ...]
    note: str = ""

    # --- derived machine quantities -------------------------------------

    @property
    def aspect_ratio(self) -> float:
        return self.R_major / self.a_minor

    @property
    def inverse_aspect_ratio(self) -> float:
        """epsilon = a/R. Enters the trapped-particle fraction and nu_star."""
        return self.a_minor / self.R_major

    @property
    def greenwald_density(self) -> float:
        """Greenwald density limit in m^-3.

        n_GW [1e20 m^-3] = I_p [MA] / (pi * a[m]^2)   (Greenwald 1988)

        TORAX reports the Greenwald *fraction* directly as
        ``fgw_n_e_line_avg``, so this is here for device characterisation and
        for sanity-checking TORAX, not for use in the reward.
        """
        ip_MA = self.Ip_nominal / 1e6
        return (ip_MA / (math.pi * self.a_minor**2)) * 1e20

    @property
    def q_cylindrical(self) -> float:
        """Cylindrical safety factor, a zeroth-order q_95 proxy.

            q_cyl = [2*pi*a^2*B_0 / (mu_0 * R * I_p)] * (1 + kappa^2)/2

        The elongation correction is **(1+kappa^2)/2**, the standard shaping
        factor (Wesson, *Tokamaks*; ITER Physics Basis). An earlier version
        used a bare `kappa`, which is not a standard form and ran ~15% low --
        for ITER it gave q = 2.80 where the standard form gives 3.22.

        Still an approximation: it omits triangularity and the finite-aspect-
        ratio correction [(1.17 - 0.65*eps)/(1-eps^2)^2], so it sits perhaps
        20-30% below a real q95. Circular geometry has no triangularity to
        supply, so this is the right level of detail here.

        Used only to place a device in dimensionless space BEFORE a simulation
        exists. Once TORAX is running, use its own ``q95``, which is computed
        from the actual flux surfaces.
        """
        mu_0 = 4e-7 * math.pi
        cylindrical = (
            2 * math.pi * self.a_minor**2 * self.B_0
            / (mu_0 * self.R_major * self.Ip_nominal)
        )
        return cylindrical * (1.0 + self.elongation**2) / 2.0

    def actuators_for(
        self, cluster: str, include_unavailable: bool = False
    ) -> tuple[Actuator, ...]:
        return tuple(
            a
            for a in self.actuators
            if a.cluster == cluster and (include_unavailable or a.available)
        )

    def actuator(self, name: str) -> Actuator:
        for a in self.actuators:
            if a.name == name:
                return a
        raise KeyError(f"{self.name} has no actuator {name!r}")


# ---------------------------------------------------------------------------
# Actuator sets
#
# SIMPLIFICATION, stated openly: every device is given the same three heating
# actuators and the same gas puff, with device-specific power limits. Real
# machines differ -- SPARC's baseline auxiliary heating is ICRF only, TCV is
# ECRH-dominated and has no ICRH at all.
#
# We keep the actuator *set* uniform so that role-matched federation has
# identically-shaped policy networks to average, which is what SPEC.md §2
# requires. Making the sets genuinely ragged is a real research question
# (heterogeneous action spaces under federation) and is deliberately out of
# scope here. Where a device physically lacks a system, its limit is set low
# rather than the actuator being removed, and the note says so.
# ---------------------------------------------------------------------------


def _standard_actuators(
    *, p_aux: float, p_ecrh: float, p_icrh: float, gas_puff: float, ip: float
) -> tuple[Actuator, ...]:
    """Build the standard actuator set.

    HEAT SOURCE REALITY CHECK (TORAX 1.4.3)
    ---------------------------------------
    TORAX 1.4.3 ships exactly two independent auxiliary heat sources we can
    drive: ``generic_heat`` (a Gaussian deposition, which Gym-TORAX and the
    bundled ITER configs use to represent NBI) and ``ecrh`` (the electron
    cyclotron source, with its own deposition location and width).

    SPEC.md Phase 2 asks for three thermal agents (NBI, ECRH, ICRH). There is
    no third source. Pointing ICRH at ``generic_heat`` as well would make two
    agents write the same config path, and their commanded powers would be
    summed or one would overwrite the other -- a controller bug that looks
    like a physics result.

    So ICRH is defined but marked ``available=False``: it is excluded from the
    default action space and documented, rather than quietly aliased. Phase 2
    therefore runs with a 2-agent thermal cluster unless and until a genuine
    ICRH source is added (TORAX has an ion-cyclotron source in newer versions;
    check before assuming, and note it would break the 1.4.3 pin).

    WHY THE FIRST ACTUATOR IS ``aux_heat`` AND NOT ``nbi``
    -----------------------------------------------------
    It used to be ``nbi``, and that composed with the ICRH decision above into
    a device with no heating at all. SPARC's baseline auxiliary heating is
    ~25 MW of ICRF and its NBI is nominally zero, so ``p_nbi=0.1e6`` was right
    -- and then the 25 MW was switched off as ``icrh``, leaving 5.1 MW of ECRH
    against B_0 = 12.2 T. Measured: full-scale command moved beta_N by 0.012.
    The device was uncontrollable, and both decisions that made it so were
    individually documented and individually correct.

    ``generic_heat`` is not "the neutral beam". It is a generic Gaussian
    deposition, and what it represents here is *the device's primary auxiliary
    heating system*: NBI on ITER, DIII-D and TCV, ICRF on SPARC. Naming it
    ``aux_heat`` is what makes role-matched federation honest -- channel 0 then
    means the same thing on every device, which is precisely what SPEC.md 2
    requires and what a per-device name would have quietly broken. Each
    device's ``note`` says which physical system its channel is.
    """
    return (
        Actuator("aux_heat", "sources.generic_heat.P_total", 0.0, p_aux,
                 "thermal", "W"),
        Actuator("ecrh", "sources.ecrh.P_total", 0.0, p_ecrh, "thermal", "W"),
        Actuator(
            "icrh",
            "sources.generic_heat.P_total",
            0.0,
            p_icrh,
            "thermal",
            "W",
            available=False,  # no independent source in 1.4.3 -- see docstring
        ),
        Actuator("gas_puff", "sources.gas_puff.S_total", 0.0, gas_puff, "particle", "s^-1"),
        # `initial=ip` because build_config starts the plasma at Ip_nominal.
        # Without it the first ramp would start from the lower bound (0.1*Ip)
        # and drop the plasma current by 90% on step 1.
        Actuator("ip", "profile_conditions.Ip", 0.1 * ip, ip, "current", "A",
                 initial=ip),
    )


DEVICES: dict[str, Device] = {
    "iter_like": Device(
        name="iter_like",
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        elongation=1.72,
        Ip_nominal=10.5e6,
        actuators=_standard_actuators(
            p_aux=33e6, p_ecrh=20e6, p_icrh=20e6, gas_puff=1e22, ip=10.5e6
        ),
        note="ITER nominal. The large, low-rho_star, low-nu_star corner.",
    ),
    "sparc_like": Device(
        name="sparc_like",
        R_major=1.85,
        a_minor=0.57,
        B_0=12.2,
        elongation=1.97,
        Ip_nominal=8.7e6,
        actuators=_standard_actuators(
            p_aux=25e6, p_ecrh=5e6, p_icrh=25e6, gas_puff=5e21, ip=8.7e6
        ),
        note=(
            "SPARC V2 nominal R/a/B_0 on CIRCULAR geometry -- not a SPARC "
            "equilibrium. Compact and very high field: reaches ITER-like "
            "rho_star at a fraction of the size, which is exactly the "
            "similarity overlap Phase 6 needs. Its aux_heat channel is ICRF "
            "(~25 MW), not a neutral beam -- SPARC's NBI is nominally zero."
        ),
    ),
    "diiid_like": Device(
        name="diiid_like",
        R_major=1.67,
        a_minor=0.67,
        B_0=2.0,
        elongation=1.8,
        Ip_nominal=1.5e6,
        actuators=_standard_actuators(
            p_aux=20e6, p_ecrh=6e6, p_icrh=2e6, gas_puff=2e21, ip=1.5e6
        ),
        note=(
            "DIII-D nominal. The PACMAN device (arXiv 2511.08818) -- included "
            "so results can be positioned against published RL control work. "
            "Low aspect ratio (A=2.5)."
        ),
    ),
    "tcv_like": Device(
        name="tcv_like",
        R_major=0.88,
        a_minor=0.25,
        B_0=1.44,
        elongation=1.5,
        Ip_nominal=0.25e6,
        actuators=_standard_actuators(
            p_aux=1.3e6, p_ecrh=4.5e6, p_icrh=0.1e6, gas_puff=5e20, ip=0.25e6
        ),
        note=(
            "TCV nominal. Small, hot-electron, ECRH-dominated: the high-rho_star, "
            "high-nu_star corner. Deliberately the FURTHEST device in "
            "dimensionless space -- it is the negative control for the "
            "similarity-weighted aggregation in SPEC.md §4b. If distance "
            "weighting works, TCV should contribute weakly to ITER."
        ),
    ),
}


def get(name: str) -> Device:
    if name not in DEVICES:
        raise KeyError(f"unknown device {name!r}; have {sorted(DEVICES)}")
    return DEVICES[name]


def all_devices() -> tuple[Device, ...]:
    return tuple(DEVICES.values())


# ---------------------------------------------------------------------------
# Representative flat-top operating points, per device.
#
# Nominal conditions, NOT simulation output. Used to place each device in
# dimensionless space before any TORAX run exists -- for the Phase 5 bandwidth
# check and the similarity-ablation figure.
#
# Replace with measured values once TORAX runs: the similarity weighting is
# computed on the OPERATING POINT, not on the machine, so two devices can be
# near or far depending on how they are being run.
# ---------------------------------------------------------------------------

OPERATING_POINTS: dict[str, dict[str, float]] = {
    # VOLUME-AVERAGED, not core. This matters and was wrong before.
    #
    # `encode` computes beta from whatever n_e and T it is given. Fed core
    # values it returns a CORE beta -- roughly 1.8x the volume-averaged beta_N
    # that TORAX reports and that the operating limits are written against. The
    # two would then be different physical quantities sharing a name, so a
    # device's position in similarity space before a run would not match its
    # position during one.
    #
    # Coherence also matters for nu*: `encode` pairs these kinetics with q95
    # and eps = a/R, which are global/edge quantities. Volume-averaged n and T
    # make that a consistent "global" collisionality; core values would mix
    # core kinetics with edge geometry.
    #
    # Values below reproduce published beta_N for each device (ITER 1.6,
    # SPARC ~0.9, DIII-D ~2.3, TCV ~1.2) at the Ip in this registry.
    "iter_like": {"T_e_keV": 8.0, "T_i_keV": 8.0, "n_e": 0.7e20},
    "sparc_like": {"T_e_keV": 7.0, "T_i_keV": 7.0, "n_e": 3.0e20},
    "diiid_like": {"T_e_keV": 2.5, "T_i_keV": 2.5, "n_e": 0.5e20},
    # TCV is ECRH-dominated: hot electrons, much colder ions.
    "tcv_like": {"T_e_keV": 1.0, "T_i_keV": 0.4, "n_e": 0.3e20},
}


# ---------------------------------------------------------------------------
# MEASURED reachable beta_N, per device AND per task.
#
# Unlike OPERATING_POINTS below, these are simulation output: end-of-shot beta_N
# at zero and at full auxiliary command, from `scripts/gate_authority.py`.
# Re-measure after ANY change to the thermal scaling, the actuator envelopes,
# the transport model or the episode length, and paste the new numbers here --
# they are what `envs/task.py` resolves a setpoint against.
#
# WHY THIS EXISTS. The task setpoint used to be one absolute number
# (beta_N = 2.0) for all four devices, and the tolerance likewise (0.15). Both
# are wrong for a heterogeneous set, for the same reason and independently:
#
#   * beta_N ~ f_GW * T / (a * B_0), so the reachable band differs per device;
#     2.0 sat outside three of the four.
#   * control authority scales as P * a / (B_0 * R * I_p * kappa), so an
#     absolute tolerance is 7.5% of DIII-D's authority and 197% of SPARC's.
#     Even with a reachable setpoint, the task would be a different difficulty
#     on every device -- and "shots to threshold" would then be measuring
#     device calibration rather than learning, which is exactly the quantity
#     SPEC.md 5 compares across the federation.
#
# WHY IT IS KEYED BY TASK TOO. The band is a property of the plant, and the
# task sets part of the plant: `transport_model` above all (`constant` vs
# `qlknn` moves every band substantially), plus fusion, episode length and
# which actuator clusters are live. Keying by device alone made `brutal`
# resolve against `easy`'s band and fail 3 of 4 devices for a reason that
# looked identical to the original bug. This is not circular -- the band
# depends only on the CONFIG half of a task, never on its setpoint.
#
# Both are expressed as fractions of the band below. See
# `envs/task.TaskSpec.resolve_for`.
# Measured 2026-09-15 on TORAX 1.4.3 / JAX 0.11.1, CPU. These are the RAW
# reachable extremes -- the shot is run to completion at fixed command without
# stopping at a limit crossing, so the upper end can sit above the operating
# envelope. `beta_N_band(soft_limit=...)` caps it at resolve time; keeping the
# raw value here means the table measures actuator AUTHORITY and the envelope
# stays a separate, changeable decision.
#
# Worth reading before trusting a result: with `qlknn` transport (the `hard`
# and `brutal` presets) `iter_like` has a band of 0.806..0.816 -- an authority
# of 0.011 in beta_N against 0.444 under `constant`. That is profile stiffness
# doing what profile stiffness does: above the critical gradient, extra power
# raises transport instead of temperature. It is physics, not a bug, but it
# means the stiff presets are near-uncontrollable on the largest device, and
# `scripts/gate_authority.py` will say so.
BETA_N_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "iter_like": {
        "trivial": (0.4526, 0.817),
        "easy": (0.4594, 0.9038),
        "moderate": (0.4589, 0.9037),
        "hard": (0.8055, 0.8164),
        "brutal": (0.8036, 0.861),
        "brink": (0.4590, 0.9040),
    },
    "sparc_like": {
        "trivial": (0.4018, 0.476),
        "easy": (0.4028, 0.4789),
        "moderate": (0.4027, 0.4789),
        "hard": (0.7045, 0.8371),
        "brutal": (0.7016, 0.8693),
        "brink": (0.4030, 0.4790),
    },
    "diiid_like": {
        "trivial": (0.4992, 4.5112),
        "easy": (0.4992, 5.4329),
        "moderate": (0.4992, 7.4953),
        "hard": (0.7649, 1.5226),
        "brutal": (0.7604, 1.5261),
        "brink": (0.4992, 3.1900),
    },
    "tcv_like": {
        "trivial": (0.6564, 10.9327),
        "easy": (0.6564, 7.3489),
        "moderate": (0.6564, 7.3507),
        "hard": (0.7124, 3.2741),
        "brutal": (0.6993, 3.1483),
        "brink": (0.6564, 6.7400),
    },
}


# WHETHER A LIMIT CAN BE CROSSED AT ALL, measured by driving each device from
# zero to full thermal command on `easy` and taking the worst margin over the
# whole episode (scratchpad probe `probe_limit_reach.py`; the table is in
# FINDINGS.md under "Measured: only beta_N is crossable").
#
#   value = worst beta_N margin at FULL command. +1 is comfortably safe, 0 is
#   exactly at the hard limit, negative means the limit was crossed.
#
# This exists because the catastrophe experiment -- "can a device avoid a
# regime it has never entered?" -- is vacuous on a device with no reachable
# regime to avoid, and its --target defaulted to exactly such a device. On
# iter_like and sparc_like, no available action sequence can fail containment,
# so the run would report perfect safety and mean nothing by it.
#
# q95 and the Greenwald fraction are omitted deliberately: neither moves with
# command on any device, because Ip lives in the unactuated `current` cluster
# and density is pinned by the pedestal and edge boundary condition. The
# three-limit envelope is a one-limit envelope.
BETA_N_MARGIN_AT_FULL_COMMAND: dict[str, float] = {
    "iter_like": 4.17,
    "sparc_like": 5.04,
    "diiid_like": -0.38,
    "tcv_like": -7.48,
}


def can_violate(name: str) -> bool:
    """True if driving this device hard is capable of crossing a limit.

    A safety experiment on a device for which this is False tests nothing.
    """
    if name not in BETA_N_MARGIN_AT_FULL_COMMAND:
        raise KeyError(
            f"no measured limit reachability for device {name!r}; drive it "
            "from zero to full command and record the worst beta_N margin in "
            "BETA_N_MARGIN_AT_FULL_COMMAND")
    return BETA_N_MARGIN_AT_FULL_COMMAND[name] < 0.0


# Measured OPEN-LOOP reference per (device, task): the mean
# |beta_N - target| a perfect static feedforward attains, commanding at
# every step the level the calibration sweep says holds the current
# target in steady state.
#
# It is NOT a lower bound, and an earlier version of this comment said it
# was. Feedback beats it: on `moderate`, tcv_like's PI controller reaches
# 0.0373 against an open-loop 0.1189. What it IS is the free performance
# the plant hands you from a sweep, so a controller that cannot beat it
# is adding nothing.
#
# Written by `scripts/gate_authority.py --task <name>`, which reports it
# as 'tracking floor'. A task using tolerance_mode='floor_multiple' with
# a multiplier BELOW 1 is asking for better than non-anticipating
# control can deliver, which is exactly where a policy that sees the
# setpoint schedule has something to earn.
TRACKING_FLOORS: dict[str, dict[str, float]] = {}


def tracking_floor(device: str, task: str) -> float:
    """The measured open-loop reference error, or a clear failure.

    Raises rather than defaulting: a task whose tolerance is a multiple
    of an unmeasured floor would silently become a task with an
    arbitrary tolerance, which is the defect this mode exists to fix.
    """
    if device not in TRACKING_FLOORS or task not in TRACKING_FLOORS[device]:
        raise KeyError(
            f"no measured tracking floor for device {device!r} on task "
            f"{task!r}. Run `scripts/gate_authority.py --task {task}` and "
            "record the floor it reports -- a floor-multiple tolerance "
            "without a measured floor is an arbitrary number.")
    return float(TRACKING_FLOORS[device][task])


def beta_N_band(
    name: str, task: str = "easy", soft_limit: float | None = None
) -> tuple[float, float]:
    """Usable beta_N band for a device on a task: (zero command, full command).

    ``soft_limit`` caps the upper end. Pass the envelope's beta_N soft edge so
    the resolved setpoint stays inside the safe region -- `diiid_like` and
    `tcv_like` can both drive past the hard limit, and a setpoint up there would
    make tracking and safety contradictory, which `validate_against_limits`
    exists to refuse.
    """
    if name not in BETA_N_BANDS:
        raise KeyError(
            f"no measured beta_N band for device {name!r}; run "
            "scripts/gate_authority.py and record it in BETA_N_BANDS"
        )
    per_task = BETA_N_BANDS[name]
    if task not in per_task:
        raise KeyError(
            f"no measured beta_N band for device {name!r} on task {task!r}; "
            f"have {sorted(per_task)}. Run "
            f"`scripts/gate_authority.py --task {task}` and record it -- the "
            "band moves with the transport model, so another task's band is "
            "not a substitute."
        )
    lo, hi = per_task[task]
    if soft_limit is not None:
        hi = min(hi, float(soft_limit))
    if hi <= lo:
        raise ValueError(
            f"{name} on {task}: usable beta_N band is empty "
            f"({lo:.3f}..{hi:.3f}). The device cannot be controlled anywhere "
            "inside its safe envelope."
        )
    return lo, hi


def operating_point(name: str) -> dict[str, float]:
    if name not in OPERATING_POINTS:
        raise KeyError(f"no operating point for device {name!r}")
    return dict(OPERATING_POINTS[name])


def encoded_states() -> dict:
    """Every device's nominal operating point, in dimensionless space."""
    from hfmarl.physics.dimensionless import encode

    return {
        d.name: encode(
            q95=d.q_cylindrical, B_0=d.B_0, R_major=d.R_major,
            a_minor=d.a_minor, Ip=d.Ip_nominal, **operating_point(d.name),
        )
        for d in all_devices()
    }
