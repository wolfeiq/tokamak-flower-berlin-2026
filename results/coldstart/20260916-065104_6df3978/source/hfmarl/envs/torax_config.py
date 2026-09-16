"""Build TORAX config dicts from a Device.

SCHEMA WARNING
--------------
This targets **TORAX 1.4.3 exactly**, which is why pyproject.toml pins it.
Post-1.4.3 ``main`` replaced the flat transport block

    'transport': {'model_name': 'constant'}

with a registry

    'transport': {'core_transport_models': {'prescribed': {...}}}

so an unpinned upgrade will fail validation here rather than silently
misbehave -- which is the good outcome, but do not be surprised by it.
Field names below were read from the 1.4.3 sources and the bundled
``iterhybrid_rampup.py``; see TORAX_NOTES.md.

WHY DENSITY IS SET AS A GREENWALD FRACTION
------------------------------------------
Initial and boundary densities use ``n_e_nbar_is_fGW`` / ``n_e_ped_is_fGW``,
i.e. they are specified as fractions of each device's own Greenwald limit
rather than in m^-3. That makes the four devices start in *physically
comparable* states despite their densities differing by an order of magnitude
in SI. Federation compares plasmas, not numbers, and this is where that starts.

...AND WHY TEMPERATURE NOW IS TOO
---------------------------------
Density was normalised and temperature was not. ``T_i_ped``/``T_e_ped`` were a
single hard-coded 1.0 keV for every device, and ``initial_T_keV`` a single 6.0.
Because

    beta_N ~ n T a / (B_0 I_p)   and   n = f_GW * I_p / (pi a^2)
        =>  beta_N ~ f_GW * T / (a * B_0)

one absolute temperature puts four devices at normalised pressures spanning a
factor of 30. Measured at zero auxiliary power, end of a 10 s shot:

    device       a*B_0    beta_N
    iter_like    10.60     0.149
    sparc_like    6.95     0.178
    diiid_like    1.34     0.741
    tcv_like      0.36     3.780      <- above the beta_N hard limit of 3.0

So `tcv_like` began every episode already in violation and `iter_like` could not
reach beta_N = 1 at full power. The single beta_N = 2.0 setpoint in
``envs/task.py`` was outside three of the four bands, which is not a hard
control problem but an absent one. See FINDINGS.md "Gate 0d".

The fix is the same one density already gets: specify the temperature as a
device-relative quantity. Equal pedestal beta requires ``T_ped ~ a * B_0``, and
anchoring that on ITER's real pedestal reproduces every device's published
value without any per-device tuning --

    iter_like   4.50 keV   (ITER baseline H-mode ~4.5)
    sparc_like  2.95 keV   (SPARC V2 predictions ~3-5)
    diiid_like  0.57 keV   (DIII-D H-mode ~0.5-1)
    tcv_like    0.15 keV   (TCV ~0.1-0.3)

-- which is the reason to believe the scaling rather than the convenience of it.
``scripts/gate_authority.py`` is the gate that checks the result.

The pedestal is the lever that matters, not the initial condition. Over a 10 s
shot the plasma relaxes to the pedestal-supported equilibrium, so end-of-shot
beta_N is essentially independent of ``initial_T_keV`` (measured: T_init 6 keV
with T_ped 4 keV gives beta_N 0.409; T_init 24 keV with the same pedestal gives
0.410). ``initial_T_keV`` is scaled along with it anyway, so the starting
profile is consistent with the pedestal it relaxes onto rather than being a
discontinuity the solver has to absorb.
"""

from __future__ import annotations

from typing import Any

from hfmarl.devices.registry import Device

# Reference point for the thermal scaling: ITER's minor radius x toroidal field,
# and the pedestal temperature a real ITER baseline H-mode runs at.
#
# Anchoring here is not arbitrary -- ITER is the device whose pedestal is the
# best predicted of the four, and the resulting scale reproduces the other
# three's published pedestals (see the module docstring). Moving the anchor
# rescales every device together, which is a deliberate global knob rather than
# per-device tuning.
_REF_A_B: float = 2.0 * 5.3          # ITER a_minor * B_0  [m T]
_REF_T_PED_KEV: float = 4.5          # ITER baseline H-mode pedestal  [keV]

# Core temperature at t=0, as a multiple of the pedestal. The plasma relaxes off
# this within a shot, so it sets the transient rather than the final state; 3x
# gives a peaked profile consistent with the pedestal it relaxes onto.
_CORE_OVER_PED: float = 3.0

# Separatrix boundary condition, as a fraction of the pedestal. Small, and it
# must scale with the pedestal or the edge gradient changes meaning per device.
_EDGE_OVER_PED: float = 0.05


def thermal_scale(device: Device) -> float:
    """Device temperature scale, relative to ITER.

    ``beta_N ~ f_GW * T / (a * B_0)`` once density is expressed as a Greenwald
    fraction, so equal normalised pressure across devices requires the
    temperature to scale as ``a * B_0``. See the module docstring for the
    measurement this was derived from and the published pedestals it
    reproduces.
    """
    return (device.a_minor * device.B_0) / _REF_A_B


def pedestal_T_keV(device: Device) -> float:
    """Pedestal temperature for this device, in keV."""
    return _REF_T_PED_KEV * thermal_scale(device)


def build_config(
    device: Device,
    *,
    t_final: float = 10.0,
    n_rho: int = 25,
    transport_model: str = "constant",
    solver_type: str = "linear",
    fixed_dt: float = 0.1,
    T_ped_keV: float | None = None,
    initial_T_keV: float | None = None,
    edge_T_keV: float | None = None,
    initial_fGW: float = 0.5,
    pedestal_fGW: float = 0.4,
    Z_eff: float = 1.6,
    enable_fusion: bool = True,
) -> dict[str, Any]:
    """Assemble the TORAX CONFIG dict for one device.

    Args:
        transport_model: ``"constant"`` is cheap and smooth -- correct for the
            Phase 0 gates and for debugging, because it makes timings
            interpretable. ``"qlknn"`` is the physically meaningful choice and
            what any reported result should use; it is a neural surrogate for
            gyrokinetic fluxes and is markedly slower to compile.
        solver_type: ``"linear"`` for speed, ``"newton_raphson"`` for accuracy
            and stiff cases. Both are JAX_STATIC -- switching either mid-run
            forces a recompile, so pick one per experiment.
        fixed_dt: internal solver step. The RL action window (``delta_t_a``)
            is separate and must be an integer multiple of this.
        T_ped_keV: pedestal temperature. ``None`` derives it from the device
            via ``pedestal_T_keV`` -- which is the point, and what makes the
            four devices comparable in normalised pressure. Pass a number only
            to study a specific pedestal deliberately.
        initial_T_keV: core temperature at t=0. ``None`` derives it as a fixed
            multiple of the pedestal. A per-shot disturbance scales this (see
            ``ToraxDeviceEnv.reset``), which is why it stays overridable.
        edge_T_keV: separatrix boundary condition. ``None`` derives it as a
            fixed fraction of the pedestal.

    Note ``enable_fast_ions`` and ``adaptive_dt`` are left at their defaults:
    both are JAX_STATIC and irrelevant to control -- but ``adaptive_dt`` must
    be OFF on the differentiable path; see ``identification/torax_inverse.py``.
    """
    if n_rho < 4:
        raise ValueError("TORAX requires at least 4 radial cells")
    if t_final <= 0 or fixed_dt <= 0:
        raise ValueError("t_final and fixed_dt must be positive")

    # Device-relative thermal conditions. Everything below is anchored on the
    # pedestal, so overriding `initial_T_keV` alone (as the disturbance path
    # does) perturbs the starting profile without moving the equilibrium the
    # plasma relaxes onto.
    T_ped = pedestal_T_keV(device) if T_ped_keV is None else float(T_ped_keV)
    if T_ped <= 0:
        raise ValueError(f"T_ped_keV must be positive, got {T_ped}")
    if initial_T_keV is None:
        initial_T_keV = _CORE_OVER_PED * T_ped
    if edge_T_keV is None:
        edge_T_keV = _EDGE_OVER_PED * T_ped
    if not 0 < edge_T_keV < initial_T_keV:
        raise ValueError(
            f"need 0 < edge_T_keV ({edge_T_keV}) < initial_T_keV "
            f"({initial_T_keV}); an inverted profile is not a plasma"
        )

    sources: dict[str, Any] = {
        # Ohmic heating and the bootstrap current are always on: they are
        # plasma self-organisation, not actuators.
        "ohmic": {},
        "ei_exchange": {"Qei_multiplier": 1.0},
        # The device's primary auxiliary heating, driven by the `aux_heat`
        # actuator: NBI on ITER/DIII-D/TCV, ICRF on SPARC.
        "generic_heat": {
            "P_total": 0.0,
            "gaussian_location": 0.3,
            "gaussian_width": 0.2,
            "electron_heat_fraction": 0.5,
        },
        # Electron cyclotron heating. Driven by the `ecrh` actuator. This is
        # the second and only other independent heat source in 1.4.3.
        "ecrh": {
            "P_total": 0.0,
            "gaussian_location": 0.1,
            "gaussian_width": 0.1,
        },
        # Non-inductive current drive, held fixed until the current cluster
        # (Phase 3) takes it over.
        "generic_current": {
            "fraction_of_total_current": 0.15,
            "gaussian_location": 0.36,
            "gaussian_width": 0.075,
        },
        # Particle sources. Driven by the `gas_puff` actuator.
        "gas_puff": {"S_total": 0.0, "puff_decay_length": 0.3},
        "generic_particle": {
            "S_total": 0.0,
            "deposition_location": 0.3,
            "particle_width": 0.25,
        },
        "pellet": {
            "S_total": 0.0,
            "pellet_width": 0.1,
            "pellet_deposition_location": 0.85,
        },
    }
    if enable_fusion:
        # Alpha heating. On for ITER/SPARC-like burning plasmas; it is the
        # nonlinearity that makes thermal control genuinely hard, so leaving
        # it off would make the problem easier than reality.
        sources["fusion"] = {}

    return {
        "plasma_composition": {
            "main_ion": {"D": 0.5, "T": 0.5},
            "impurity": "Ne",
            "Z_eff": Z_eff,
        },
        "profile_conditions": {
            "Ip": device.Ip_nominal,
            "T_i": {0.0: {0.0: initial_T_keV, 1.0: edge_T_keV}},
            "T_i_right_bc": edge_T_keV,
            "T_e": {0.0: {0.0: initial_T_keV, 1.0: edge_T_keV}},
            "T_e_right_bc": edge_T_keV,
            # Density in Greenwald-fraction units -- see module docstring.
            "n_e_nbar_is_fGW": True,
            "nbar": initial_fGW,
            "n_e": {0: {0.0: 1.5, 1.0: 1.0}},  # shape only; nbar sets the level
            "n_e_right_bc_is_fGW": True,
            "n_e_right_bc": 0.5 * initial_fGW,
        },
        "numerics": {
            "t_initial": 0.0,
            "t_final": t_final,
            "fixed_dt": fixed_dt,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": True,
            "resistivity_multiplier": 1,
        },
        "geometry": {
            "geometry_type": "circular",
            "R_major": device.R_major,
            "a_minor": device.a_minor,
            "B_0": device.B_0,
            "elongation_LCFS": device.elongation,
            "n_rho": n_rho,
        },
        "neoclassical": {"bootstrap_current": {"bootstrap_multiplier": 1.0}},
        "sources": sources,
        "pedestal": {
            "model_name": "set_T_ped_n_ped",
            "set_pedestal": True,
            "T_i_ped": T_ped,
            "T_e_ped": T_ped,
            "n_e_ped_is_fGW": True,
            "n_e_ped": pedestal_fGW,
            "rho_norm_ped_top": 0.9,
        },
        "transport": {"model_name": transport_model},
        # `use_pereverzev` adds the Pereverzev-Corrigan inner-loop terms that
        # keep a linear solver stable against stiff transport. TORAX warns for
        # it explicitly whenever the linear solver meets a stiff model:
        #
        #   "use_pereverzev=False in a configuration where setting
        #    use_pereverzev=True is recommended."
        #
        # It fired on every `qlknn` shot and was being ignored. It is off for
        # `constant` transport, where it is unnecessary and would add numerical
        # diffusion for nothing, and on for anything stiff.
        "solver": {
            "solver_type": solver_type,
            "use_pereverzev": solver_type == "linear" and transport_model != "constant",
        },
        # 'fixed' pairs with numerics.fixed_dt and makes every action window an
        # exact integer number of solver steps, which keeps step timings
        # comparable. 'chi' adapts dt to the transport and is more efficient
        # but makes per-step wall clock vary, muddying the JIT gate.
        "time_step_calculator": {"calculator_type": "fixed"},
    }
