"""TORAX as a steppable environment.

UNVERIFIED AGAINST A LIVE TORAX. Written from the 1.4.3 sources and from
Gym-TORAX's working implementation, but never executed -- the machine it was
written on cannot run TORAX. ``scripts/gate0_env.py`` and
``scripts/gate0_jit.py`` exist to find out whether it is right. Run them
first.

The call sequence below is the one the TORAX lead endorsed for RL use in
github.com/google-deepmind/torax/discussions/1625:

    step_fn   = make_step_fn(torax_config)
    state, pp = get_initial_state_and_post_processed_outputs(t, step_fn)
    provider  = step_fn.runtime_params_provider.update_provider_from_mapping(u)
    step_fn   = SimulationStepFn(solver=..., runtime_params_provider=provider, ...)
    state, pp = step_fn.jitted_fixed_time_step(dt, state, pp)

Two clocks, kept deliberately distinct (SPEC.md §8 asks for explicit clock
separation, not emergent):

    fixed_dt     the solver's internal timestep, set in the TORAX config
    delta_t_a    the ACTION window -- how long one RL step lasts

``delta_t_a`` should be an integer multiple of ``fixed_dt``. One env step
advances the plasma by ``delta_t_a``, internally taking several solver steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hfmarl.devices.registry import Device
from hfmarl.envs.actuators import ActuatorBank, bank_from_device
from hfmarl.envs.limits import DEFAULT_LIMITS, LimitReport, LimitSet
from hfmarl.envs.task import PRESETS, TaskSpec, validate_against_limits
from hfmarl.envs.torax_config import build_config


# Pseudo-limit name for a solver blow-up, so it is counted alongside real
# limit crossings in the violation metrics rather than being invisible.
SOLVER_FAILURE = "solver_failure"

# A TORAX state carrying NaN/inf in the quantities the limits are written
# against. Distinct from SOLVER_FAILURE so the two causes stay separable in the
# logs, but treated identically: an unusable state is a crashed plasma.
NONFINITE_STATE = "nonfinite_state"


class ToraxUnavailableError(RuntimeError):
    """TORAX is not installed or failed to import. Raised with install advice."""


def _import_torax():
    """Import TORAX lazily.

    Deferred because importing TORAX pulls in JAX and costs seconds, and
    because everything else in this package -- device definitions, the
    dimensionless encoder, the similarity metric, the limits -- must stay
    importable and testable on a machine with no TORAX at all.
    """
    try:
        import torax
        from torax._src.state import SimError
        from torax.experimental import (
            RuntimeParamsProvider,
            SimulationStepFn,
            get_initial_state_and_post_processed_outputs,
            make_step_fn,
        )
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ToraxUnavailableError(
            f"TORAX could not be imported ({e}). Install it with\n"
            "    uv pip install torax==1.4.3\n"
            "and see SETUP.md. Everything in hfmarl outside envs/torax_env.py "
            "works without it."
        ) from e
    return {
        "torax": torax,
        "SimError": SimError,
        "RuntimeParamsProvider": RuntimeParamsProvider,
        "SimulationStepFn": SimulationStepFn,
        "get_initial_state": get_initial_state_and_post_processed_outputs,
        "make_step_fn": make_step_fn,
    }


@dataclass
class StepResult:
    """One env step's worth of everything worth keeping.

    SPEC.md §8 requires full state trajectories per episode, because
    cross-device comparisons cannot be reconstructed afterwards. This is the
    record that gets logged.
    """

    t: float
    ok: bool
    done: bool
    scalars: dict[str, float] = field(default_factory=dict)
    profiles: dict[str, np.ndarray] = field(default_factory=dict)
    actuators: dict[str, float] = field(default_factory=dict)
    target: float = float("nan")  # setpoint in force at this step
    wall_time: float = 0.0
    error: str = ""
    # Limit names crossed at this step, or the pseudo-limits SOLVER_FAILURE /
    # NONFINITE_STATE. Carried on the record rather than only in the `info`
    # dict because `trajectory` is what gets logged, and a violation that is
    # not on the trajectory cannot be reconstructed afterwards -- which is
    # exactly what the violations-during-training curve is computed from.
    violations: tuple[str, ...] = ()


# Scalars pulled from PostProcessedOutputs. Every one is a direct field -- none
# is derived here, so what the policy is graded on is exactly what TORAX
# computed. Names verified against TORAX 1.4.3 post_processing.py.
SCALAR_FIELDS: tuple[str, ...] = (
    "q95",
    "li3",
    "beta_N",
    "beta_tor",
    "beta_pol",
    "f_bootstrap",
    "f_non_inductive",
    "fgw_n_e_line_avg",
    "n_e_line_avg",
    "n_e_volume_avg",
    "W_thermal_total",
    "tau_E",
    "H98",
    "Q_fusion",
    "P_heat_total",
    "P_SOL_total",
    "P_aux_total",
    "P_alpha_total",
)

# Profiles pulled from SimState.core_profiles. CellVariable fields expose
# `.value`; q_face and s_face are plain arrays on the face grid.
PROFILE_FIELDS: tuple[str, ...] = ("T_i", "T_e", "n_e", "psi")
FACE_FIELDS: tuple[str, ...] = ("q_face", "s_face")


class ToraxCore:
    """Lifecycle and stepping for one TORAX simulation.

    Deliberately free of any RL vocabulary -- no rewards, no observation
    spaces. Those belong to the Gymnasium layer above, and keeping them out
    means this class can also drive an open-loop scan or a PID baseline.
    """

    def __init__(self, config_dict: dict[str, Any], delta_t_a: float):
        if delta_t_a <= 0:
            raise ValueError("delta_t_a must be positive")
        fixed_dt = config_dict.get("numerics", {}).get("fixed_dt")
        if fixed_dt:
            ratio = delta_t_a / fixed_dt
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"delta_t_a ({delta_t_a}) should be an integer multiple of "
                    f"numerics.fixed_dt ({fixed_dt}); got ratio {ratio:.4f}. "
                    "A fractional ratio leaves a cropped final substep whose "
                    "cost varies between steps and confuses the JIT timing gate."
                )
        self.config_dict = config_dict
        self.delta_t_a = float(delta_t_a)
        self._tx: dict[str, Any] | None = None
        self._started = False
        self.t = 0.0
        # Populated properly by start(), but seeded here from the config so
        # `_observe()` is safe to call before the first reset() -- policies are
        # sized from the observation shape, which happens before any rollout
        # (gate_headroom does exactly this).
        self.t_initial = float(config_dict.get("numerics", {}).get("t_initial", 0.0))
        self.t_final = float(config_dict.get("numerics", {}).get("t_final", 1.0))
        self.state = None
        self.post = None
        self.compile_seconds: float | None = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Build the TORAX config, solver and step function. Compiles nothing yet."""
        tx = self._tx = _import_torax()
        self.torax_config = tx["torax"].ToraxConfig.from_dict(self.config_dict)
        self.step_fn = tx["make_step_fn"](self.torax_config)
        self.t_initial = float(self.torax_config.numerics.t_initial)
        self.t_final = float(self.torax_config.numerics.t_final)
        self._init_state, self._init_post = tx["get_initial_state"](
            t=self.t_initial, step_fn=self.step_fn
        )
        self._started = True

    def reset(self) -> None:
        """Return to initial conditions and a pristine runtime-parameter provider.

        The provider is rebuilt from the original config rather than mutated
        back, because ``numerics.t_final`` is advanced on every step and a
        stale value would end the next episode early. Rebuilding does not
        recompile: the pytree structure and all leaf shapes are unchanged, so
        the cached executable still applies.
        """
        if not self._started:
            self.start()
        tx = self._tx
        self.state, self.post = self._init_state, self._init_post
        self.t = self.t_initial
        self.step_fn = tx["SimulationStepFn"](
            solver=self.step_fn.solver,
            time_step_calculator=self.step_fn.time_step_calculator,
            runtime_params_provider=tx["RuntimeParamsProvider"].from_config(
                self.torax_config
            ),
            geometry_provider=self.torax_config.geometry.build_provider,
        )

    # -- stepping --------------------------------------------------------

    def apply_updates(self, updates: dict[str, Any]) -> None:
        """Patch runtime parameters for the coming action window.

        ``numerics.t_final`` must track the end of the window. The advance
        length is set by the ``dt`` handed to ``jitted_fixed_time_step``, but
        TORAX compares ``t + dt`` against ``numerics.t_final`` when handling
        the final, possibly cropped, substep -- so leaving it at the episode
        end changes the solver's retry behaviour. Gym-TORAX does the same.
        """
        if not self._started:
            raise RuntimeError("call start() or reset() before apply_updates()")
        tx = self._tx
        payload = dict(updates)
        payload["numerics.t_final"] = float(self.t + self.delta_t_a)
        provider = self.step_fn.runtime_params_provider.update_provider_from_mapping(
            payload
        )
        self.step_fn = tx["SimulationStepFn"](
            solver=self.step_fn.solver,
            time_step_calculator=self.step_fn.time_step_calculator,
            runtime_params_provider=provider,
            geometry_provider=self.step_fn.geometry_provider,
        )

    def advance(self) -> tuple[bool, bool, str]:
        """Advance one action window. Returns (ok, episode_done, error_text)."""
        if self.state is None:
            raise RuntimeError("call reset() before advance()")
        tx = self._tx
        t0 = time.perf_counter()
        try:
            state, post = self.step_fn.jitted_fixed_time_step(
                self.delta_t_a, self.state, self.post
            )
            sim_error = self.step_fn.check_for_errors(state, post)
        except Exception as e:  # pragma: no cover - needs a live solver
            # A solver blow-up is a normal outcome of a bad policy, not a bug.
            # The episode ends; training continues.
            return False, True, f"exception during step: {e}"
        finally:
            self._last_wall = time.perf_counter() - t0

        if sim_error != tx["SimError"].NO_ERROR:
            return False, True, f"TORAX SimError: {sim_error}"

        self.state, self.post = state, post
        self.t += self.delta_t_a
        return True, self.t >= self.t_final - 1e-9, ""

    # -- observation -----------------------------------------------------

    def read_scalars(self) -> dict[str, float]:
        """Scalars straight off PostProcessedOutputs.

        Missing names are skipped rather than defaulted -- a silently absent
        safety quantity is exactly how a controller comes to look safe when it
        is not. ``gate0_env.py`` reports which of SCALAR_FIELDS actually exist.
        """
        out: dict[str, float] = {}
        for name in SCALAR_FIELDS:
            v = getattr(self.post, name, None)
            if v is not None:
                out[name] = float(np.asarray(v).reshape(-1)[0])
        return out

    def read_profiles(self) -> dict[str, np.ndarray]:
        cp = self.state.core_profiles
        out: dict[str, np.ndarray] = {}
        for name in PROFILE_FIELDS:
            var = getattr(cp, name, None)
            if var is None:
                continue
            out[name] = np.asarray(getattr(var, "value", var), dtype=float)
        for name in FACE_FIELDS:
            var = getattr(cp, name, None)
            if var is not None:
                out[name] = np.asarray(var, dtype=float)
        return out

    def nonfinite_scalars(self) -> tuple[str, ...]:
        """Names of limit-relevant scalars that came back NaN or inf.

        TORAX can produce non-finite values when the solver is struggling.
        These must NOT be quietly dropped: `LimitSet.evaluate` skips missing
        keys by design, so a NaN beta_N that silently became "absent" would be
        scored as SAFE -- precisely the failure the limits module exists to
        prevent. Detected here and escalated by the env.
        """
        out = []
        for name in ("beta_N", "q95", "fgw_n_e_line_avg"):
            v = getattr(self.post, name, None)
            if v is None:
                continue
            if not np.isfinite(np.asarray(v, dtype=float)).all():
                out.append(name)
        return tuple(out)

    def limit_inputs(self) -> dict[str, float]:
        """Map TORAX scalar names onto the names the LimitSet expects."""
        s = {k: v for k, v in self.read_scalars().items() if np.isfinite(v)}
        out = {}
        if "fgw_n_e_line_avg" in s:
            out["greenwald_fraction"] = s["fgw_n_e_line_avg"]
        if "beta_N" in s:
            out["beta_N"] = s["beta_N"]
        if "q95" in s:
            out["q95"] = s["q95"]
        return out


class ToraxDeviceEnv:
    """A device, its actuators, and its operating envelope.

    Gymnasium-compatible in shape (``reset`` / ``step`` returning the 5-tuple)
    without importing gymnasium at module scope, so it stays usable in a plain
    loop. ``as_gym_env()`` wraps it when the real Space objects are wanted.

    Phase 1 uses this directly with the thermal cluster.
    """

    def __init__(
        self,
        device: Device,
        *,
        task: TaskSpec | str = "moderate",
        clusters: tuple[str, ...] | None = None,
        limits: LimitSet | None = None,
        config_overrides: dict[str, Any] | None = None,
        limit_penalty: float = 5.0,
        violation_penalty: float = 50.0,
        seed: int | None = None,
        strict_task_check: bool = True,
        action_cap: float | None = None,
    ):
        """A device, its actuators, its operating envelope and its task.

        `task` sets the difficulty -- setpoint schedule, tolerance, transport
        model, episode length. It is an experimental variable, not a constant:
        if isolated training converges too fast there is no headroom for
        federation to show anything, so `scripts/gate_headroom.py` sweeps these
        presets and this constructor takes whichever one it selected.
        """
        self.device = device
        self.limits = limits if limits is not None else LimitSet(DEFAULT_LIMITS)
        # Bind the task to THIS device before anything reads it. Presets carry
        # the setpoint and tolerance as fractions of the device's measured
        # reachable beta_N band; `resolve_for` turns them into beta_N. Every
        # consumer below (the reward, the observation, the gates) then sees an
        # ordinary absolute task and needs to know nothing about normalisation.
        spec = PRESETS[task] if isinstance(task, str) else task
        self.task = spec.resolve_for(device, self.limits)

        # A setpoint above the beta limit makes tracking and safety strictly
        # contradictory. That is unsatisfiable rather than hard, and it is an
        # easy mistake to make when turning difficulty up after a headroom
        # failure -- so it is caught at construction, not after a training run.
        problems = validate_against_limits(self.task, self.limits)
        if problems and strict_task_check:
            raise ValueError(
                f"task {self.task.name!r} is not satisfiable on this envelope:\n  "
                + "\n  ".join(problems)
                + "\n(pass strict_task_check=False if the trade-off is the object "
                "of study)"
            )
        self.task_warnings = problems

        self.bank: ActuatorBank = bank_from_device(
            device, clusters if clusters is not None else self.task.clusters
        )
        # AUDIT #4. Ceiling on every unit action, applied above the policy.
        # None means no interlock.
        self.action_cap = None if action_cap is None else float(action_cap)
        self.limit_penalty = float(limit_penalty)
        self.violation_penalty = float(violation_penalty)
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        overrides = dict(self.task.config_overrides())
        overrides.update(config_overrides or {})
        self._base_overrides = overrides
        cfg = build_config(device, t_final=self.task.episode_length, **overrides)
        self.core = ToraxCore(cfg, self.task.delta_t_a)
        self._config_cache: dict[float, dict] = {}
        self._last_control_cost: float = 0.0
        self.trajectory: list[StepResult] = []
        self.shot_index: int = -1
        self.federated_updates_received: int = 0

    @property
    def target_beta_N(self) -> float:
        """Current target. Time-varying for every task above `easy`."""
        return self.task.setpoint.target(
            self.core.t, getattr(self.core, "t_final", self.task.episode_length)
        )

    @property
    def n_actions(self) -> int:
        return len(self.bank)

    def reset(self) -> tuple[np.ndarray, dict]:
        """Start a new shot.

        A per-shot disturbance perturbs the initial temperature, so a policy
        must be robust rather than having memorised one trajectory. It is
        applied by rebuilding the config, which does NOT recompile: initial
        conditions are traced leaves of unchanged shape.
        """
        self.shot_index += 1
        if self.task.disturbance_std > 0:
            factor = float(
                np.clip(self.rng.normal(1.0, self.task.disturbance_std), 0.5, 1.5)
            )
            # Quantised so the same perturbation recurs. Rebuilding the config
            # means a full `ToraxConfig.from_dict` pydantic validation plus a
            # fresh `make_step_fn`, and the headroom gate runs thousands of
            # shots -- paying that per shot is a real cost for no extra
            # robustness. Rounding to 0.01 gives ~60 distinct initial states
            # across the clip range, plenty of diversity, each validated once.
            key = round(factor, 2)
            if key not in self._config_cache:
                # Perturb the device's OWN nominal core temperature, not an
                # absolute 6 keV. A fixed 6.0 here meant the disturbance was a
                # +-5% wobble on ITER and a factor of 13 on TCV, i.e. a
                # different experiment on every device -- the same mistake the
                # hard-coded pedestal made. `build_config` derives the nominal
                # from the pedestal, so ask it rather than restating the number.
                nominal_core = float(
                    build_config(self.device, t_final=self.task.episode_length,
                                 **self._base_overrides)
                    ["profile_conditions"]["T_i"][0.0][0.0]
                )
                self._config_cache[key] = build_config(
                    self.device,
                    t_final=self.task.episode_length,
                    initial_T_keV=nominal_core * key,
                    **self._base_overrides,
                )
            cfg = self._config_cache[key]
            if cfg is not self.core.config_dict:
                self.core.config_dict = cfg
                self.core._started = False  # rebuild against this perturbation
        self.core.reset()
        self.bank.reset()
        self.trajectory = []
        # Seed the running control cost from the initial state, so a crash on
        # step 1 is still billed at a real rate rather than at zero.
        self._last_control_cost = self._control_cost(
            self.limits.evaluate(self.core.limit_inputs())
        )
        return self._observe(), {
            "device": self.device.name,
            "shot": self.shot_index,
            "task": self.task.name,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # AUDIT #4. An OPERATIONAL INTERLOCK, not a training trick. A
        # held-out-region experiment that only lowers the setpoint does not
        # hold anything out: the search still explores the full envelope
        # and the target visits the region it is supposed never to have
        # seen. Real machines enforce this in hardware, above the
        # controller, which is exactly where this sits.
        if self.action_cap is not None:
            action = np.minimum(np.asarray(action, float), self.action_cap)
        values = self.bank.from_unit_actions(action)
        self.core.apply_updates(self.bank.build_updates(values, self.core.t, self.core.delta_t_a))
        ok, done, err = self.core.advance()

        if not ok:
            # Solver failure is a catastrophic outcome, not a neutral one: a
            # policy that drives the plasma somewhere the solver cannot follow
            # has not found a loophole, it has crashed the plasma.
            #
            # Two things this must do, and originally did neither:
            #   * charge the escaped remainder, exactly as a limit violation
            #     does. Otherwise crashing the SOLVER is the cheap exit that
            #     crashing into a LIMIT no longer is, and the policy simply
            #     learns the other door.
            #   * report it as a violation, so it appears in the
            #     violations-during-training curve. Left out, a condition that
            #     crashes more often would score as SAFER.
            reward = -self.violation_penalty
            reward += self._early_exit_charge()
            self.trajectory.append(
                StepResult(t=self.core.t, ok=False, done=True, error=err,
                           violations=(SOLVER_FAILURE,))
            )
            return (
                self._observe(), reward, True, False,
                {"error": err, "violations": (SOLVER_FAILURE,)},
            )

        bad = self.core.nonfinite_scalars()
        if bad:
            # An unusable state is a crashed plasma, not a safe one. Falling
            # through would let LimitSet skip the NaN quantity and score the
            # step as within limits.
            reward = -self.violation_penalty
            reward += self._early_exit_charge()
            msg = f"non-finite {', '.join(bad)}"
            self.trajectory.append(
                StepResult(t=self.core.t, ok=False, done=True, error=msg,
                           violations=(NONFINITE_STATE,))
            )
            return (
                self._observe(), reward, True, False,
                {"error": msg, "violations": (NONFINITE_STATE,)},
            )

        report = self.limits.evaluate(self.core.limit_inputs())
        reward = self._reward(report)
        if report.any_violated:
            reward += self._early_exit_charge()
        scalars = self.core.read_scalars()

        self.trajectory.append(
            StepResult(
                t=self.core.t,
                ok=True,
                done=done,
                scalars=scalars,
                profiles=self.core.read_profiles(),
                actuators=dict(zip(self.bank.names, values)),
                target=self.target_beta_N,
                wall_time=getattr(self.core, "_last_wall", 0.0),
                violations=report.violations,
            )
        )
        terminated = bool(report.any_violated)
        info = {
            "limits": report,
            "scalars": scalars,
            "violations": report.violations,
        }
        return self._observe(), reward, terminated, done and not terminated, info

    def _control_cost(self, report: LimitReport) -> float:
        """Per-step cost of control quality alone -- no terminal penalties.

        Kept separate from `_reward` because the early-exit charge needs the
        rate the policy was incurring, and deriving that by subtracting the
        terminal penalty back out of the reward is what broke the solver-failure
        path: that path passes exactly `-violation_penalty`, so the subtraction
        yielded zero and the charge never fired.
        """
        s = self.core.read_scalars()
        beta = s.get("beta_N", 0.0)
        err = abs(beta - self.target_beta_N)
        tracking = -(max(0.0, err - self.task.tolerance) ** 2)
        soft = -self.limit_penalty * max(0.0, 1.0 - report.worst_margin) ** 2
        cost = tracking + soft
        if not np.isfinite(cost):
            return 0.0
        # Floor the per-step cost. Tracking cost is quadratic in the error, so a
        # single excursion (beta_N briefly in the hundreds before the limit
        # terminates the shot) produces a value many orders of magnitude larger
        # than a normal step and swamps the episode return, the learning curve
        # and every statistic downstream.
        #
        # The floor is HALF the violation penalty, not equal to it. The
        # no-free-exit guarantee needs the per-step cost to stay strictly above
        # -violation_penalty: terminating early is worse than finishing by
        # exactly (per_step + violation_penalty), which would go to zero if the
        # floor met the penalty.
        return float(max(cost, -0.5 * self.violation_penalty))

    def _reward(self, report: LimitReport) -> float:
        """Setpoint tracking, penalised by proximity to the operating limits.

        Three terms, deliberately simple so Phase 1's gate is interpretable:
          * squared error on normalised beta outside a tolerance deadband --
            the tracking objective. The deadband means "close enough" is not
            punished, which keeps difficulty controlled by `task.tolerance`;
          * a soft penalty that grows once any limit margin falls below 1,
            giving the policy a gradient *before* the cliff rather than only
            at it;
          * a large terminal penalty for an actual violation.
        """
        cost = self._control_cost(report)
        self._last_control_cost = cost
        hard = -self.violation_penalty if report.any_violated else 0.0
        return float(cost + hard)

    def _early_exit_charge(self) -> float:
        """Cost of the episode the policy just escaped by terminating early.

        THE BUG THIS FIXES. Both a limit violation and a solver failure end the
        episode. Without this term a policy that cannot track pays a per-step
        cost for the full episode but only a one-off `violation_penalty` for
        crashing out on step 1 -- and at the beta_N error an untrained policy
        actually starts from (~1.9), crashing scores BETTER on every task
        preset. The agent learns to drive into the limit, or into the solver,
        on purpose. That would corrupt the violations-during-training curve,
        which carries the safety half of the claim.

        Fix: bill the remaining steps at the control cost the policy was
        already incurring (`_last_control_cost`, maintained on every step and
        seeded at reset so it exists even for a step-1 crash). The episode then
        costs what continuing would have cost, plus `violation_penalty` -- so
        terminating is always strictly worse, by exactly that penalty,
        independent of task length, of when it happens, and of WHICH exit was
        taken.
        """
        steps_taken = len(self.trajectory) + 1
        remaining = max(0, self.task.steps_per_shot - steps_taken)
        if remaining == 0:
            return 0.0
        return float(remaining * min(self._last_control_cost, 0.0))

    def _observe(self) -> np.ndarray:
        """Observation vector.

        Phase 1 observes SI-ish scalars directly. Phase 4 replaces this with
        the dimensionless encoder -- that swap is the experiment, so the
        method lives here to be overridden rather than being inlined.
        """
        s = self.core.read_scalars()
        target = self.target_beta_N
        return np.array(
            [
                s.get("beta_N", 0.0),
                # The tracking error against the CURRENT target. Without this
                # the policy cannot see a moving setpoint at all and every task
                # above `easy` is unsolvable for reasons that look like a
                # learning failure.
                s.get("beta_N", 0.0) - target,
                target,
                s.get("q95", 0.0),
                s.get("fgw_n_e_line_avg", 0.0),
                s.get("li3", 0.0),
                s.get("H98", 0.0),
                s.get("f_bootstrap", 0.0),
                self.core.t / max(self.core.t_final, 1e-9),
            ],
            dtype=np.float32,
        )
