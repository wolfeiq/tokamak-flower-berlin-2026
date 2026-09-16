"""Regressions for bugs found in the 2026-09-15 audit.

Each test names the bug it prevents and why it mattered. None of these needs
TORAX.
"""

import numpy as np
import pytest

from hfmarl.devices.registry import all_devices, encoded_states, get
from hfmarl.envs.actuators import bank_from_device
from hfmarl.envs.limits import LimitSet
from hfmarl.envs.task import SetpointSchedule, get as get_task
from hfmarl.envs.torax_config import build_config
from hfmarl.envs.torax_env import SOLVER_FAILURE, ToraxDeviceEnv
from hfmarl.federation.similarity import DEFAULT_WEIGHTS, describe_device_set
from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord
from hfmarl.physics.dimensionless import _RHO_S_COEFF, rho_star


def _dig(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        node = node[part]
    return node


# -- C: actuator start values must match the config ----------------------


@pytest.mark.parametrize("device", [d.name for d in all_devices()])
def test_every_actuator_starts_where_the_config_puts_it(device):
    """Bug: Ip's actuator started at its lower bound (0.1*Ip) while the config
    started the plasma at Ip_nominal, so the very first action ramped the
    plasma current down by 90% -- a disruption on step 1 of every episode,
    caused by bookkeeping rather than by the policy."""
    d = get(device)
    cfg = build_config(d)
    bank = bank_from_device(d, include_unavailable=False)
    for spec, start in zip(bank.specs, bank.prev_values):
        cfg_value = float(_dig(cfg, spec.torax_path))
        assert start == pytest.approx(cfg_value, rel=1e-9, abs=1e-12), (
            f"{device}.{spec.name}: starts at {start:.4g} but config has "
            f"{cfg_value:.4g} at {spec.torax_path}"
        )


def test_reset_restores_the_config_start_value_not_the_lower_bound():
    d = get("iter_like")
    bank = bank_from_device(d, ("current",))
    bank.prev_values[0] = 1.0  # simulate a command having been issued
    bank.reset()
    assert bank.prev_values[0] == pytest.approx(d.Ip_nominal)


# -- B: observation must be available before the first reset -------------


def test_observation_works_before_reset():
    """Bug: `_observe()` read `core.t_final`, set only in `start()`, so sizing a
    policy from the observation shape crashed. gate_headroom -- the blocking
    gate -- did exactly that on its first line."""
    env = ToraxDeviceEnv(get("iter_like"), task="moderate")
    obs = env._observe()
    assert obs.ndim == 1 and obs.size > 0
    assert np.all(np.isfinite(obs))


def test_observation_shape_is_stable_across_tasks():
    """Federation averages policies; a device whose task changed the obs width
    would produce an unmergeable network."""
    sizes = {
        ToraxDeviceEnv(get("iter_like"), task=t)._observe().size
        for t in ("easy", "moderate", "hard")
    }
    assert len(sizes) == 1


# -- L + M: a solver crash is a catastrophe, not a cheap exit ------------


def _crash_cost(env, err, exit_at):
    """Total return when the SOLVER fails at `exit_at`.

    The crashing step has no usable scalars, so it pays no control cost of its
    own; the escaped remainder is billed at the last known rate.
    """
    n = env.task.steps_per_shot
    per = -(max(0.0, err - env.task.tolerance) ** 2)
    env._last_control_cost = per
    env.trajectory = [None] * (exit_at - 1)
    return (exit_at - 1) * per + (-env.violation_penalty) + env._early_exit_charge()


def _finish_cost(env, err):
    per = -(max(0.0, err - env.task.tolerance) ** 2)
    return env.task.steps_per_shot * per


def test_solver_failure_is_reported_as_a_violation():
    """Bug: a solver blow-up recorded no violation, so a condition that crashed
    more often scored as SAFER on the violations-during-training curve."""
    r = RunLog("isolated", "iter_like", 0)
    r.add(ShotRecord(shot=0, reward=-50.0, steps=3, violations=(SOLVER_FAILURE,)))
    assert r.violation_flags()[0]
    assert r.cumulative_violations()[-1] == 1


def test_solver_failure_constant_is_not_a_real_limit_name():
    """It must not collide with a limit, or it would corrupt limit bookkeeping."""
    assert SOLVER_FAILURE not in {l.name for l in LimitSet().limits}


@pytest.mark.parametrize("task", ["easy", "hard", "brutal"])
def test_crashing_the_solver_is_not_cheaper_than_finishing(task):
    """Bug: the early-exit charge was applied to limit violations but NOT to
    solver failures, so crashing the solver became the cheap exit that crashing
    into a limit no longer was. The policy just learns the other door."""
    env = ToraxDeviceEnv(get("iter_like"), task=task)
    finish = _finish_cost(env, 1.9)
    for k in (1, env.task.steps_per_shot // 2):
        assert _crash_cost(env, 1.9, k) < finish, (
            f"{task}: crashing at step {k} scores {_crash_cost(env, 1.9, k):.1f} "
            f"vs {finish:.1f} for finishing"
        )


# -- N: never pool reward curves across devices --------------------------


def test_learning_curve_refuses_to_pool_devices():
    """Bug: device=None averaged an ITER-like return with a TCV-like one --
    scales differing by ~100x -- and titled it 'all devices'."""
    pytest.importorskip("matplotlib")
    from hfmarl.metrics import plots

    e = ExperimentLog()
    for dev, val in (("iter_like", -1.0), ("tcv_like", -100.0)):
        r = RunLog("isolated", dev, 0)
        for i in range(10):
            r.add(ShotRecord(shot=i, reward=val, steps=1))
        e.add(r)
    with pytest.raises(ValueError, match="pool"):
        plots.plot_learning_curves(e, device=None)
    with pytest.raises(ValueError, match="pool"):
        plots.plot_violations(e, device=None)


def test_single_device_log_still_plots_without_a_device_argument():
    pytest.importorskip("matplotlib")
    from hfmarl.metrics import plots

    e = ExperimentLog()
    r = RunLog("isolated", "iter_like", 0)
    for i in range(10):
        r.add(ShotRecord(shot=i, reward=-1.0, steps=1))
    e.add(r)
    assert plots.plot_learning_curves(e) is not None


# -- A: diagnostics must survive a degenerate device set -----------------


def test_describe_handles_a_single_device():
    """Bug: min() over an empty off-diagonal raised ValueError."""
    text = describe_device_set({"iter_like": encoded_states()["iter_like"]})
    assert "only one device" in text


def test_describe_handles_an_empty_set():
    assert isinstance(describe_device_set({}), str)


# -- E: the Mach dimension is deliberately inert -------------------------


def test_mach_weight_is_zero_until_rotation_is_modelled():
    """Nothing sets a nonzero Mach number. A nonzero weight is silently inert
    and would start changing every distance the day rotation is wired in."""
    assert DEFAULT_WEIGHTS[4] == 0.0


def test_all_encoded_states_have_zero_mach():
    assert all(s.mach == 0.0 for s in encoded_states().values())


# -- D: the gyroradius constant is derived, not copied --------------------


def test_rho_s_coefficient_matches_the_published_value():
    assert _RHO_S_COEFF == pytest.approx(1.0217e-4, rel=1e-4)


def test_rho_star_still_reproduces_iter():
    assert float(rho_star(10.0, 5.3, 2.0)) == pytest.approx(1.52e-3, rel=0.02)


# -- F: a "restricted" envelope must actually restrict -------------------


def test_restricted_rejects_a_no_op_fraction():
    """Bug: fraction=1.0 was accepted and returned the real envelope, so
    Phase 6's 'safe' device would have met the very limit it must never see."""
    with pytest.raises(ValueError):
        LimitSet().restricted(1.0)


# -- J: setpoint schedules must agree about out-of-episode queries -------


@pytest.mark.parametrize("kind", ["constant", "ramp", "steps", "sine"])
def test_every_schedule_clamps_outside_the_episode(kind):
    """Bug: sine evaluated raw t while ramp/steps clamped, so the same query
    outside the episode behaved differently depending on the schedule."""
    s = SetpointSchedule(kind, base=2.0, amplitude=0.4, n_steps=4, period=5.0)
    assert s.target(-5.0, 10.0) == pytest.approx(s.target(0.0, 10.0))
    assert s.target(50.0, 10.0) == pytest.approx(s.target(10.0, 10.0))


def test_every_preset_stays_positive_and_finite_throughout():
    for name in ("trivial", "easy", "moderate", "hard", "brutal"):
        t = get_task(name)
        traj = t.setpoint.trajectory(np.linspace(0, t.episode_length, 100),
                                     t.episode_length)
        assert np.all(np.isfinite(traj)) and np.all(traj > 0), name


# -- O: the charge must actually fire on the solver path -----------------


def test_early_exit_charge_is_not_structurally_zero_for_a_crash():
    """Bug: the charge derived the per-step rate as `reward + violation_penalty`,
    but the solver path passes exactly `-violation_penalty`, so the rate was
    always 0 and the charge never fired. The previous round's fix covered limit
    violations only; crashing the solver stayed free."""
    env = ToraxDeviceEnv(get("iter_like"), task="hard")
    env._last_control_cost = -3.24  # a real per-step cost
    env.trajectory = []
    assert env._early_exit_charge() < -50.0


def test_running_cost_is_seeded_so_a_step_one_crash_is_still_billed():
    """With no prior step there is no measured rate; reset() must seed one."""
    env = ToraxDeviceEnv(get("iter_like"), task="hard")
    assert hasattr(env, "_last_control_cost")
    assert np.isfinite(env._last_control_cost)


def test_control_cost_excludes_terminal_penalties():
    """It is a rate, not a reward; folding in the one-off penalty is what broke."""
    env = ToraxDeviceEnv(get("iter_like"), task="hard")

    class R:
        worst_margin = 2.0
        any_violated = True

    cost = env._control_cost(R())
    assert cost > -env.violation_penalty


# -- P: non-finite state is a crash, not a safe state --------------------


def test_nonfinite_scalars_are_detected():
    """Bug: LimitSet skips missing keys by design, so a NaN beta_N that became
    'absent' would be scored as SAFE -- the exact failure limits.py warns of."""
    env = ToraxDeviceEnv(get("iter_like"), task="easy")

    class P:
        beta_N = float("nan")
        q95 = 3.0
        fgw_n_e_line_avg = 0.5

    env.core.post = P()
    assert "beta_N" in env.core.nonfinite_scalars()


def test_limit_inputs_never_forward_nonfinite_values():
    env = ToraxDeviceEnv(get("iter_like"), task="easy")

    class P:
        beta_N = float("inf")
        q95 = 3.0
        fgw_n_e_line_avg = 0.5

    env.core.post = P()
    assert "beta_N" not in env.core.limit_inputs()
    assert "q95" in env.core.limit_inputs()


def test_finite_state_reports_nothing():
    env = ToraxDeviceEnv(get("iter_like"), task="easy")

    class P:
        beta_N = 2.0
        q95 = 3.0
        fgw_n_e_line_avg = 0.5

    env.core.post = P()
    assert env.core.nonfinite_scalars() == ()


def test_nonfinite_returns_are_counted_not_silently_censored():
    """A NaN return never clears a threshold, so the run looks like a learning
    failure and gets right-censored into the headline ratio."""
    r = RunLog("isolated", "iter_like", 0)
    for i in range(10):
        r.add(ShotRecord(shot=i, reward=float("nan") if i > 6 else -1.0, steps=1))
    assert r.n_nonfinite() == 3
    e = ExperimentLog()
    e.add(r)
    assert "non-finite" in e.coverage()


def test_clean_runs_produce_no_nonfinite_warning():
    r = RunLog("isolated", "iter_like", 0)
    for i in range(10):
        r.add(ShotRecord(shot=i, reward=-1.0, steps=1))
    e = ExperimentLog()
    e.add(r)
    assert "non-finite" not in e.coverage()


# -- Q: non-finite returns must rank worst in CEM ------------------------


class _NanEnv:
    """An environment that returns NaN on alternate steps."""

    def __init__(self, task_steps=5):
        self.n = 0
        self.task_steps = task_steps

    def reset(self):
        self.n = 0
        return np.zeros(4), {}

    def step(self, a):
        self.n += 1
        r = float("nan") if self.n % 2 else -1.0
        return np.zeros(4), r, self.n > 3, False, {}


def test_cem_does_not_let_nan_returns_become_elite():
    """Bug: np.argsort places NaN LAST, so NaN-scoring samples were selected as
    the elite set and poisoned the search mean -- while returns.max() stayed
    NaN so best_return never updated. The run silently did nothing."""
    from hfmarl.agents.cem import train_cem
    from hfmarl.agents.policy import make_policy

    p = make_policy(4, 2, seed=0)
    train_cem(_NanEnv(), p, iterations=3, population=4, seed=0, max_steps=5,
              verbose=False)
    assert np.all(np.isfinite(p.get_flat()))


def test_cem_reports_how_many_rollouts_were_non_finite():
    """A data-quality problem must be visible, not absorbed into the curve."""
    from hfmarl.agents.cem import train_cem
    from hfmarl.agents.policy import make_policy

    res = train_cem(_NanEnv(), make_policy(4, 2, seed=0), iterations=2,
                    population=4, seed=0, max_steps=5, verbose=False)
    assert all("n_nonfinite" in h for h in res.history)
    assert sum(h["n_nonfinite"] for h in res.history) > 0


def test_cem_mean_return_ignores_non_finite_members():
    """Including -inf would make the mean -inf and hide the population."""
    from hfmarl.agents.cem import train_cem
    from hfmarl.agents.policy import make_policy

    res = train_cem(_NanEnv(), make_policy(4, 2, seed=0), iterations=2,
                    population=4, seed=0, max_steps=5, verbose=False)
    for h in res.history:
        assert np.isfinite(h["mean_return"]) or h["n_nonfinite"] == 4


def test_cem_still_works_on_a_clean_environment():
    """The guard must not break the normal path."""
    from hfmarl.agents.cem import train_cem
    from hfmarl.agents.policy import make_policy

    class Clean(_NanEnv):
        def step(self, a):
            self.n += 1
            return np.zeros(4), -float(np.sum(a ** 2)), self.n > 3, False, {}

    res = train_cem(Clean(), make_policy(4, 2, seed=1), iterations=3,
                    population=6, seed=1, max_steps=5, verbose=False)
    assert np.isfinite(res.best_return)


# -- S: a record must be serialisable the moment it exists ---------------


def test_numpy_scalars_are_coerced_at_construction():
    """Bug: a np.int64 leaking in from a rollout made ExperimentLog.save raise
    -- hours into a run, after the data it was meant to protect existed."""
    import json

    rec = ShotRecord(shot=np.int64(2), reward=np.float64(-1.5), steps=np.int64(7),
                     beta_error=np.float64(0.3), wall_seconds=np.float32(0.01))
    assert isinstance(rec.shot, int) and isinstance(rec.reward, float)
    assert isinstance(rec.steps, int)
    r = RunLog("c", "d", 0)
    r.add(rec)
    json.dumps(r.to_dict())


def test_violations_are_coerced_to_a_tuple_of_str():
    rec = ShotRecord(shot=0, reward=-1.0, steps=1, violations=["beta_N", "q95"])
    assert rec.violations == ("beta_N", "q95")


def test_full_experiment_roundtrip_with_numpy_values(tmp_path):
    e = ExperimentLog("np")
    r = RunLog("isolated", "iter_like", 0)
    for i in range(5):
        r.add(ShotRecord(shot=np.int64(i), reward=np.float64(-i), steps=np.int64(3)))
    e.add(r)
    back = ExperimentLog.load(e.save(tmp_path / "e.json"))
    assert back.get("isolated", "iter_like", 0).rewards().tolist() == [0, -1, -2, -3, -4]


# -- V: the elongation correction in q_cylindrical -----------------------


def test_q_cylindrical_uses_the_standard_shaping_factor():
    """Bug: a bare `kappa` was used where the standard form is (1+kappa^2)/2.
    For ITER that read 2.80 instead of 3.22 -- ~15% low, and q feeds both nu*
    and the similarity metric."""
    import math

    d = get("iter_like")
    mu0 = 4e-7 * math.pi
    base = 2 * math.pi * d.a_minor**2 * d.B_0 / (mu0 * d.R_major * d.Ip_nominal)
    assert d.q_cylindrical == pytest.approx(base * (1 + d.elongation**2) / 2)
    assert d.q_cylindrical != pytest.approx(base * d.elongation)


def test_q_cylindrical_scales_inversely_with_current():
    """q ~ 1/Ip is the defining behaviour; a shaping bug would not show here,
    but a structural one would."""
    from dataclasses import replace

    d = get("iter_like")
    half = replace(d, Ip_nominal=d.Ip_nominal / 2)
    assert half.q_cylindrical == pytest.approx(2 * d.q_cylindrical, rel=1e-9)


@pytest.mark.parametrize("device", [d.name for d in all_devices()])
def test_q_is_in_the_realistic_band(device):
    """Every one of these machines runs q95 ~ 3-4."""
    assert 2.5 < get(device).q_cylindrical < 5.0


# -- W: beta must be the same quantity TORAX reports ---------------------


@pytest.mark.parametrize(
    "device,expected",
    [("iter_like", 1.6), ("sparc_like", 0.9), ("diiid_like", 2.3), ("tcv_like", 1.2)],
)
def test_operating_points_reproduce_published_beta_N(device, expected):
    """Bug: OPERATING_POINTS held CORE n and T, so `encode` returned a core
    beta ~1.8x the volume-averaged beta_N TORAX reports and the limits are
    written against -- two different quantities sharing a name. A device would
    sit somewhere different in similarity space before a run than during one."""
    assert encoded_states()[device].beta_N == pytest.approx(expected, abs=0.35)


def test_no_device_sits_above_the_beta_limit_at_its_nominal_point():
    """If it did, the device could not be operated at its own reference point."""
    from hfmarl.envs.limits import DEFAULT_LIMITS

    hard = next(l for l in DEFAULT_LIMITS if l.name == "beta_N").hard
    for name, s in encoded_states().items():
        assert s.beta_N < hard, f"{name} nominal beta_N={s.beta_N:.2f} >= {hard}"


def test_encode_accepts_a_measured_beta_N_override():
    """Once TORAX is running its beta_N is strictly better than a 2-scalar
    estimate, so it must be passable directly."""
    from hfmarl.physics.dimensionless import encode

    d = get("iter_like")
    kw = dict(T_e_keV=8.0, T_i_keV=8.0, n_e=0.7e20, q95=3.2, B_0=d.B_0,
              R_major=d.R_major, a_minor=d.a_minor, Ip=d.Ip_nominal)
    assert encode(**kw, beta_N=2.75).beta_N == pytest.approx(2.75)
    assert encode(**kw).beta_N != pytest.approx(2.75)


def test_beta_scales_correctly_with_density_and_field():
    """beta ~ n*T/B^2. A sign or exponent slip would show up here."""
    from hfmarl.physics.dimensionless import beta_toroidal

    b = float(beta_toroidal(1e20, 5.0, 5.0, 3.0))
    assert float(beta_toroidal(2e20, 5.0, 5.0, 3.0)) == pytest.approx(2 * b)
    assert float(beta_toroidal(1e20, 10.0, 10.0, 3.0)) == pytest.approx(2 * b)
    assert float(beta_toroidal(1e20, 5.0, 5.0, 6.0)) == pytest.approx(b / 4)


# -- X: the similarity structure must match the physics ------------------


def test_sparc_is_iters_nearest_peer():
    """SPARC-like is a compact high-field machine that reaches ITER-like rho*.
    If the metric does not rank it as ITER's closest peer, something in the
    encoding is wrong -- this caught the core-vs-volume beta bug, which had
    put DIII-D closer to ITER than SPARC."""
    from hfmarl.federation.similarity import similarity_distance

    s = encoded_states()
    d = {n: similarity_distance(s["iter_like"], v)
         for n, v in s.items() if n != "iter_like"}
    assert min(d, key=d.get) == "sparc_like", d


def test_tcv_is_iters_most_distant_peer():
    """The designed negative control for the similarity weighting."""
    from hfmarl.federation.similarity import similarity_distance

    s = encoded_states()
    d = {n: similarity_distance(s["iter_like"], v)
         for n, v in s.items() if n != "iter_like"}
    assert max(d, key=d.get) == "tcv_like", d


def test_collisionality_ordering_follows_machine_size():
    """Small cold machines are more collisional. TCV must top the list."""
    s = encoded_states()
    assert s["tcv_like"].nu_star > s["diiid_like"].nu_star
    assert s["tcv_like"].nu_star > s["iter_like"].nu_star


def test_every_device_is_in_the_banana_regime():
    """nu* < 1 for all of these; crossing nu*=1 would change the transport
    physics qualitatively and invalidate weight sharing across the set."""
    for name, s in encoded_states().items():
        assert 0 < s.nu_star < 1.0, f"{name}: nu*={s.nu_star}"


# -- X: FINDINGS.md is UTF-8, and record() must write it that way --------


def test_record_writes_utf8_not_the_platform_default(tmp_path):
    """`open("a")` uses cp1252 on Windows. One section sign in a gate's
    output then writes a lone 0xA7 into a file every other tool reads as
    UTF-8, and nothing complains until something tries to decode it. It
    corrupted FINDINGS.md once; this pins the fix.
    """
    from hfmarl.util.report import record

    target = tmp_path / "FINDINGS.md"
    target.write_text("# Findings" + chr(10), encoding="utf-8")
    body = "SPEC.md \u00a72 \u2014 \u03c1*, \u03bd*, \u03b2_N"
    record("encoding check", body, path=target)

    raw = target.read_bytes()
    assert body in raw.decode("utf-8")
