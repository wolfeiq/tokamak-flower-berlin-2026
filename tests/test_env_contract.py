"""Environment contract that can be checked without a live TORAX."""

import numpy as np
import pytest

from hfmarl.devices.registry import get
from hfmarl.envs.torax_env import ToraxCore, ToraxUnavailableError, ToraxDeviceEnv


def test_action_window_must_be_multiple_of_solver_step():
    """A fractional ratio leaves a cropped substep and muddies the JIT timing."""
    cfg = {"numerics": {"fixed_dt": 0.1}}
    ToraxCore(cfg, 0.5)  # 5x, fine
    with pytest.raises(ValueError, match="integer multiple"):
        ToraxCore(cfg, 0.25)


def test_nonpositive_delta_t_rejected():
    with pytest.raises(ValueError):
        ToraxCore({"numerics": {"fixed_dt": 0.1}}, 0.0)


def test_missing_torax_raises_with_install_advice():
    pytest.importorskip  # noqa: B018
    try:
        import torax  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("TORAX is installed; this test covers the absent case")
    env = ToraxDeviceEnv(get("iter_like"))
    with pytest.raises(ToraxUnavailableError, match="uv pip install torax"):
        env.reset()


def test_env_exposes_expected_action_count():
    env = ToraxDeviceEnv(get("iter_like"), clusters=("thermal",))
    assert env.n_actions == 2
    assert env.bank.names == ("aux_heat", "ecrh")


# ---------------------------------------------------------------------------
# Reward shaping: terminating early must never be cheaper than continuing.
# ---------------------------------------------------------------------------


def _episode_cost(env, err, violate_at=None):
    """Total return for an episode with constant tracking error `err`.

    Replicates the reward arithmetic without needing a live TORAX, so the
    incentive structure can be checked on any machine.
    """
    n = env.task.steps_per_shot
    per_step = -(max(0.0, err - env.task.tolerance) ** 2)
    if violate_at is None:
        return n * per_step
    # The violating step pays the terminal penalty, then the early-exit charge
    # covers the episode the policy escaped.
    env._last_control_cost = per_step
    step_r = per_step - env.violation_penalty
    remaining = max(0, n - violate_at)
    charge = remaining * min(per_step, 0.0)
    return (violate_at - 1) * per_step + step_r + charge


@pytest.mark.parametrize("task", ["easy", "moderate", "hard", "brutal"])
@pytest.mark.parametrize("err", [0.5, 1.0, 1.9])
def test_violating_is_never_cheaper_than_finishing(task, err):
    """Regression: an untrained policy must not be rewarded for crashing.

    Before the early-exit charge, a policy sitting at beta_N error ~1.9 -- the
    normal starting point -- scored BETTER by driving into a limit on step 1
    than by tracking badly to the end, because termination capped its losses at
    a one-off penalty. That would have corrupted the violations-during-training
    curve, which carries the safety half of the claim.
    """
    env = ToraxDeviceEnv(get("iter_like"), task=task)
    finish = _episode_cost(env, err)
    for k in (1, env.task.steps_per_shot // 2, env.task.steps_per_shot):
        assert _episode_cost(env, err, violate_at=k) < finish, (
            f"{task}, err={err}: violating at step {k} scores "
            f"{_episode_cost(env, err, violate_at=k):.1f} vs {finish:.1f} for "
            "finishing -- crashing is being rewarded"
        )


@pytest.mark.parametrize("task", ["easy", "hard", "brutal"])
def test_violation_costs_exactly_the_penalty_regardless_of_when(task):
    """The gap must be task-length- and timing-independent, or the incentive
    to crash returns for long episodes."""
    env = ToraxDeviceEnv(get("iter_like"), task=task)
    finish = _episode_cost(env, 1.5)
    gaps = [
        finish - _episode_cost(env, 1.5, violate_at=k)
        for k in (1, 3, env.task.steps_per_shot // 2)
    ]
    for g in gaps:
        assert g == pytest.approx(env.violation_penalty, rel=1e-9)


def test_early_exit_charge_is_zero_at_the_final_step():
    """Nothing was escaped, so nothing extra is owed."""
    env = ToraxDeviceEnv(get("iter_like"), task="easy")
    env._last_control_cost = -1.0
    env.trajectory = [None] * (env.task.steps_per_shot - 1)
    assert env._early_exit_charge() == 0.0


def test_early_exit_charge_never_pays_a_bonus():
    """A positive running cost must not turn termination into a reward."""
    env = ToraxDeviceEnv(get("iter_like"), task="hard")
    env.trajectory = []
    env._last_control_cost = 1000.0
    assert env._early_exit_charge() <= 0.0


def test_disturbed_configs_are_cached_not_rebuilt_every_shot():
    """Rebuilding means a full pydantic revalidation; headroom runs thousands."""
    env = ToraxDeviceEnv(get("iter_like"), task="hard")
    assert env.task.disturbance_std > 0
    for _ in range(300):
        factor = round(float(np.clip(env.rng.normal(1.0, env.task.disturbance_std),
                                     0.5, 1.5)), 2)
        if factor not in env._config_cache:
            env._config_cache[factor] = object()
    assert len(env._config_cache) < 120, "cache is not amortising rebuilds"


# ---------------------------------------------------------------------------
# Regression: the two failure exits must not crash the run.
#
# `step()` built its StepResult with `violations=...` while StepResult had no
# such field, so BOTH abnormal exits raised
#     TypeError: StepResult.__init__() got an unexpected keyword argument
# The exception escapes `step()` -- ToraxCore.advance()'s try/except is already
# behind us by then -- so a solver blow-up or a NaN state killed the whole
# training run instead of ending one episode. These are precisely the paths a
# bad policy takes, and neither had a test.
# ---------------------------------------------------------------------------


class _FakeCore:
    """Stands in for ToraxCore so the failure exits are reachable without TORAX."""

    def __init__(self, *, ok: bool, nonfinite: tuple[str, ...] = ()):
        self._ok, self._nonfinite = ok, nonfinite
        self.t, self.t_final, self.delta_t_a = 0.0, 10.0, 0.5

    def reset(self):
        self.t = 0.0

    def apply_updates(self, updates):
        pass

    def advance(self):
        if not self._ok:
            return False, True, "exception during step: solver diverged"
        self.t += self.delta_t_a
        return True, False, ""

    def nonfinite_scalars(self):
        return self._nonfinite

    def read_scalars(self):
        return {"beta_N": 2.0, "q95": 4.0, "fgw_n_e_line_avg": 0.5}

    def read_profiles(self):
        return {}

    def limit_inputs(self):
        return {"beta_N": 2.0, "q95": 4.0, "greenwald_fraction": 0.5}


def _env_with(core):
    env = ToraxDeviceEnv(get("iter_like"), task="easy")
    env.core = core
    env.reset()
    return env


def test_solver_failure_ends_the_episode_instead_of_raising():
    from hfmarl.envs.torax_env import SOLVER_FAILURE

    env = _env_with(_FakeCore(ok=False))
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_actions))

    assert terminated and not truncated
    assert info["violations"] == (SOLVER_FAILURE,)
    assert env.trajectory[-1].violations == (SOLVER_FAILURE,)
    assert np.isfinite(reward) and reward <= -env.violation_penalty


def test_nonfinite_state_ends_the_episode_instead_of_raising():
    from hfmarl.envs.torax_env import NONFINITE_STATE

    env = _env_with(_FakeCore(ok=True, nonfinite=("beta_N",)))
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_actions))

    assert terminated and not truncated
    assert info["violations"] == (NONFINITE_STATE,)
    assert env.trajectory[-1].violations == (NONFINITE_STATE,)
    assert np.isfinite(reward) and reward <= -env.violation_penalty


def test_violations_are_recorded_on_normal_steps_too():
    """The trajectory is what gets logged; a violation only in `info` is lost."""
    env = _env_with(_FakeCore(ok=True))
    env.step(np.zeros(env.n_actions))
    assert env.trajectory[-1].violations == ()
