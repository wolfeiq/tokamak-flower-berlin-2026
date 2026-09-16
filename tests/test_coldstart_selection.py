"""A handover trial must not certify a controller that was rejected."""

import numpy as np
import pytest

from hfmarl.agents.policy import make_policy
from hfmarl.metrics.log import RunLog, ShotRecord


@pytest.mark.parametrize("handover_return,adopted", [(1.0, True), (-1.0, False)])
def test_only_installed_handover_controller_is_eligible(monkeypatch,
                                                       handover_return,
                                                       adopted):
    from hfmarl.experiments import runner

    theta = make_policy(2, 1, seed=0).get_flat()
    source_meta = {
        "rounds": 1,
        "final_params": {"iter_like": theta},
        "shots_by_device": {"iter_like": 7},
        "total_shots": 7,
    }
    monkeypatch.setattr(runner, "run_condition",
                        lambda *args, **kwargs: ([], source_meta, None))

    class FakeEnv:
        n_actions = 1

        def __init__(self, *args, **kwargs):
            pass

        def _observe(self):
            return np.zeros(2)

    monkeypatch.setattr(runner, "ToraxDeviceEnv", FakeEnv)

    def fake_shot(env, policy, vector, shot, evaluation=False):
        total = handover_return if shot == 0 else 0.0
        return ShotRecord(shot, total, 1, beta_error=0.0,
                          is_evaluation=evaluation), total

    monkeypatch.setattr(runner, "fire_shot", fake_shot)
    log, meta = runner.run_cold_start(
        "tcv_like", ["iter_like"], "moderate", seed=0,
        pretrain_shots=7, join_shots=2, probe_shots=0,
        arm="single_source", accept_if_better=True)

    assert meta["handover_adopted"] is adopted
    assert meta["inherit"] is True
    assert meta["use_similarity"] is False
    assert len(log) == 2  # Both trials remain charged and inspectable.
    assert all(r.is_evaluation for r in log.shots)
    assert log.shots[0].evaluation_eligible is adopted
    assert log.shots[1].evaluation_eligible is not adopted
    assert meta["source_shots_by_device"] == {"iter_like": 7}
    assert meta["source_shots"] == 7
    assert meta["joiner_shots"] == 2
    assert meta["joiner_selection_shots"] == 2
    assert meta["joiner_probe_shots"] == 0
    assert meta["joiner_training_shots"] == 0
    assert meta["joiner_evaluation_shots"] == 0
    assert meta["shared_calibration_shots"] == 5

    reloaded = RunLog.from_dict(log.to_dict())
    assert [r.evaluation_eligible for r in reloaded.shots] == [adopted, not adopted]


def test_legacy_shot_without_eligibility_remains_loadable():
    old = {"condition": "scratch", "device": "iter_like", "seed": 0,
           "shots": [{"shot": 0, "reward": 1.0, "steps": 20,
                      "is_evaluation": True}]}
    assert RunLog.from_dict(old).shots[0].evaluation_eligible is True


def test_isolated_source_training_is_invariant_to_subset_and_order(monkeypatch):
    from hfmarl.experiments import runner

    class FakeEnv:
        n_actions = 1

        def __init__(self, device, **kwargs):
            self.name = device.name

        def _observe(self):
            return np.zeros(2)

    monkeypatch.setattr(runner, "ToraxDeviceEnv", FakeEnv)
    states = runner.encoded_states()
    monkeypatch.setattr(runner, "state_from_env", lambda env: states[env.name])
    proposals = {}

    def fake_shot(env, policy, vector, shot, evaluation=False):
        proposals.setdefault(env.name, []).append(vector.copy())
        total = float(shot)  # Each proposed point is adopted, exercising the RNG.
        return ShotRecord(shot, total, 1, is_evaluation=evaluation), total

    monkeypatch.setattr(runner, "fire_shot", fake_shot)
    sequences = []
    for names in (["sparc_like"], ["diiid_like", "sparc_like"],
                  ["sparc_like", "diiid_like"]):
        proposals.clear()
        runner.run_condition(runner.ISOLATED, names, "moderate", seed=3,
                             shots=5, eval_every=0)
        sequences.append(np.stack(proposals["sparc_like"]))
    np.testing.assert_array_equal(sequences[0], sequences[1])
    np.testing.assert_array_equal(sequences[0], sequences[2])
