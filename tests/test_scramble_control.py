"""The scramble control, and the ways it can silently stop being one.

Publish each client under another device's measured coordinates and see
whether the similarity weighting still appears to work. If it does, it was
never the physics -- which is the only test of SPEC.md 4b that does not
depend on believing the coordinates in the first place.

It lived in exp_federation.py alone until the cold-start experiment needed
it. A control that only one script can run is a control that does not get
run.
"""
import sys
from pathlib import Path

import pytest

from hfmarl.devices.registry import DEVICES
from hfmarl.experiments.runner import derangements

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

COLDSTART = (Path(__file__).resolve().parents[1]
             / "scripts" / "exp_coldstart.py").read_text(encoding="utf-8")
RUNNER = (Path(__file__).resolve().parents[1]
          / "hfmarl" / "experiments" / "runner.py").read_text(encoding="utf-8")


def test_no_device_ever_keeps_its_own_coordinates():
    """A plain shuffle can leave a device in place, which makes the control
    partly not a control."""
    names = sorted(DEVICES)
    for d in derangements(names, 9):
        assert all(d[n] != n for n in names), d
        assert sorted(d.values()) == names   # a permutation, not a resample


def test_asking_for_more_than_exist_repeats_rather_than_inventing():
    """Four devices have nine derangements. Asking for twenty is sampling
    with replacement, and that is a fact about the control worth not
    hiding."""
    names = sorted(DEVICES)
    many = derangements(names, 20)
    assert len(many) == 20
    assert all(all(d[n] != n for n in names) for d in many)


def test_two_devices_have_exactly_one_derangement():
    assert derangements(["a", "b"], 5) == [{"a": "b", "b": "a"}] * 5


def test_one_device_has_none():
    assert derangements(["a"], 3) == []


def test_the_cold_start_arms_all_receive_the_scramble():
    """Including handover_merge, which publishes its own states.

    If it kept true coordinates while the federated rungs published permuted
    ones, the control would change two things at once -- the same defect as
    the ladder bug it sits next to.
    """
    assert RUNNER.count("scramble=scramble") == 3   # the three source-training arms
    assert "pub_states = ({d: handover_states[scramble.get(d, d)]" in RUNNER
    call = RUNNER[RUNNER.index("agg = _merge_finals("):]
    call = call[:call.index("violation_rates")]
    assert "pub_states" in call


def test_source_publication_can_scramble_to_the_held_out_device(monkeypatch):
    """The full-device derangement includes a joiner absent from visited."""
    import numpy as np

    from hfmarl.devices.registry import encoded_states
    from hfmarl.experiments import runner
    from hfmarl.metrics.log import ShotRecord

    class FakeEnv:
        n_actions = 1

        def __init__(self, device, **kwargs):
            self.device = device

        def _observe(self):
            return np.zeros(2)

    states = encoded_states()
    monkeypatch.setattr(runner, "ToraxDeviceEnv", FakeEnv)
    monkeypatch.setattr(runner, "state_from_env",
                        lambda env: states[env.device.name])
    monkeypatch.setattr(runner, "fire_shot",
                        lambda env, policy, vector, shot, evaluation=False:
                        (ShotRecord(shot, 0.0, 1,
                                    is_evaluation=evaluation), 0.0))
    _, meta, server = runner.run_condition(
        "fedbuff_similarity", ["iter_like"], "moderate", seed=0, shots=1,
        local_shots=1, scramble={"iter_like": "tcv_like"})

    update = server.buffered(runner.ONLY_CLUSTER)[0]
    assert update.state == states["tcv_like"]
    assert np.allclose(update.region.mean,
                       runner.region_from_state(states["tcv_like"]).mean)
    assert meta["nominal_region_fallbacks"] == 1


@pytest.mark.parametrize("scramble", [None, {
    "iter_like": "sparc_like", "sparc_like": "tcv_like"}])
def test_handover_permutation_moves_measured_states_and_regions_together(
        monkeypatch, scramble):
    """A mapped held-out coordinate gets its nominal point in both fields."""
    from dataclasses import replace

    import numpy as np

    from hfmarl.agents.policy import make_policy
    from hfmarl.devices.registry import encoded_states
    from hfmarl.experiments import runner
    from hfmarl.metrics.log import ShotRecord

    states = encoded_states()
    sources = ["iter_like", "sparc_like"]
    measured = {n: replace(states[n], beta_N=states[n].beta_N * 1.5)
                for n in sources}
    regions = {n: runner.region_from_state(measured[n]) for n in sources}
    theta = make_policy(2, 1, seed=0).get_flat()
    source_meta = {
        "rounds": 1,
        "final_params": {n: theta.copy() for n in sources},
        "final_states": measured,
        "final_regions": regions,
        "final_violation_rates": dict.fromkeys(sources, 0.0),
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
    monkeypatch.setattr(runner, "fire_shot",
                        lambda env, policy, vector, shot, evaluation=False:
                        (ShotRecord(shot, 0.0, 1,
                                    is_evaluation=evaluation), 0.0))
    published = []
    real_merge = runner._merge_finals

    def capture_merge(*args, **kwargs):
        aggregate = real_merge(*args, **kwargs)
        published.extend(args[2].buffered(runner.ONLY_CLUSTER))
        return aggregate

    monkeypatch.setattr(runner, "_merge_finals", capture_merge)
    runner.run_cold_start(
        "tcv_like", sources, "moderate", seed=0, pretrain_shots=1,
        join_shots=1, probe_shots=0, arm="handover_merge", scramble=scramble)

    for update in published:
        source = (scramble or {}).get(update.device, update.device)
        expected = measured.get(source, states[source])
        assert update.state == expected
        assert np.allclose(update.region.mean,
                           runner.region_from_state(expected).mean)


def test_a_scrambled_run_says_so_in_its_header_and_its_artifacts():
    """A scrambled fold whose output looks like a real one is worse than no
    control: a wrong number carrying a real number's provenance."""
    assert "SCRAMBLE    :" in COLDSTART
    assert "this is the negative control" in COLDSTART
    assert '"scramble": dict(scramble) if scramble else {}' in RUNNER


@pytest.mark.parametrize("n", [1, 3, 9])
def test_derangements_are_deterministic_for_a_seed(n):
    names = sorted(DEVICES)
    assert derangements(names, n, seed=7) == derangements(names, n, seed=7)


def test_publishing_under_a_peers_coordinates_changes_who_gets_the_weight():
    """The control has to actually do something, or it proves nothing.

    Same models, same joiner, same bandwidth -- only the coordinates each
    source publishes under are permuted. If the weights came back unchanged,
    a scrambled fold would be indistinguishable from a real one and the
    control would be decoration.
    """
    import numpy as np

    from hfmarl.devices.registry import encoded_states
    from hfmarl.experiments.runner import ONLY_CLUSTER
    from hfmarl.federation.server import FedBuffServer
    from hfmarl.federation.similarity import ClientUpdate, suggest_bandwidth

    states = encoded_states()
    sources = ["diiid_like", "iter_like", "sparc_like"]
    joiner = "tcv_like"
    bw = suggest_bandwidth(list(states.values()))

    def weights(mapping):
        srv = FedBuffServer(bandwidth=bw, use_similarity=True)
        for d in sources:
            srv.publish(ClientUpdate(
                device=d, cluster=ONLY_CLUSTER,
                weights={"flat": np.zeros(8)},
                state=states[mapping.get(d, d)],
                n_samples=1, round_produced=1))
        srv.aggregate_for(joiner, states[joiner], ONLY_CLUSTER, 2)
        return srv.weight_table()[joiner]

    true = weights({})
    # A derangement of the sources among themselves.
    permuted = weights({"diiid_like": "iter_like", "iter_like": "sparc_like",
                        "sparc_like": "diiid_like"})

    assert set(true) == set(permuted) == set(sources)
    assert any(abs(true[d] - permuted[d]) > 1e-6 for d in sources), (
        "permuting the coordinates left every weight unchanged; the scramble "
        "is not a control")
    # And the nearest source by TRUE coordinates should lose weight when it
    # publishes under someone else's.
    nearest = max(true, key=true.get)
    assert permuted[nearest] < true[nearest]


def test_the_bandwidth_is_the_same_in_every_leave_one_out_fold():
    """Why the joiner is inside the bandwidth set, stated as a test.

    PROTOCOL.md 2 forbids tuning a hyperparameter on the held-out device and
    names the bandwidth. The median heuristic is a fixed rule rather than a
    fit, and computing it over the incumbents alone would give a different
    kernel width in every fold -- 1.1024 to 1.3644 against 1.1760 for the
    full set -- so the folds would stop being comparable for a reason that
    has nothing to do with the arms. This pins the choice so a future
    literal reading of the protocol cannot quietly undo it.
    """
    from hfmarl.devices.registry import encoded_states
    from hfmarl.federation.similarity import suggest_bandwidth

    states = encoded_states()
    full = suggest_bandwidth(list(states.values()))
    per_fold = {
        j: suggest_bandwidth([states[n] for n in states if n != j])
        for j in states
    }
    assert min(per_fold.values()) < full < max(per_fold.values())
    spread = max(per_fold.values()) / min(per_fold.values()) - 1
    assert spread > 0.2, (
        "if the incumbent-only bandwidths no longer differ, the reason for "
        "using the full set has gone and the choice should be revisited")

    runner = (Path(__file__).resolve().parents[1] / "hfmarl" / "experiments"
              / "runner.py").read_text(encoding="utf-8")
    body = runner[runner.index("def run_cold_start("):]
    call = body[body.index("bandwidth = suggest_bandwidth("):]
    call = call[:call.index(")])") + 3]
    assert "[joiner] + list(incumbents)" in call
