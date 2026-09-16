"""Each rung of the cold-start ladder must add exactly one ingredient.

That property is the ladder's entire justification: PROTOCOL.md 3 says a gap
between adjacent rungs is attributable to the one thing that differs, and to
nothing else. It was false where it mattered most.

`handover_merge` built its server with `use_similarity=True` while the rung
directly above it, `federated_uniform`, weights uniformly -- so climbing the
ladder REMOVED the physics weighting. And nothing it published carried a
violation rate, so it also escaped the safety downweighting every federated
update pays. Measured weights for a tcv_like joiner, from a saved run:

    handover_merge      diiid 0.558  sparc 0.232  iter 0.210
    federated_uniform   iter  0.377  sparc 0.377  diiid 0.245

Three changes between those rows, not one. These tests are here so that
cannot come back silently.
"""
from pathlib import Path

import numpy as np
import pytest

from hfmarl.experiments.runner import ONLY_CLUSTER
from hfmarl.devices.registry import encoded_states
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import ClientUpdate, suggest_bandwidth

RUNNER = (Path(__file__).resolve().parents[1]
          / "hfmarl" / "experiments" / "runner.py").read_text(encoding="utf-8")
STATES = encoded_states()


def test_handover_merge_weights_uniformly_like_the_rung_above_it():
    """Read from the source, because the arm is chosen by a branch."""
    branch = RUNNER[RUNNER.index('elif arm == "handover_merge":'):
                    RUNNER.index('else:  # federated_uniform')]
    assert "use_similarity=False" in branch
    assert "use_similarity=True" not in branch


@pytest.mark.parametrize("align", [True, False])
def test_handover_merge_aligns_equivalent_permuted_source_policies(monkeypatch,
                                                                 align):
    """Exercise the actual cold-start branch without running TORAX.

    Two source networks implement the same function in different hidden-unit
    orders. With alignment enabled their handover must recover that function;
    the explicit no-alignment ablation must still average the mismatched rows.
    """
    from hfmarl.agents.policy import make_policy
    from hfmarl.experiments import runner
    from hfmarl.federation.robust import permute_hidden
    from hfmarl.metrics.log import ShotRecord

    reference = make_policy(2, 1, seed=0)
    theta = reference.get_flat()
    permuted = permute_hidden(theta, np.arange(reference.hidden)[::-1],
                              reference.obs_dim, reference.hidden,
                              reference.act_dim)
    sources = ["iter_like", "sparc_like"]
    source_meta = {
        "rounds": 1,
        "final_params": dict(zip(sources, [theta, permuted])),
        "final_regions": {},
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
    evaluated = []

    def fake_shot(env, policy, vector, shot, evaluation=False):
        evaluated.append(vector.copy())
        return ShotRecord(shot, 0.0, 1, is_evaluation=evaluation), 0.0

    monkeypatch.setattr(runner, "fire_shot", fake_shot)
    log, meta = runner.run_cold_start(
        "tcv_like", sources, "moderate", seed=0, pretrain_shots=1,
        join_shots=1, probe_shots=0, arm="handover_merge", rule="mean",
        align=align, clip_factor=None)

    assert len(log) == 1
    assert meta["handover_adopted"] is True
    expected = theta if align else (theta + permuted) / 2
    assert np.allclose(evaluated[0], expected)
    if not align:
        assert not np.allclose(evaluated[0], theta)


def test_handover_merge_publishes_the_violation_rate():
    """Otherwise it merges under a safety rule the federated arms pay and it
    does not, which is a second difference in the same comparison."""
    assert "violation_rates=meta.get(\"final_violation_rates\")" in RUNNER
    merge = RUNNER[RUNNER.index("def _merge_finals("):
                   RUNNER.index("def run_cold_start(")]
    assert "violation_rate=float(rates.get(name, 0.0))" in merge


def test_run_condition_reports_each_source_violation_rate():
    assert '"final_violation_rates"' in RUNNER


def _publish(srv, rates):
    for d, r in rates.items():
        srv.publish(ClientUpdate(
            device=d, cluster=ONLY_CLUSTER,
            weights={"flat": np.full(8, 1.0)},
            state=STATES[d], n_samples=1, round_produced=1,
            violation_rate=r))


def test_a_uniform_server_still_punishes_the_device_that_can_violate():
    """The mechanism behind the 0.377/0.377/0.245 row, stated as a test.

    Two of the four devices cannot cross a limit at ANY command, so their
    violation rate is structurally zero and their safety factor is
    permanently 1. "Uniform" weighting is therefore uniform only among
    devices with equal headroom -- which is worth knowing, and is why the
    rung is described as uniform-modulo-safety rather than uniform.
    """
    bw = suggest_bandwidth(list(STATES.values()))
    srv = FedBuffServer(bandwidth=bw, use_similarity=False, use_safety=True)
    _publish(srv, {"iter_like": 0.0, "sparc_like": 0.0, "diiid_like": 0.30})
    srv.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2)
    w = srv.weight_table()["tcv_like"]
    assert w["iter_like"] == w["sparc_like"]
    assert w["diiid_like"] < w["iter_like"]
    # (1 - 0.30) against 1.0, normalised over the three
    assert w["diiid_like"] / w["iter_like"] == pytest.approx(0.70, rel=1e-6)


def test_with_safety_off_uniform_really_is_uniform():
    """The control for the test above: nothing else is tilting the weights."""
    bw = suggest_bandwidth(list(STATES.values()))
    srv = FedBuffServer(bandwidth=bw, use_similarity=False, use_safety=False)
    _publish(srv, {"iter_like": 0.0, "sparc_like": 0.0, "diiid_like": 0.30})
    srv.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2)
    w = srv.weight_table()["tcv_like"]
    assert len(set(round(v, 9) for v in w.values())) == 1


def test_the_run_prints_every_arm_weights_not_just_the_similarity_one():
    """How the bug above stayed hidden.

    The script printed `received` for `federated_similarity` alone, so the
    row that gave it away -- handover_merge's 0.558/0.232/0.210 sitting next
    to federated_uniform's 0.377/0.377/0.245 -- only existed inside
    `runs.json`. Two settings of the same rule do not look like that, and
    three rows side by side say so at a glance.
    """
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "exp_coldstart.py").read_text(encoding="utf-8")
    block = script[script.index("inherited from"):]
    assert "for arm, _ in arms:" in script[:script.index("inherited from")][-400:]
    assert 'received.get(arm)' in script
    del block


def test_merging_without_violation_rates_raises_instead_of_assuming_clean():
    """The fallback that would have re-created the bug in silence.

    `violation_rates or {}` meant a caller that forgot them merged with every
    source treated as never having crossed a limit -- the exact behaviour the
    argument exists to remove, and indistinguishable in the output from the
    fixed version.
    """
    from hfmarl.experiments.runner import _merge_finals

    with pytest.raises(ValueError, match="violation rate"):
        _merge_finals({"iter_like": np.zeros(8)}, {}, None, STATES,
                      "tcv_like", STATES["tcv_like"], None, reference=None,
                      violation_rates=None)


def test_merging_with_a_source_missing_its_rate_raises():
    from hfmarl.experiments.runner import _merge_finals

    finals = {"iter_like": np.zeros(8), "diiid_like": np.zeros(8)}
    with pytest.raises(ValueError, match="no violation rate for"):
        _merge_finals(finals, {}, None, STATES, "tcv_like",
                      STATES["tcv_like"], None, reference=None,
                      violation_rates={"iter_like": 0.0})


def test_run_condition_meta_includes_a_rate_for_every_device():
    """Static check on the comprehension, so the key cannot go missing.

    If it did, `meta.get(...)` would hand None to a function that now
    refuses it -- loud rather than silent, which is the point, but better
    caught here.
    """
    block = RUNNER[RUNNER.index('"final_violation_rates"'):]
    block = block[:block.index("},")]
    assert "for d in devices" in block
    assert "r.violated" in block


def test_a_similarity_arm_whose_kernel_underflows_says_so():
    """When every kernel underflows, the similarity arm IS the uniform arm.

    `aggregation_weights` falls back to uniform over the admissible peers --
    the right answer, since the alternative is stalling training -- but the
    returned weights were the only place that ever showed it. Two rungs of
    the ladder producing the same answer for an unstated reason is how a
    null result gets mistaken for a finding.
    """
    from hfmarl.federation.similarity import aggregation_weights

    bw = 1e-6   # every peer is astronomically far at this bandwidth
    updates = [
        ClientUpdate(device=d, cluster=ONLY_CLUSTER,
                     weights={"flat": np.zeros(8)}, state=STATES[d],
                     n_samples=1, round_produced=1)
        for d in ("iter_like", "diiid_like")
    ]
    diag: dict = {}
    w = aggregation_weights(updates, target_state=STATES["tcv_like"],
                            current_round=2, bandwidth=bw,
                            use_similarity=True, diagnostics=diag)
    assert diag["uniform_fallback"] is True
    assert w == pytest.approx([0.5, 0.5])


def test_a_healthy_kernel_does_not_report_a_fallback():
    from hfmarl.federation.similarity import (
        aggregation_weights,
        suggest_bandwidth,
    )

    bw = suggest_bandwidth(list(STATES.values()))
    updates = [
        ClientUpdate(device=d, cluster=ONLY_CLUSTER,
                     weights={"flat": np.zeros(8)}, state=STATES[d],
                     n_samples=1, round_produced=1)
        for d in ("iter_like", "diiid_like")
    ]
    diag: dict = {}
    w = aggregation_weights(updates, target_state=STATES["tcv_like"],
                            current_round=2, bandwidth=bw,
                            use_similarity=True, diagnostics=diag)
    assert diag["uniform_fallback"] is False
    assert w[0] != pytest.approx(w[1])   # it actually discriminated


def test_the_server_counts_fallback_rounds_only_for_the_similarity_arm():
    """For federated_uniform, uniform weights are the design, not a
    degradation, and counting them would cry wolf every round."""
    from hfmarl.federation.server import FedBuffServer

    def run(use_similarity):
        srv = FedBuffServer(bandwidth=1e-6, use_similarity=use_similarity)
        for d in ("iter_like", "diiid_like"):
            srv.publish(ClientUpdate(
                device=d, cluster=ONLY_CLUSTER,
                weights={"flat": np.zeros(8)}, state=STATES[d],
                n_samples=1, round_produced=1))
        srv.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2)
        return srv.uniform_fallback_rounds

    assert run(True) == 1
    assert run(False) == 0
