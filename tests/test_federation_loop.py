"""The half of federation that was missing: buffer, aggregate, adopt.

These run without TORAX. The expensive part of the experiment is the plant;
none of the contracts below need it, and a contract that only gets checked by
an 80-minute run does not get checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.agents.search import HillClimber
from hfmarl.devices.registry import encoded_states
from hfmarl.experiments.runner import (
    ONLY_CLUSTER,
    RoleControlUnavailable,
    run_condition,
)
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import ClientUpdate, suggest_bandwidth

STATES = encoded_states()
NP = 8


def update(device: str, value: float, rnd: int = 1) -> ClientUpdate:
    return ClientUpdate(
        device=device, cluster=ONLY_CLUSTER,
        weights={"flat": np.full(NP, value)},
        state=STATES[device], n_samples=10, round_produced=rnd,
    )


def server(**kw) -> FedBuffServer:
    bw = suggest_bandwidth(list(STATES.values()))
    return FedBuffServer(bandwidth=bw, **kw)


# -- the buffer ----------------------------------------------------------


def test_a_new_update_replaces_that_device_rather_than_queueing():
    """A machine that ran all week must not out-vote one that fired twice.

    Sample-count weighting is the arbitrary rule SPEC.md 4b replaces; letting
    a chatty client stack up buffer slots would reintroduce it by the back
    door.
    """
    s = server()
    s.publish(update("tcv_like", 1.0, rnd=1))
    s.publish(update("tcv_like", 2.0, rnd=2))
    buf = s.buffered(ONLY_CLUSTER)
    assert len(buf) == 1
    assert buf[0].weights["flat"][0] == 2.0


def test_an_empty_channel_aggregates_to_nothing_not_to_zeros():
    """None means 'no peers yet'. A zero vector would be a policy, and the
    client would adopt it and unlearn."""
    s = server()
    assert s.aggregate_for("iter_like", STATES["iter_like"], ONLY_CLUSTER, 1) is None


# -- the aggregate -------------------------------------------------------


def test_the_aggregate_is_a_convex_combination_of_what_was_published():
    s = server()
    for d, v in [("iter_like", 1.0), ("sparc_like", 2.0), ("tcv_like", 3.0)]:
        s.publish(update(d, v))
    agg = s.aggregate_for("diiid_like", STATES["diiid_like"], ONLY_CLUSTER, 1)
    assert agg is not None
    assert 1.0 - 1e-9 <= agg[0] <= 3.0 + 1e-9
    assert np.allclose(agg, agg[0])  # every coordinate mixes identically


def test_similarity_weighting_favours_the_nearer_peer():
    """The whole of SPEC.md 4b in one assertion.

    TCV is the furthest device in dimensionless space and DIII-D the nearest to
    it, so an aggregate built FOR tcv_like must lean toward diiid_like. Under
    uniform weights it must not.
    """
    near, far = "diiid_like", "iter_like"

    sim = server(use_similarity=True)
    uni = server(use_similarity=False)
    for srv in (sim, uni):
        srv.publish(update(near, 0.0))
        srv.publish(update(far, 1.0))

    a_sim = sim.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 1)
    a_uni = uni.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 1)

    # Values are 0.0 for the near peer and 1.0 for the far one, so a smaller
    # aggregate means more weight on the near peer.
    assert a_sim[0] < a_uni[0]
    assert np.isclose(a_uni[0], 0.5, atol=1e-6)


def test_staleness_discounts_an_update_from_an_earlier_round():
    s = server(use_similarity=False)
    s.publish(update("iter_like", 0.0, rnd=10))   # fresh
    s.publish(update("sparc_like", 1.0, rnd=1))   # stale
    agg = s.aggregate_for("diiid_like", STATES["diiid_like"], ONLY_CLUSTER, 10)
    assert agg[0] < 0.5, "the stale client should count for less than half"


def test_the_weight_table_records_who_contributed():
    """A federation that cannot say who it listened to has not been measured."""
    s = server()
    s.publish(update("iter_like", 1.0))
    s.publish(update("tcv_like", 2.0))
    s.aggregate_for("sparc_like", STATES["sparc_like"], ONLY_CLUSTER, 1)
    table = s.weight_table()["sparc_like"]
    assert set(table) == {"iter_like", "tcv_like"}
    assert np.isclose(sum(table.values()), 1.0)


# -- the local search ----------------------------------------------------


def test_the_first_proposal_is_the_incumbent_so_its_return_is_measured():
    c = HillClimber(n_params=NP, seed=0)
    c.seed_from(np.ones(NP))
    assert np.allclose(c.propose(), np.ones(NP))


def test_adopting_an_aggregate_is_unconditional():
    """Adopting only when the aggregate beats the local best would turn every
    federated condition into a local search with an occasional free hint, and
    the comparison would no longer be about aggregation."""
    c = HillClimber(n_params=NP, seed=0)
    c.seed_from(np.zeros(NP))
    c.observe(np.ones(NP), total=10.0)
    c.adopt(np.full(NP, 7.0), total=-100.0)
    assert np.allclose(c.best, 7.0)
    assert c.best_return == -100.0


def test_sigma_anneals_within_its_bounds():
    c = HillClimber(n_params=NP, seed=0, sigma=0.3)
    c.seed_from(np.zeros(NP))
    for _ in range(500):
        c.observe(np.zeros(NP), total=-1.0)  # never an improvement
    assert c.sigma == pytest.approx(c.sigma_min)


# -- the control that cannot run yet -------------------------------------


def test_role_blind_control_refuses_rather_than_duplicating_uniform():
    """With one cluster, role-blind and role-matched aggregation are the same
    computation. Running it would put a duplicate of fedbuff_uniform in the
    figure labelled 'negative control'."""
    with pytest.raises(RoleControlUnavailable, match="one cluster"):
        run_condition("fedavg_naive", ["iter_like"], "easy", seed=0, shots=1)


# -- the ladder's rungs must actually differ -----------------------------


def test_merging_several_models_differs_from_handing_over_one():
    """`single_source` and `handover_merge` differ by WHAT is handed over, so
    if the merge returns one source's model the two rungs are the same
    experiment under two names and any gap between them is noise.

    A smoke run showed them reporting identical plateaus to three decimals.
    That was explainable -- eight pretrain shots leave every source near the
    common initialisation, so both hand over something near theta0 -- but
    "explainable" is not "checked", so this checks it.
    """
    from hfmarl.experiments.runner import _merge_finals

    # Non-collinear, like real policy vectors. COLLINEAR sources are a
    # genuine degenerate case -- the geometric median of 1, 2 and 3 is
    # exactly 2, one source returned unchanged -- but a 194-parameter
    # policy is not collinear with its peers, and measured on random
    # 194-dim vectors the merge sits 9-12 units from every source.
    rng = np.random.default_rng(0)
    finals = {"iter_like": rng.normal(size=16),
              "sparc_like": rng.normal(size=16),
              "diiid_like": rng.normal(size=16)}
    regions = {n: None for n in finals}
    s = server(use_similarity=True)

    merged = _merge_finals(
        finals, regions, s, STATES, "tcv_like", STATES["tcv_like"],
        joiner_region=None, reference=np.zeros(16),
        violation_rates={n: 0.0 for n in finals})

    assert merged is not None
    for name, vec in finals.items():
        assert not np.allclose(merged, vec), (
            f"the merge returned {name}'s model unchanged; handover_merge and "
            "single_source would be the same arm")
    lo = min(v.min() for v in finals.values())
    hi = max(v.max() for v in finals.values())
    assert lo - 1e-9 <= merged.min() and merged.max() <= hi + 1e-9


def test_the_merge_leans_toward_the_nearer_source_when_asked_to():
    """A property of the FUNCTION, not of the handover-merge arm.

    The docstring here used to argue the opposite: that the handover-merge
    rung must weight by similarity or "its gap to the federated rungs is not
    about exchange timing". That reasoning is what produced the ladder bug --
    the rung directly above handover_merge weights uniformly, so a merge that
    weights by similarity makes the gap between them two changes rather than
    one, in opposite directions. The arm now passes use_similarity=False; see
    tests/test_ladder_integrity.py.

    The function must still be ABLE to weight by similarity, because
    `federated_similarity` uses the same server, and this is that check.
    """
    from hfmarl.experiments.runner import _merge_finals

    near, far = "diiid_like", "iter_like"
    finals = {near: np.zeros(6), far: np.ones(6)}
    regions = {n: None for n in finals}
    clean = {n: 0.0 for n in finals}

    sim = _merge_finals(finals, regions, server(use_similarity=True), STATES,
                        "tcv_like", STATES["tcv_like"], None, np.zeros(6),
                        violation_rates=clean)
    uni = _merge_finals(finals, regions, server(use_similarity=False), STATES,
                        "tcv_like", STATES["tcv_like"], None, np.zeros(6),
                        violation_rates=clean)
    assert sim[0] < uni[0], "similarity weighting did not favour the near peer"


def test_a_source_that_kept_crossing_limits_is_downweighted_in_the_merge():
    """The safety rule now reaches the handover merge, which it did not.

    Same two sources, same distance, one of them having spent half its recent
    shots crossing a limit. The merge must move away from it -- otherwise
    handover_merge is merging under a rule the federated rungs do not use.
    """
    from hfmarl.experiments.runner import _merge_finals

    finals = {"diiid_like": np.zeros(6), "iter_like": np.ones(6)}
    regions = {n: None for n in finals}

    clean = _merge_finals(finals, regions, server(use_similarity=False),
                          STATES, "tcv_like", STATES["tcv_like"], None,
                          np.zeros(6),
                          violation_rates={"diiid_like": 0.0,
                                           "iter_like": 0.0})
    dirty = _merge_finals(finals, regions, server(use_similarity=False),
                          STATES, "tcv_like", STATES["tcv_like"], None,
                          np.zeros(6),
                          violation_rates={"diiid_like": 0.0,
                                           "iter_like": 0.5})
    # iter_like holds the ones, so downweighting it pulls the merge down.
    assert dirty[0] < clean[0]


def test_every_arm_labels_its_own_log():
    """Artifacts must say which arm produced them.

    The label was derived from the legacy `inherit`/`use_similarity` booleans,
    which now default to True/True whatever arm runs -- so every saved log,
    scratch included, was stamped `cold_inherit_similarity`. The experiment
    script keyed its own results dict correctly, so the reported numbers were
    right; but `runs.json` is the artefact AUDIT #10 added for reanalysis, and
    a reanalysis reading the condition field would have credited all five arms
    to the similarity arm.

    Source-level, because reaching the label needs a full cold-start run.
    """
    import inspect

    from hfmarl.experiments.runner import COLD_START_ARMS, run_cold_start

    src = inspect.getsource(run_cold_start)
    assert "label = arm" in src, (
        "the run label is not the arm name; saved logs will misattribute runs")
    assert 'label = "cold_inherit_' not in src
    assert len(COLD_START_ARMS) == 5


# -- when the aggregate comes back empty, somebody has to be told ---------


def test_rejecting_every_client_is_recorded_not_merely_returned_as_none():
    """`None` alone cannot distinguish 'all rejected' from 'nothing queued'.

    That ambiguity is how a NaN state once disabled federation silently: the
    arm ran as scratch and the label still said federated. The server records
    which of the two it was.
    """
    srv = server(use_safety=True)
    for d in ("iter_like", "sparc_like"):
        u = update(d, 1.0)
        srv.publish(ClientUpdate(
            device=u.device, cluster=u.cluster, weights=u.weights,
            state=u.state, n_samples=u.n_samples,
            round_produced=u.round_produced,
            violation_rate=1.0,  # inadmissible: crossed a limit every shot
        ))
    agg = srv.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2)
    assert agg is None
    assert srv.last_rejected_all is True


def test_an_empty_buffer_is_not_reported_as_a_rejection():
    """The other half of the distinction, which is the point of having it."""
    srv = server(use_safety=True)
    agg = srv.aggregate_for("tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2)
    assert agg is None
    assert srv.last_rejected_all is False


def test_the_cold_start_meta_carries_the_rejection_flags():
    """A flag nobody reads is a comment.

    The server has recorded these since the NaN-region bug and nothing
    outside it ever looked, so an arm that rejected every round would have
    been reported as an ordinary federated arm that happened to match
    scratch -- which is exactly the shape of the result being chased.
    """
    from pathlib import Path

    runner = (Path(__file__).resolve().parents[1]
              / "hfmarl" / "experiments" / "runner.py").read_text(
                  encoding="utf-8")
    assert '"aggregate_rejected_all"' in runner
    assert "last_rejected_all" in runner

    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "exp_coldstart.py").read_text(encoding="utf-8")
    assert "aggregate_rejected_all" in script
    assert "REJECTED EVERY CLIENT" in script


def test_rejected_rounds_accumulate_across_calls():
    """`last_rejected_all` describes one call and nothing else.

    An arm whose aggregate was empty for twenty rounds and fine on the
    twenty-first reports last_rejected_all=False -- true, and useless: those
    twenty rounds were isolated training wearing a federated label. The
    counter is what makes that visible in the run.
    """
    srv = server(use_safety=True)
    for _ in range(3):
        for d in ("iter_like", "sparc_like"):
            u = update(d, 1.0)
            srv.publish(ClientUpdate(
                device=u.device, cluster=u.cluster, weights=u.weights,
                state=u.state, n_samples=u.n_samples,
                round_produced=u.round_produced, violation_rate=1.0))
        assert srv.aggregate_for(
            "tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2) is None
    assert srv.rejected_rounds == 3

    # A successful round leaves the running total alone and clears the flag.
    for d in ("iter_like", "sparc_like"):
        srv.publish(update(d, 1.0))
    assert srv.aggregate_for(
        "tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 3) is not None
    assert srv.last_rejected_all is False
    assert srv.rejected_rounds == 3


def test_an_empty_buffer_does_not_count_as_a_rejected_round():
    srv = server(use_safety=True)
    assert srv.aggregate_for(
        "tcv_like", STATES["tcv_like"], ONLY_CLUSTER, 2) is None
    assert srv.rejected_rounds == 0
