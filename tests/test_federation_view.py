"""The federation view's data layer.

The app's whole claim to being worth looking at is that the picture is computed
by the aggregation code rather than drawn to look plausible. These tests pin
the properties a viewer would read off the scene and be wrong about if the
visualiser ever started inventing numbers.

No TORAX, no GPU -- ``viz.federation_data`` is pure NumPy, same as the modules
it calls.
"""

import numpy as np
import pytest

from hfmarl.devices import registry
from viz import federation_data as fd


def test_channels_are_the_actuator_clusters():
    """The three coils in the scene are the three federation channels.

    If a fourth cluster is ever added to the registry -- the exhaust cluster of
    SPEC.md Phase 7 is the obvious candidate -- the scene would silently stop
    drawing it. Fail here instead.
    """
    from_registry = {
        a.cluster
        for d in registry.all_devices()
        for a in d.actuators
    }
    assert set(fd.CLUSTERS) == from_registry


def test_every_device_appears_in_every_channel():
    graph = fd.build_graph()
    for d in graph.devices:
        assert set(d.links) == set(fd.CLUSTERS), d.key


def test_payload_matches_the_measured_claim():
    """README states 776 bytes for the two-actuator thermal policy.

    It is a stated deliverable (SPEC.md §8), so the app must not quote a
    different number from the one the policy measures.
    """
    graph = fd.build_graph()
    assert graph.diagnostics["thermal"].act_dim == 2
    assert graph.diagnostics["thermal"].n_params == 194
    assert graph.diagnostics["thermal"].payload_bytes == 776


def test_unavailable_actuators_are_shown_not_dropped():
    """ICRH is defined and excluded, and the diagram says so.

    ``actuators_for`` hides it by default. A scene built on that default would
    show a thermal cluster that had always had exactly two agents, which is not
    what the registry records.
    """
    graph = fd.build_graph()
    thermal = graph.device("sparc_like").links["thermal"]
    names = {a.name for a in thermal.agents}
    assert names == {"aux_heat", "ecrh", "icrh"}
    assert not next(a for a in thermal.agents if a.name == "icrh").available
    # ...and the action dimension still counts only the live ones.
    assert graph.diagnostics["thermal"].act_dim == 2


def test_weights_are_personalised_per_target():
    """ITER and TCV must receive different mixtures of the same buffer.

    If these ever coincide, the weighting has stopped depending on the target
    and the app is drawing a single global model -- which is the thing
    SPEC.md §4b exists to replace.
    """
    a = fd.build_graph(target="iter_like")
    b = fd.build_graph(target="tcv_like")
    wa = [a.device(k).links["thermal"].weight for k in ("iter_like", "tcv_like")]
    wb = [b.device(k).links["thermal"].weight for k in ("iter_like", "tcv_like")]
    assert not np.allclose(wa, wb)


def test_weights_sum_to_one_per_channel():
    graph = fd.build_graph()
    for cluster in fd.CLUSTERS:
        total = sum(d.links[cluster].weight for d in graph.devices)
        assert total == pytest.approx(1.0)


def test_a_violating_peer_is_refused_not_downweighted():
    """Past 50% violations the admissibility gate fires.

    The scene draws that differently from a small weight, so the distinction
    has to survive the trip through this module.
    """
    graph = fd.build_graph(violation_rate={"tcv_like": 0.9})
    link = graph.device("tcv_like").links["thermal"]
    assert not link.admissible
    assert not link.excluded  # it was buffered and then refused
    assert link.weight == 0.0


def test_excluded_self_is_distinguishable_from_refused():
    """``include_self=False`` is a configuration choice, not a gate firing."""
    graph = fd.build_graph(target="iter_like", include_self=False)
    link = graph.device("iter_like").links["thermal"]
    assert link.excluded
    assert not link.admissible


def test_staleness_shrinks_a_weight():
    fresh = fd.build_graph()
    stale = fd.build_graph(round_age={"diiid_like": 30})
    assert (
        stale.device("diiid_like").links["thermal"].weight
        < fresh.device("diiid_like").links["thermal"].weight
    )


def test_uniform_arm_drops_the_distance_dependence():
    """``use_similarity=False`` is baseline 2, and must actually be uniform."""
    graph = fd.build_graph(use_similarity=False)
    weights = [d.links["thermal"].weight for d in graph.devices]
    assert np.allclose(weights, weights[0])


def test_scene_payload_is_json_serialisable():
    """The scene is injected into JavaScript by ``json.dumps``.

    A NumPy scalar anywhere in the payload raises there rather than here, in a
    Streamlit callback, with a traceback that says nothing about which field.
    """
    import json

    json.dumps(fd.scene_payload(fd.build_graph()), allow_nan=False)


def test_all_silent_preserves_channels_without_aggregates():
    graph = fd.build_graph(participants=[])
    assert set(graph.diagnostics) == set(fd.CLUSTERS)
    assert graph.diagnostics["thermal"].payload_bytes == 776
    assert all(not diag.aggregated for diag in graph.diagnostics.values())
    assert all(link.excluded and link.weight == 0 for d in graph.devices for link in d.links.values())


def test_silent_target_receives_selected_peers_aggregate():
    graph = fd.build_graph(target="iter_like", participants=["tcv_like", "sparc_like"])
    assert all(link.excluded for link in graph.device("iter_like").links.values())
    for c in fd.CLUSTERS:
        assert graph.diagnostics[c].aggregated
        assert sum(d.links[c].weight for d in graph.devices) == pytest.approx(1.0)


def test_participation_sweep_preserves_gates_and_sidebar_settings():
    import itertools
    settings = dict(target="tcv_like", include_self=False, use_similarity=False,
                    violation_rate={"sparc_like": 0.9}, round_age={"diiid_like": 4})
    graph = fd.build_graph(**settings)
    sweep = fd.participation_sweep(graph)
    keys = [d.key for d in graph.devices]
    assert len(sweep) == 16
    for bits in itertools.product([False, True], repeat=len(keys)):
        participants = [k for k, enabled in zip(keys, bits) if enabled]
        expected = fd.build_graph(participants=participants, **settings)
        actual = sweep[fd.participation_mask(keys, participants)]
        for d in expected.devices:
            for c, link in d.links.items():
                assert actual["devices"][d.key][c] == dict(
                    weight=link.weight, admissible=link.admissible, excluded=link.excluded)
        for c, diag in expected.diagnostics.items():
            assert actual["clusters"][c]["aggregated"] == diag.aggregated


def test_scene_starts_silent_with_geography_and_complete_states():
    payload = fd.scene_payload(fd.build_graph())
    assert payload["sending"] == []
    assert len(payload["participation"]) == 16
    for d in payload["devices"]:
        assert -180 <= d["lon"] <= 180
        assert -90 <= d["lat"] <= 90
        assert d["epsilon"] == pytest.approx(d["a"] / d["R"])
