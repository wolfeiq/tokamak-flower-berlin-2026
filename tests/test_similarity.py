"""Similarity metric and aggregation weighting -- SPEC.md §4b."""

import numpy as np
import pytest

from hfmarl.federation.similarity import (
    ClientUpdate,
    aggregation_weights,
    describe_device_set,
    similarity_distance,
    similarity_weight,
    staleness_factor,
    suggest_bandwidth,
)
from hfmarl.physics.dimensionless import DimensionlessState

ITER = DimensionlessState(1.52e-3, 0.017, 2.89, 2.80)
SPARC = DimensionlessState(2.32e-3, 0.014, 1.30, 2.43)
DIIID = DimensionlessState(6.60e-3, 0.024, 3.24, 3.23)
TCV = DimensionlessState(1.74e-2, 0.053, 2.24, 3.07)


def _update(name, state, **kw):
    return ClientUpdate(
        device=name, cluster=kw.pop("cluster", "thermal"),
        weights={"W1": np.zeros((2, 2))}, state=state,
        n_samples=kw.pop("n_samples", 100),
        round_produced=kw.pop("round_produced", 0), **kw,
    )


# -- metric properties ---------------------------------------------------


def test_distance_is_zero_to_self():
    assert similarity_distance(ITER, ITER) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_symmetric():
    assert similarity_distance(ITER, TCV) == pytest.approx(
        similarity_distance(TCV, ITER), rel=1e-12
    )


def test_distance_obeys_triangle_inequality():
    """It is a weighted Euclidean norm, so this must hold exactly."""
    d_it = similarity_distance(ITER, TCV)
    d_id = similarity_distance(ITER, DIIID)
    d_dt = similarity_distance(DIIID, TCV)
    assert d_it <= d_id + d_dt + 1e-9


def test_tcv_is_furthest_from_iter():
    """TCV is the designed negative control -- it must be the most distant."""
    d = {n: similarity_distance(ITER, s) for n, s in
         [("sparc", SPARC), ("diiid", DIIID), ("tcv", TCV)]}
    assert d["tcv"] == max(d.values())


def test_no_single_parameter_dominates_the_metric():
    """Regression test for a real bug.

    Before per-parameter scaling, beta_N differences (O(1) in raw units)
    swamped log-space rho* differences (O(0.2)), so two machines a decade
    apart in size could be called similar. Here rho* is varied by a full
    decade with everything else fixed; that must register as a large distance.
    """
    a = DimensionlessState(1e-3, 0.02, 2.5, 3.0)
    b = DimensionlessState(1e-2, 0.02, 2.5, 3.0)  # one decade in rho* only
    assert similarity_distance(a, b) > 1.0


def test_weight_decreases_with_distance():
    assert similarity_weight(0.0) == pytest.approx(1.0)
    assert 0.0 < similarity_weight(2.0) < similarity_weight(1.0) < 1.0


def test_weight_rejects_bad_bandwidth():
    with pytest.raises(ValueError):
        similarity_weight(1.0, bandwidth=0.0)


def test_suggest_bandwidth_is_positive_and_finite():
    bw = suggest_bandwidth([ITER, SPARC, DIIID, TCV])
    assert 0.0 < bw < 10.0


# -- staleness -----------------------------------------------------------


def test_staleness_decays_with_rounds():
    u = _update("a", ITER, round_produced=0)
    assert staleness_factor(u, 0, 0) == pytest.approx(1.0)
    assert staleness_factor(u, 3, 0) < staleness_factor(u, 1, 0) < 1.0


def test_config_epoch_penalises_beyond_wall_clock():
    """SPEC.md §5: a device that changed its wall is stale however recent."""
    fresh = _update("a", ITER, round_produced=5, config_epoch=0)
    same_time_same_config = _update("b", ITER, round_produced=5, config_epoch=1)
    assert staleness_factor(fresh, 5, 1) < staleness_factor(same_time_same_config, 5, 1)


def test_staleness_is_bounded():
    u = _update("a", ITER, round_produced=0, config_epoch=0)
    for r in (0, 1, 10, 1000):
        assert 0.0 < staleness_factor(u, r, 3) <= 1.0


# -- aggregation ---------------------------------------------------------


def test_role_matching_is_enforced_not_assumed():
    """Mixing clusters is the exact failure role matching exists to prevent."""
    ups = [_update("a", ITER, cluster="thermal"),
           _update("b", DIIID, cluster="particle")]
    with pytest.raises(ValueError, match="mixed clusters"):
        aggregation_weights(ups, ITER, current_round=0)


def test_weights_sum_to_one():
    ups = [_update(n, s) for n, s in [("i", ITER), ("s", SPARC), ("t", TCV)]]
    w = aggregation_weights(ups, ITER, current_round=0)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)


def test_similar_device_outweighs_distant_one():
    """The whole claim of §4b, as a test."""
    ups = [_update("diiid", DIIID), _update("tcv", TCV)]
    w = aggregation_weights(ups, ITER, current_round=0)
    assert w[0] > w[1]


def test_uniform_mode_reproduces_baseline_two():
    """use_similarity=False must give equal weights -- baseline 2 in Phase 5."""
    ups = [_update("i", ITER), _update("t", TCV)]
    w = aggregation_weights(ups, ITER, current_round=0, use_similarity=False)
    assert w[0] == pytest.approx(w[1])


def test_empty_buffer_returns_empty():
    assert aggregation_weights([], ITER, current_round=0).shape == (0,)


def test_all_zero_weights_fall_back_to_uniform_not_nan():
    """A stalled-but-defined result beats silently emitting NaN."""
    ups = [_update("t", TCV, round_produced=0), _update("t2", TCV, round_produced=0)]
    w = aggregation_weights(ups, ITER, current_round=10**6,
                            current_config_epoch=50, bandwidth=1e-6)
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)


# -- diagnostics ---------------------------------------------------------


def test_describe_warns_when_every_peer_is_ignored():
    text = describe_device_set(
        {"i": ITER, "s": SPARC, "d": DIIID, "t": TCV}, bandwidth=0.01
    )
    assert "WARNING" in text and "collapse onto baseline 1" in text


def test_describe_warns_when_all_peers_are_equal():
    text = describe_device_set(
        {"i": ITER, "s": SPARC, "d": DIIID, "t": TCV}, bandwidth=1e4
    )
    assert "WARNING" in text and "collapse onto baseline 2" in text


def test_describe_reports_ok_at_suggested_bandwidth():
    text = describe_device_set({"i": ITER, "s": SPARC, "d": DIIID, "t": TCV})
    assert "OK: weights discriminate" in text
