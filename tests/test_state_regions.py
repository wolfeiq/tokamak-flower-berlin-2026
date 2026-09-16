"""Weighting by overlap of operating regions rather than distance between points.

The measurement that forced this (`scripts/measure_operating_points.py`, easy):

    device       own excursion   nearest peer
    diiid_like           2.055          1.421
    tcv_like             1.806          1.421

For two of four devices the within-device excursion exceeds the between-device
distance, so a distance between two tabulated points reports which point got
tabulated, not a physical relationship.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.federation.similarity import (
    ClientUpdate,
    aggregation_weights,
    overlap_weight,
    region_from_state,
    region_from_states,
    similarity_distance,
    similarity_weight,
)
from hfmarl.physics.dimensionless import DimensionlessState


def state(rho=3e-3, nu=0.05, beta=1.5, q=4.0):
    return DimensionlessState(rho_star=rho, nu_star=nu, beta_N=beta, q95=q,
                              mach=0.0)


def spread_of(centre, factor, n=9):
    """States fanned out around a centre, log-spaced in rho* and nu*."""
    mult = np.geomspace(1 / factor, factor, n)
    return [state(rho=centre.rho_star * m, nu=centre.nu_star * m,
                  beta=centre.beta_N, q=centre.q95)
            for m in mult]


# -- the reduction to the old rule ---------------------------------------


def test_two_point_regions_reproduce_the_point_kernel():
    """The old behaviour has to be the zero-extent special case, not a separate
    code path -- otherwise every number computed before this existed becomes
    incomparable with every number computed after."""
    a, b = state(rho=2e-3), state(rho=8e-3)
    h = 1.0
    d = similarity_distance(a, b)
    got = overlap_weight(region_from_state(a), region_from_state(b), h)
    assert got == pytest.approx(similarity_weight(d, h), rel=1e-9)


def test_a_region_built_from_one_state_has_no_extent():
    r = region_from_state(state())
    assert r.spread == pytest.approx(0.0)
    assert r.n == 1


# -- the behaviour the measurement demands --------------------------------


def test_broad_regions_that_straddle_each_other_score_higher_than_their_means():
    """The diiid/tcv case. Two devices whose means are far apart but whose
    operating envelopes overlap are, physically, telling each other something.
    A point kernel cannot see that; this must."""
    a_centre, b_centre = state(rho=2e-3, nu=0.02), state(rho=2e-2, nu=0.5)
    h = 1.0

    narrow = overlap_weight(region_from_state(a_centre),
                            region_from_state(b_centre), h)
    broad = overlap_weight(region_from_states(spread_of(a_centre, 6.0)),
                           region_from_states(spread_of(b_centre, 6.0)), h)
    assert broad > narrow


def test_a_peer_that_roams_is_penalised_when_you_do_not():
    """Same centre, very different extent. A machine that sweeps a decade in
    nu* is not reporting on your narrow corner of it, and the Bhattacharyya
    prefactor is what expresses that -- the exponent alone would not."""
    centre = state()
    h = 0.35
    narrow = region_from_states(spread_of(centre, 1.05))
    wide = region_from_states(spread_of(centre, 8.0))

    same = overlap_weight(narrow, narrow, h)
    mismatched = overlap_weight(narrow, wide, h)
    assert mismatched < same


def test_overlap_is_symmetric_and_bounded():
    h = 1.0
    a = region_from_states(spread_of(state(rho=2e-3), 3.0))
    b = region_from_states(spread_of(state(rho=1e-2), 2.0))
    ab, ba = overlap_weight(a, b, h), overlap_weight(b, a, h)
    assert ab == pytest.approx(ba)
    assert 0.0 < ab <= 1.0 + 1e-12
    assert overlap_weight(a, a, h) == pytest.approx(1.0)


def test_bandwidth_is_a_floor_not_the_whole_width():
    """A device measured from nearly identical shots must not become infinitely
    selective. Without a floor its variance is ~0 and the kernel collapses."""
    centre = state()
    identical = region_from_states([centre] * 5)
    other = region_from_states([state(rho=4e-3)] * 5)
    assert overlap_weight(identical, other, 1.0) > 0.0
    with pytest.raises(ValueError):
        overlap_weight(identical, other, 0.0)


# -- end to end through the aggregation ----------------------------------


def test_aggregation_uses_regions_when_both_ends_have_them():
    target = state(rho=2e-2, nu=0.5)
    near, far = state(rho=1.5e-2, nu=0.4), state(rho=1.5e-3, nu=0.02)

    def upd(name, st, region=None):
        return ClientUpdate(device=name, cluster="thermal",
                            weights={"flat": np.zeros(3)}, state=st,
                            n_samples=5, round_produced=1, region=region)

    points = [upd("near", near), upd("far", far)]
    regions = [upd("near", near, region_from_states(spread_of(near, 5.0))),
               upd("far", far, region_from_states(spread_of(far, 5.0)))]

    w_points = aggregation_weights(points, target, current_round=1)
    w_regions = aggregation_weights(
        regions, target, current_round=1,
        target_region=region_from_states(spread_of(target, 5.0)))

    # The near peer still wins either way; the far one is no longer dismissed,
    # because with real extent the two envelopes genuinely overlap.
    assert w_points[0] > w_points[1]
    assert w_regions[1] > w_points[1]


def test_falling_back_to_points_when_a_peer_has_no_region():
    """Old updates and the cold-start joiner have no measured region yet. They
    must still aggregate, on the point kernel, rather than raise."""
    target = state()
    u = ClientUpdate(device="d", cluster="thermal",
                     weights={"flat": np.zeros(3)}, state=state(rho=5e-3),
                     n_samples=5, round_produced=1, region=None)
    w = aggregation_weights([u], target, current_round=1,
                            target_region=region_from_state(target))
    assert np.isclose(w.sum(), 1.0)


def test_similarity_weight_still_exists_for_the_ablation():
    """The point kernel is the thing being replaced, so it has to stay
    runnable: 'regions helped' needs the point number beside it."""
    assert similarity_weight(1.0, 1.0) == pytest.approx(np.exp(-0.5))
