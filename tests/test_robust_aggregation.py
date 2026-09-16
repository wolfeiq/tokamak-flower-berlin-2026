"""Alignment and robust means -- the two defences, tested separately.

They fix different failures and it matters that the tests say which is which:
alignment fixes an average of healthy-but-mismatched clients, the robust mean
fixes an average that one bad client has captured. A test that only shows "the
aggregate got better" would not distinguish them.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.agents.policy import make_policy
from hfmarl.federation.robust import (
    align_to,
    centered_clip,
    permute_hidden,
    weighted_geometric_median,
)

OBS, HID, ACT = 9, 16, 2


def policy_vec(seed: int) -> np.ndarray:
    return make_policy(OBS, ACT, hidden=HID, seed=seed).get_flat()


def outputs(vec: np.ndarray, obs: np.ndarray) -> np.ndarray:
    p = make_policy(OBS, ACT, hidden=HID, seed=0)
    p.set_flat(vec)
    return np.array([p.act(o) for o in obs])


# -- permutation alignment -----------------------------------------------


def test_permuting_hidden_units_does_not_change_the_function():
    """The invariance the whole alignment step rests on. If this were false,
    aligning would silently change each client's policy before merging it."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, OBS))
    v = policy_vec(1)
    perm = rng.permutation(HID)
    w = permute_hidden(v, perm, OBS, HID, ACT)

    assert not np.allclose(v, w), "the permutation should move the parameters"
    assert np.allclose(outputs(v, obs), outputs(w, obs), atol=1e-12)


def test_alignment_recovers_a_known_permutation():
    rng = np.random.default_rng(2)
    v = policy_vec(3)
    perm = rng.permutation(HID)
    scrambled = permute_hidden(v, perm, OBS, HID, ACT)
    assert np.allclose(align_to(v, scrambled, OBS, HID, ACT), v, atol=1e-12)


def test_averaging_a_scrambled_copy_is_broken_and_alignment_fixes_it():
    """The failure mode in one test.

    A client and a permuted copy of itself are the SAME policy. Their naive
    mean should therefore be that policy -- and is not, badly. Aligning first
    makes the mean exact.
    """
    rng = np.random.default_rng(4)
    obs = rng.normal(size=(64, OBS))
    v = policy_vec(5)
    scrambled = permute_hidden(v, rng.permutation(HID), OBS, HID, ACT)

    naive = 0.5 * (v + scrambled)
    aligned = 0.5 * (v + align_to(v, scrambled, OBS, HID, ACT))

    truth = outputs(v, obs)
    err_naive = np.abs(outputs(naive, obs) - truth).mean()
    err_aligned = np.abs(outputs(aligned, obs) - truth).mean()

    assert err_aligned < 1e-12
    assert err_naive > 100 * max(err_aligned, 1e-12)


# -- robust mean ----------------------------------------------------------


def test_geometric_median_resists_a_client_the_mean_cannot():
    """A mean has a breakdown point of zero: one client, arbitrarily far, moves
    it arbitrarily far. Push that client further and the gap grows."""
    good = np.zeros((3, 40))
    good[1] += 0.01
    good[2] -= 0.01

    for magnitude in (1e2, 1e6):
        points = np.vstack([good, np.full((1, 40), magnitude)])
        w = np.full(4, 0.25)
        mean = np.average(points, axis=0, weights=w)
        gm = weighted_geometric_median(points, w)
        assert np.linalg.norm(gm) < np.linalg.norm(mean) / 10


def test_geometric_median_keeps_the_similarity_weighting():
    """It has to accept weights, or switching to a robust rule would discard
    SPEC.md 4b -- the physics weighting -- to get robustness."""
    points = np.array([np.full(8, 0.0), np.full(8, 1.0), np.full(8, 2.0)])
    near = weighted_geometric_median(points, np.array([0.8, 0.1, 0.1]))
    far = weighted_geometric_median(points, np.array([0.1, 0.1, 0.8]))
    assert near[0] < far[0]


def test_two_points_with_unequal_weights_return_the_heavier_one():
    """AUDIT #9. This test previously asserted the weighted mean, which is
    the WRONG answer: minimising w0*|z-p0| + w1*|z-p1| over the segment puts
    the optimum at the heavier point whenever the weights differ. For [0, 10]
    with weights [.25, .75] the mean scores 3.75 and the optimum, 10, scores
    2.50. A test can encode a bug as firmly as the code does."""
    points = np.array([np.zeros(4), np.full(4, 10.0)])
    got = weighted_geometric_median(points, np.array([0.25, 0.75]))
    assert np.allclose(got, 10.0)

    def objective(z):
        return 0.25 * abs(z - 0.0) + 0.75 * abs(z - 10.0)

    assert objective(got[0]) < objective(7.5)


def test_two_points_with_equal_weights_return_the_midpoint():
    """Only equal weights make the whole segment optimal."""
    points = np.array([np.zeros(4), np.full(4, 10.0)])
    got = weighted_geometric_median(points, np.array([0.5, 0.5]))
    assert np.allclose(got, 5.0)


# -- centred clipping ------------------------------------------------------


def test_clipping_bounds_the_deviation_and_reports_that_it_acted():
    centre = np.zeros(16)
    points = np.vstack([np.full((3, 16), 0.1), np.full((1, 16), 50.0)])
    clipped, n = centered_clip(points, centre, factor=2.0)

    assert n == 1, "only the outlier should be clipped"
    radius = 2.0 * float(np.median(np.linalg.norm(points - centre, axis=1)))
    assert np.linalg.norm(clipped[-1] - centre) <= radius + 1e-9
    assert np.allclose(clipped[:3], points[:3]), "inliers must pass through"


def test_clipping_is_a_no_op_when_every_client_agrees():
    centre = np.zeros(8)
    points = np.tile(np.full(8, 0.3), (4, 1))
    clipped, n = centered_clip(points, centre, factor=2.0)
    assert n == 0
    assert np.allclose(clipped, points)


@pytest.mark.parametrize("rule", ["mean", "median", "geomedian"])
def test_every_rule_is_reachable_by_name(rule):
    """A typo in a rule name must fail loudly rather than silently falling back
    to the mean, which is the thing being replaced."""
    from hfmarl.devices.registry import encoded_states
    from hfmarl.federation.server import FedBuffServer
    from hfmarl.federation.similarity import ClientUpdate

    states = encoded_states()
    s = FedBuffServer(bandwidth=1.0, rule=rule, align=False, clip_factor=None)
    for i, d in enumerate(["iter_like", "sparc_like", "tcv_like"]):
        s.publish(ClientUpdate(device=d, cluster="thermal",
                               weights={"flat": np.full(6, float(i))},
                               state=states[d], n_samples=1, round_produced=1))
    assert s.aggregate_for("diiid_like", states["diiid_like"], "thermal", 1) is not None

    s.rule = "mena"
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        s.aggregate_for("diiid_like", states["diiid_like"], "thermal", 1)
