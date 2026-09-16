"""Aggregation that survives a bad client -- and alignment that prevents one.

WHY A WEIGHTED MEAN IS THE WRONG DEFAULT HERE
---------------------------------------------
Two failure modes, and they need opposite fixes.

**Permutation mismatch.** Two one-hidden-layer MLPs that permute their hidden
units compute exactly the same function and average to something worse than
either. This is not a corrupted client; every client is fine and the mean is
still broken. Ainsworth et al., "Git Re-Basin" (arXiv:2209.04836), fix it by
permuting one model into alignment with a reference before merging.

For THIS architecture the fix is exact rather than approximate. Git Re-Basin
needs coordinate descent because permutations in adjacent layers couple; a
one-hidden-layer net has a single permutable layer, so the symmetry group is
exactly S_hidden and one linear assignment problem solves it optimally. 16
hidden units means a 16x16 LAP.

**A genuinely bad update.** A client whose search wandered somewhere useless
drags the mean with it in proportion to its weight, and the mean has a
breakdown point of zero -- one arbitrarily bad client moves it arbitrarily far.
Pillutla et al., "Robust Aggregation for Federated Learning"
(arXiv:1912.13445), replace it with the geometric median, computed by a
smoothed Weiszfeld iteration (RFA).

WHY THE GEOMETRIC MEDIAN AND NOT KRUM OR A TRIMMED MEAN
--------------------------------------------------------
Both are better known and both are arithmetically unavailable at this scale.
Krum scores each update by its distance to its n-f-2 nearest neighbours; with
three incumbents and one tolerated failure that is zero neighbours. A
coordinate-wise trimmed mean that drops one value from each end of three leaves
one value, which is the coordinate-wise median -- so it is not a separate
option, it is the median under another name (both are in Yin et al., ICML
2018).

The geometric median also keeps something the rank statistics throw away: it
accepts WEIGHTS, so the similarity weighting of SPEC.md 4b survives the switch
to a robust rule. A coordinate-wise median would discard the physics weighting
that is the entire contribution.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Permutation alignment
# ---------------------------------------------------------------------------


def _unpack(flat: np.ndarray, obs_dim: int, hidden: int, act_dim: int):
    i = obs_dim * hidden
    W1 = flat[:i].reshape(obs_dim, hidden)
    b1 = flat[i : i + hidden]
    i += hidden
    W2 = flat[i : i + hidden * act_dim].reshape(hidden, act_dim)
    b2 = flat[i + hidden * act_dim :]
    return W1, b1, W2, b2


def _pack(W1, b1, W2, b2) -> np.ndarray:
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])


def permute_hidden(flat: np.ndarray, perm: np.ndarray, obs_dim: int,
                   hidden: int, act_dim: int) -> np.ndarray:
    """Reorder hidden units. The network computes exactly the same function.

    That invariance is the whole point: it means the reordering is free to do,
    and it means an un-reordered average is comparing coordinates that do not
    correspond to each other.
    """
    W1, b1, W2, b2 = _unpack(flat, obs_dim, hidden, act_dim)
    return _pack(W1[:, perm], b1[perm], W2[perm, :], b2)


def align_to(reference: np.ndarray, candidate: np.ndarray, obs_dim: int,
             hidden: int, act_dim: int) -> np.ndarray:
    """Permute `candidate`'s hidden units into alignment with `reference`.

    Maximises the inner product between the two parameter vectors over all
    permutations -- Git Re-Basin's "weight matching", which for one hidden
    layer is a single linear assignment problem and therefore exact.
    """
    rW1, rb1, rW2, _ = _unpack(reference, obs_dim, hidden, act_dim)
    cW1, cb1, cW2, _ = _unpack(candidate, obs_dim, hidden, act_dim)
    # cost[i, j] = how well candidate unit j matches reference unit i
    cost = rW1.T @ cW1 + np.outer(rb1, cb1) + rW2 @ cW2.T
    rows, cols = linear_sum_assignment(-cost)
    perm = np.empty(hidden, dtype=int)
    perm[rows] = cols
    return permute_hidden(candidate, perm, obs_dim, hidden, act_dim)


# ---------------------------------------------------------------------------
# Robust means
# ---------------------------------------------------------------------------


def weighted_geometric_median(points: np.ndarray, weights: np.ndarray,
                              eps: float = 1e-8, iters: int = 100,
                              tol: float = 1e-10) -> np.ndarray:
    """Smoothed Weiszfeld iteration (RFA, Pillutla et al. 2019).

    Minimises the weighted sum of EUCLIDEAN distances rather than squared
    distances, so a single far-away client pulls on the result with bounded
    force instead of in proportion to how far away it is. `eps` smooths the
    1/||.|| reweighting so the iteration cannot divide by zero when the
    estimate lands exactly on a point -- which happens routinely with three
    clients.

    Degrades exactly, not gracefully: one point returns that point; two points
    return the HEAVIER one, which is the true optimum unless the weights are
    equal, in which case the whole segment is optimal and the midpoint
    represents it. The weighted mean is wrong here and used to be the
    fallback -- see AUDIT #9 in the two-point branch.
    """
    P = np.atleast_2d(np.asarray(points, float))
    w = np.asarray(weights, float)
    if P.shape[0] == 0:
        raise ValueError("no points to aggregate")
    if P.shape[0] == 1:
        return P[0].copy()
    if P.shape[0] == 2:
        # AUDIT #9. The weighted mean is NOT the geometric median of two
        # points unless the weights are equal. Minimising w0*|z-p0| +
        # w1*|z-p1| over the segment puts the optimum at the heavier
        # point: for [0, 10] with weights [.25, .75] the mean scores
        # 3.75 and the correct answer, 10, scores 2.50. Only equal
        # weights make the whole segment optimal, and the midpoint is
        # the canonical representative of it.
        if np.isclose(w[0], w[1]):
            return np.average(P, axis=0, weights=w)
        return P[int(np.argmax(w))].copy()

    z = np.average(P, axis=0, weights=w)
    for _ in range(iters):
        d = np.linalg.norm(P - z, axis=1)
        beta = w / np.maximum(d, eps)
        z_new = (beta[:, None] * P).sum(axis=0) / beta.sum()
        if np.linalg.norm(z_new - z) <= tol * max(1.0, np.linalg.norm(z)):
            return z_new
        z = z_new
    return z


def coordinate_median(points: np.ndarray) -> np.ndarray:
    """Coordinate-wise median (Yin et al., ICML 2018).

    Kept for comparison. Note it discards the weights entirely, which means it
    also discards SPEC.md 4b -- so it is an ablation, not a candidate default.
    """
    return np.median(np.atleast_2d(np.asarray(points, float)), axis=0)


def centered_clip(points: np.ndarray, center: np.ndarray,
                  factor: float = 2.0) -> tuple[np.ndarray, int]:
    """Clip each update's deviation from `center` to a radius.

    Centered clipping (Karimireddy et al., ICML 2021) bounds how far any single
    client can move the result, whatever it sends. Composes with the weighted
    mean, so unlike a rank statistic it leaves the similarity weighting intact.

    The radius is `factor` x the median deviation rather than a fixed constant:
    a parameter vector's natural scale changes as training proceeds, and a
    constant tuned at round 1 is wrong by round 50. Returns the clipped points
    and how many were actually clipped -- an aggregation that cannot say
    whether it did anything has not been measured.
    """
    P = np.atleast_2d(np.asarray(points, float))
    delta = P - np.asarray(center, float)
    norms = np.linalg.norm(delta, axis=1)
    radius = factor * float(np.median(norms))
    if radius <= 0:
        # AUDIT #9. A zero median deviation means over half the rows sit
        # exactly on the reference -- which common initialisation makes
        # routine early on -- and the old code took that as a licence to
        # clip nothing, passing an arbitrarily large outlier through
        # untouched and reporting zero clips. The mean deviation is
        # non-zero whenever any row differs, so the bound survives.
        radius = factor * float(np.mean(norms))
    if radius <= 0:
        # Every row is the reference. Nothing to bound.
        return P.copy(), 0
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return np.asarray(center, float) + delta * scale[:, None], int((scale < 1).sum())
