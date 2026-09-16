"""Physics-informed aggregation weights -- SPEC.md §4b.

FedAvg weights clients by sample count. That is an arbitrary rule with no
physics in it: a device that ran many shots in a regime unlike yours should
not dominate your update. Here the weight is set by *distance in dimensionless
parameter space* instead, which similarity theory justifies directly -- a
device operating near your (rho*, nu*, beta, q) is, by Connor-Taylor, running
approximately your plasma, and its gradients transfer.

The spec notes no prior work does this. Nearest relatives are inertia-weighted
FedAvg for grid stability (a static per-node physical property) and
graph-adjacency aggregation in traffic forecasting (network topology). Neither
uses a similarity metric derived from scale invariance.

This module is pure NumPy and has no TORAX or Flower dependency, so the
weighting scheme can be tested and characterised on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hfmarl.physics.dimensionless import DimensionlessState

# Which components are compared in log space.
#
# rho* and nu* are positive-definite and span decades across the device set
# (rho* runs 1.5e-3 for ITER-like to 1.7e-2 for TCV-like). A linear difference
# would be numerically dominated by the largest machine and would make every
# small device equidistant from every large one. beta_N and q are O(1) and
# comparable across devices, so they are compared linearly.
_LOG_SCALED = {"rho_star": True, "nu_star": True, "beta_N": False, "q95": False, "mach": False}

# Characteristic scale of each component, i.e. "how big a difference is a big
# difference" for that parameter. Every component is divided by its scale
# before entering the norm.
#
# This is NOT cosmetic. Without it the metric is incoherent: log10(rho*)
# differences across our device set are ~0.2-1.0, while beta_N differences are
# ~1.5 in raw units, so beta_N alone would decide every distance and two
# machines a decade apart in size could be called "similar" because they
# happened to share a beta. Scales put all five on a common footing, after
# which DEFAULT_WEIGHTS expresses genuine physics priority rather than an
# accident of units.
#
#   rho*, nu*  : 1.0  -- one decade is a large difference (already in log10)
#   beta_N     : 1.5  -- spans roughly 1-4 across accessible scenarios
#   q95        : 1.5  -- spans roughly 2-5
#   mach       : 0.3  -- small absolute range
_SCALES = np.array([1.0, 1.0, 1.5, 1.5, 0.3], dtype=float)

# Per-parameter weights in the distance metric, applied AFTER scaling.
#
# rho* carries the most weight because it is the parameter that most sharply
# separates machine sizes and the one gyro-Bohm transport scaling turns on.
# nu* is next: it selects the collisionality regime (banana vs
# Pfirsch-Schlueter), and transport physics changes qualitatively across
# nu* = 1. beta and q matter but vary less between our devices.
#
# These are a judgement call, not derived. They are exposed so the Phase 5
# ablation can vary them; report sensitivity rather than presenting them as
# given.
# Mach's weight is 0.0, not 0.5: nothing in the codebase sets a nonzero Mach
# number (TORAX's toroidal rotation is not wired through), so a nonzero weight
# is silently inert and would start contributing the day rotation is added,
# changing every previously-computed distance without anyone deciding to.
# Set it deliberately when rotation is modelled.
DEFAULT_WEIGHTS = np.array([2.0, 1.5, 1.0, 1.0, 0.0], dtype=float)


def _to_metric_space(state: DimensionlessState, weights: np.ndarray) -> np.ndarray:
    """Map a dimensionless state into the space the distance is Euclidean in."""
    vec = state.as_vector()
    labels = DimensionlessState.labels()
    out = np.empty_like(vec)
    for i, label in enumerate(labels):
        v = vec[i]
        if _LOG_SCALED[label]:
            out[i] = np.log10(max(v, 1e-12))
        else:
            out[i] = v
    return (out / _SCALES) * np.sqrt(weights)


def similarity_distance(
    a: DimensionlessState,
    b: DimensionlessState,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted Euclidean distance between two operating points.

    Zero means identical plasmas in the Connor-Taylor sense. Order unity means
    roughly a decade apart in rho* -- i.e. genuinely different machines.
    """
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, float)
    if w.shape != (5,):
        raise ValueError(f"weights must have shape (5,), got {w.shape}")
    return float(np.linalg.norm(_to_metric_space(a, w) - _to_metric_space(b, w)))


def similarity_weight(distance: float, bandwidth: float = 1.0) -> float:
    """Gaussian kernel turning a distance into an aggregation weight.

        w = exp( -d^2 / (2 * bandwidth^2) )

    A Gaussian is chosen over a hard cutoff so the weighting is smooth and
    differentiable, and over 1/d because that diverges for identical devices.

    ``bandwidth`` is the one free parameter and it means something concrete:
    the distance at which a peer's contribution falls to exp(-1/2) = 0.61.

    The default of 1.0 is a starting point, not a result. Use
    ``suggest_bandwidth`` to set it from the actual device set, and sweep it in
    Phase 5 -- a bandwidth too small makes every device an island (federation
    does nothing, and baseline 3 collapses onto baseline 1), while one too
    large makes the weighting uniform (baseline 3 collapses onto baseline 2).
    Both failure modes silently destroy the experiment, so CHECK the weight
    matrix with ``describe_device_set`` before running Phase 5.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return float(np.exp(-(distance**2) / (2.0 * bandwidth**2)))


@dataclass(frozen=True)
class StateRegion:
    """Where a device operates, as a REGION rather than a point.

    WHY A POINT IS NOT ENOUGH, MEASURED
    -----------------------------------
    `scripts/measure_operating_points.py`, task `easy`:

        device       own excursion   nearest peer
        diiid_like           2.055          1.421
        iter_like            0.866          1.270
        sparc_like           0.226          1.270
        tcv_like             1.806          1.421

    For two of four devices, the distance the device travels across its own
    command envelope EXCEEDS its distance to its nearest peer. So
    `similarity_distance(a, b)` between two tabulated points is not a
    physical relationship; it is a report on which point happened to be
    tabulated. Averaging the trajectory instead only moves the arbitrariness
    -- the mean of a region wider than the gap between regions still throws
    away the fact that they overlap.

    Mean and per-component variance, both in the METRIC space -- log-scaled,
    scaled and weighted -- because that is the space the kernel measures in.
    Computing a variance in raw units and then comparing it to a log-scaled
    distance would be two different geometries wearing one name.
    """

    mean: np.ndarray  # (5,) in metric space
    var: np.ndarray  # (5,) diagonal variance in metric space
    n: int

    @property
    def spread(self) -> float:
        """Root-mean-square extent, comparable to a distance."""
        return float(np.sqrt(np.sum(self.var)))


def region_from_states(
    states: list[DimensionlessState],
    weights: np.ndarray | None = None,
) -> StateRegion:
    """Summarise the states a device actually visited.

    A single state gives a region of zero extent, which reduces the overlap
    kernel below to exactly the point kernel -- so the old behaviour is the
    special case of one observation, not a separate code path.

    SELF-AUDIT. Non-finite states are dropped, not averaged. A shot whose
    solve failed encodes to NaN, and one of those poisoned the whole
    region: mean and variance went NaN, `overlap_weight` returned NaN,
    every weight went NaN, and the server -- reading `any(w > 0)` as
    False -- returned no aggregate at all. Federation switched itself off
    for that round and said nothing, so the experiment would have
    reported that federating did not help when federating had not
    happened. Same class as AUDIT #6, written after reading it.

    Raises when NOTHING is usable, because the caller has to decide
    between falling back to a nominal point and skipping the round, and
    that decision must not be made silently here.
    """
    if not states:
        raise ValueError("no states to summarise")
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, float)
    pts = np.stack([_to_metric_space(s, w) for s in states])
    usable = np.isfinite(pts).all(axis=1)
    if not usable.any():
        raise ValueError(
            f"all {len(states)} states are non-finite; the device never "
            "produced a usable solve, so it has no measured operating region")
    pts = pts[usable]
    return StateRegion(
        mean=pts.mean(axis=0),
        var=pts.var(axis=0, ddof=0),
        n=int(usable.sum()),
    )


def region_from_state(state: DimensionlessState,
                      weights: np.ndarray | None = None) -> StateRegion:
    """A region of zero extent, for callers that still only have a point."""
    return region_from_states([state], weights)


def overlap_weight(a: StateRegion, b: StateRegion,
                   bandwidth: float = 1.0) -> float:
    """Bhattacharyya overlap of two operating regions, in (0, 1].

    The generalisation the measurement forces. `similarity_weight` is
    exp(-d^2 / 2h^2): a Gaussian kernel that ALREADY assumes every device has
    a characteristic extent -- it just assumes the same one, h, for all of
    them, and never measures it. Here each device brings its own.

    For diagonal Gaussians the Bhattacharyya coefficient factorises:

        BC = prod_i sqrt(2 s_ai s_bi / (s_ai^2 + s_bi^2))
             * exp(-(m_ai - m_bi)^2 / (4 (s_ai^2 + s_bi^2)))

    Two things fall out, and both are what the physics wants:

      * broad regions that straddle each other score high even when their
        MEANS are far apart -- which is the diiid/tcv case exactly;
      * a device whose extent is very different from yours is penalised by
        the prefactor even if the means coincide, because a machine that
        roams a decade in nu* is not telling you about your narrow corner
        of it.

    `bandwidth` enters as a variance FLOOR rather than as the whole width. It
    stops a device measured from a handful of nearly identical shots from
    reporting zero extent and becoming infinitely selective, and it keeps the
    one calibration knob the old kernel had. With both regions at zero extent
    this returns exactly `similarity_weight(d, bandwidth)`.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    # bandwidth^2 / 4, not bandwidth^2, so that two zero-extent regions
    # return EXACTLY similarity_weight(d, bandwidth). An approximate
    # reduction would make every number computed before regions existed
    # quietly incomparable with every number computed after.
    floor = bandwidth**2 / 4.0
    va = np.asarray(a.var, float) + floor
    vb = np.asarray(b.var, float) + floor
    sum_v = va + vb
    dm = np.asarray(a.mean, float) - np.asarray(b.mean, float)

    # Per-component log BC, summed -- in logs, so a five-way product of
    # small numbers cannot underflow to exactly zero and silently drop a peer
    # that the admissibility gates deliberately let through.
    log_bc = np.sum(
        0.25 * np.log(4.0 * va * vb / sum_v**2) - dm**2 / (4.0 * sum_v)
    )
    return float(np.exp(log_bc))


@dataclass(frozen=True)
class ClientUpdate:
    """One device's contribution to the buffer.

    ``config_epoch`` is the physics-informed staleness counter of SPEC.md §5:
    it increments whenever the device changes something that invalidates old
    gradients in a way wall-clock time does not capture -- wall material,
    divertor geometry, heating configuration. An update from before such a
    change is stale no matter how recently it arrived.
    """

    device: str
    cluster: str  # thermal | particle | current -- the federation channel
    weights: dict[str, np.ndarray]  # the policy delta
    state: DimensionlessState  # where this device was operating
    n_samples: int
    round_produced: int
    config_epoch: int = 0
    # Fraction of the shots that produced this update which crossed a
    # limit. Measured, catastrophe run, tcv_like: federating with peers
    # that never meet a limit raised TCV's own violation rate from 5.3%
    # to 24.4%. A peer that has never been near an edge has no reason to
    # approach one carefully, and averaging its confidence in is how that
    # travels. Defaults to 0.0 so old call sites keep working, but a
    # producer that leaves it unset is asserting a clean campaign.
    violation_rate: float = 0.0
    # The operating REGION the producing shots covered, not a point. See
    # StateRegion: for two of four devices the within-device excursion is
    # larger than the between-device distance, so a point cannot express
    # where this update came from. None falls back to the point kernel.
    region: StateRegion | None = None


def staleness_factor(
    update: ClientUpdate,
    current_round: int,
    current_config_epoch: int,
    alpha: float = 0.5,
    epoch_penalty: float = 0.5,
) -> float:
    """FedBuff staleness weighting, extended with physics-informed staleness.

    The standard FedBuff term (Nguyen et al., AISTATS 2022) is

        s_time = 1 / sqrt(1 + tau),     tau = rounds elapsed

    generalised here to ``(1 + tau)^-alpha``. On top of that, SPEC.md §5 asks
    for staleness that wall-clock time cannot express: an update produced
    before the device changed its wall, divertor or heating configuration
    describes a machine that no longer exists. Each intervening config epoch
    multiplies the weight by ``epoch_penalty``.

    Both factors are multiplicative and in (0, 1].
    """
    tau = max(0, current_round - update.round_produced)
    s_time = (1.0 + tau) ** (-alpha)
    n_epochs = max(0, current_config_epoch - update.config_epoch)
    s_config = epoch_penalty**n_epochs
    return float(s_time * s_config)


def regime_valid(state: DimensionlessState, nu_star_max: float = 1.0) -> bool:
    """Does the similarity premise hold for this plasma at all?

    Connor-Taylor scale invariance is a statement about plasmas in the same
    transport regime. nu* = 1 separates the banana (collisionless) branch
    from Pfirsch-Schlueter; across it the transport physics differs
    qualitatively, and weight sharing stops being justified by anything.

    This is a HARD gate, not a downweight. A Gaussian kernel says 'far, so
    count less'; this says 'the reason for counting you at all has failed'.

    ARMED BUT CANNOT FIRE YET, and that should be stated rather than
    discovered: `ClientUpdate.state` is currently the hand-written nominal
    operating point, which never moves, and all four devices sit at nu* <
    0.1. It becomes live the moment states are measured per shot -- the same
    re-derivation RUNBOOK 5 asks for.
    """
    return bool(np.isfinite(state.nu_star) and 0.0 < state.nu_star < nu_star_max)


def regime_fraction_inside(region: StateRegion,
                           weights: np.ndarray | None = None) -> float:
    """How much of a device's operating envelope is in the banana regime.

    WHY THE MEAN IS NOT ENOUGH, MEASURED
    ------------------------------------
    `regime_valid` tests a single nu*. On measured data that hides the thing
    it exists to catch: tcv_like reaches nu* = 2.58 at low power -- well
    across the banana / Pfirsch-Schlueter boundary -- while its mean over a
    command sweep is 0.63, comfortably inside. A gate on the mean would pass
    a device that spends part of every campaign in a different transport
    regime, and those shots are not similarity-justified at all.

    In metric space the nu* component is (weight/scale) * log10(nu*), so the
    regime boundary nu* = 1 sits at exactly zero whatever the positive weight
    and scale are. Under the region's Gaussian approximation the fraction
    inside is therefore Phi(-mean / sd), which is exact rather than sampled.

    A zero weight on nu* makes the question unanswerable rather than false,
    so it returns 1.0 -- refusing every peer because a weight was switched
    off would be a silent, total change of behaviour.
    """
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, float)
    idx = list(_LOG_SCALED).index("nu_star")
    if w[idx] <= 0:
        return 1.0
    mean = float(region.mean[idx])
    sd = float(np.sqrt(region.var[idx]))
    if sd <= 0:
        return 1.0 if mean < 0 else 0.0
    # P(X < 0) for X ~ N(mean, sd^2); 0 is the nu* = 1 boundary.
    from math import erfc

    return float(0.5 * erfc(mean / (sd * np.sqrt(2.0))))


def regime_factor(region: StateRegion | None,
                  weights: np.ndarray | None = None) -> float:
    """Graded version of the regime gate, in [0, 1].

    Same shape as `safety_factor`: graded, with the hard refusal left to the
    admissibility check in `aggregation_weights`. A peer that spends a tenth
    of its campaign collisional is weaker evidence, not no evidence; one that
    spends most of it there is telling you about different physics.
    """
    if region is None:
        return 1.0
    return regime_fraction_inside(region, weights)


def safety_factor(update: ClientUpdate, exponent: float = 1.0) -> float:
    """Downweight an update by how often its shots crossed a limit.

    Graded rather than binary, because a campaign that violated 5% of its
    shots still learned something and one that violated 60% mostly learned
    how to disrupt. `(1 - rate)^exponent` is the simplest rule with the
    right endpoints: a clean campaign is untouched, a campaign that violated
    everything counts for nothing.
    """
    rate = float(np.clip(update.violation_rate, 0.0, 1.0))
    return float((1.0 - rate) ** exponent)


def aggregation_weights(
    updates: list[ClientUpdate],
    target_state: DimensionlessState,
    current_round: int,
    current_config_epoch: int = 0,
    bandwidth: float = 1.0,
    use_similarity: bool = True,
    use_sample_count: bool = False,
    target_region: StateRegion | None = None,
    use_safety: bool = True,
    max_violation_rate: float = 0.5,
    nu_star_max: float = 1.0,
    min_regime_fraction: float = 0.5,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Final normalised weights for one role-matched aggregation.

    Args:
        updates: buffered client updates. MUST all be the same cluster --
            mixing thermal with particle updates is the failure mode role
            matching exists to prevent, so it is checked, not assumed.
        target_state: the operating point of the device being updated.
            Weighting is *relative to a target*, which is what makes this
            personalised rather than a single global model.
        use_similarity: False reproduces baseline 2 (uniform, role-matched).
            True is the method, baseline 3.
        use_sample_count: True adds the FedAvg n_k factor on top. Off by
            default -- SPEC.md §4b is explicit that sample count is the
            arbitrary rule being replaced.

    Returns:
        Weights summing to 1.0, aligned with ``updates``.
    """
    if not updates:
        return np.zeros(0, dtype=float)

    clusters = {u.cluster for u in updates}
    if len(clusters) > 1:
        raise ValueError(
            f"role-matched aggregation received mixed clusters {sorted(clusters)}; "
            "thermal must never aggregate with particle or current (SPEC.md §2)"
        )

    # ADMISSIBILITY IS NOT A WEIGHT. Two different questions get asked in
    # order: may this update be aggregated at all, and if so how much
    # should it count. Collapsing them into one continuous weight means a
    # peer whose physics disqualifies it still contributes, just less.
    admissible = np.ones(len(updates), dtype=bool)
    if use_safety:
        for i, u in enumerate(updates):
            if u.region is not None:
                # A measured envelope answers the regime question properly;
                # a point can only report where the centre happened to fall.
                inside = regime_fraction_inside(u.region)
                if inside < min_regime_fraction:
                    admissible[i] = False
            elif not regime_valid(u.state, nu_star_max):
                admissible[i] = False
            if admissible[i] and u.violation_rate > max_violation_rate:
                admissible[i] = False

    if not admissible.any():
        # Nothing here may be used. Zeros, deliberately -- the caller must
        # be able to tell 'no admissible peer' from 'peers that all count
        # equally', and a uniform fallback here would average in exactly
        # the updates the physics just disqualified.
        return np.zeros(len(updates), dtype=float)

    w = admissible.astype(float)
    for i, u in enumerate(updates):
        if not admissible[i]:
            continue
        if use_similarity:
            if target_region is not None and u.region is not None:
                # Both ends measured as regions: weight by how much the two
                # operating envelopes actually overlap.
                w[i] *= overlap_weight(u.region, target_region, bandwidth)
            else:
                d = similarity_distance(u.state, target_state)
                w[i] *= similarity_weight(d, bandwidth)
        w[i] *= staleness_factor(u, current_round, current_config_epoch)
        if use_safety:
            w[i] *= safety_factor(u)
            w[i] *= regime_factor(u.region)
        if use_sample_count:
            w[i] *= u.n_samples

    total = w.sum()
    if total <= 0:
        # Admissible peers exist but every kernel underflowed: too stale
        # or too far, not disqualified. Uniform over the ADMISSIBLE ones
        # is the honest fallback; silently returning zeros would stall
        # training without saying why.
        #
        # SAY SO, THOUGH. When this fires with use_similarity=True, the
        # similarity arm has just become the uniform arm, and nothing else
        # in the output would distinguish them. That is the same failure as
        # the NaN region that quietly disabled federation: a degradation
        # with a plausible-looking result.
        if diagnostics is not None:
            diagnostics["uniform_fallback"] = True
        return admissible.astype(float) / admissible.sum()
    if diagnostics is not None:
        diagnostics["uniform_fallback"] = False
    return w / total


# ---------------------------------------------------------------------------
# Calibration and diagnostics
#
# The bandwidth must be checked against the real device set before Phase 5,
# because both of its failure modes are silent.
# ---------------------------------------------------------------------------


def suggest_bandwidth(states: list[DimensionlessState]) -> float:
    """Median heuristic for the Gaussian kernel bandwidth.

    Setting the bandwidth to the median pairwise distance is the standard
    kernel-methods choice (it puts the typical pair at exp(-1/2) = 0.61, so
    the kernel is neither saturated nor collapsed). It gives a defensible,
    data-driven default instead of a hand-tuned constant.
    """
    if len(states) < 2:
        return 1.0
    d = [
        similarity_distance(states[i], states[j])
        for i in range(len(states))
        for j in range(i + 1, len(states))
    ]
    med = float(np.median(d))
    return med if med > 1e-9 else 1.0


def describe_device_set(
    named_states: dict[str, DimensionlessState], bandwidth: float | None = None
) -> str:
    """Render the pairwise distance and weight matrices as text.

    Print this before Phase 5 and paste it into FINDINGS.md. Phase 6 is
    impossible if the devices do not overlap in dimensionless space, and this
    is how you find that out in a second rather than after a training run.
    """
    names = list(named_states)
    states = [named_states[n] for n in names]
    bw = suggest_bandwidth(states) if bandwidth is None else bandwidth

    lines = [f"similarity bandwidth = {bw:.3f}", ""]
    lines.append("dimensionless operating points:")
    lines.append(
        f"  {'device':14s}{'rho*':>10s}{'nu*':>10s}{'beta_N':>9s}{'q95':>7s}"
    )
    for n, s in zip(names, states):
        lines.append(
            f"  {n:14s}{s.rho_star:10.2e}{s.nu_star:10.3f}{s.beta_N:9.2f}{s.q95:7.2f}"
        )

    lines += ["", "pairwise distance:", "  " + " " * 14 + "".join(f"{n:>12s}" for n in names)]
    for i, ni in enumerate(names):
        row = "".join(f"{similarity_distance(states[i], states[j]):12.3f}" for j in range(len(names)))
        lines.append(f"  {ni:14s}{row}")

    lines += ["", "aggregation weight (row contributes to column):",
              "  " + " " * 14 + "".join(f"{n:>12s}" for n in names)]
    off = []
    for i, ni in enumerate(names):
        vals = [similarity_weight(similarity_distance(states[i], states[j]), bw) for j in range(len(names))]
        off += [v for j, v in enumerate(vals) if j != i]
        lines.append(f"  {ni:14s}" + "".join(f"{v:12.3f}" for v in vals))

    if not off:
        lines += ["", "only one device -- no peers, so nothing to weight. "
                      "Federation needs at least two."]
        return "\n".join(lines)

    lo, hi = min(off), max(off)
    lines += ["", f"off-diagonal weights: min={lo:.3f} max={hi:.3f}"]
    if hi < 0.05:
        lines.append(
            "  WARNING: every peer is effectively ignored. Federation will do "
            "nothing and baseline 3 will collapse onto baseline 1. Increase "
            "bandwidth or choose devices with more overlap."
        )
    elif lo > 0.9:
        lines.append(
            "  WARNING: all peers weighted near-equally. Baseline 3 will "
            "collapse onto baseline 2 and the method's contribution will be "
            "untestable. Decrease bandwidth."
        )
    else:
        lines.append("  OK: weights discriminate between peers.")
    return "\n".join(lines)
