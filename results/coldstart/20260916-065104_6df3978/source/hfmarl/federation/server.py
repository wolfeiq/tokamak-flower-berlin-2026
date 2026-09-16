"""The buffer and the aggregation step -- the part that was never looped.

`similarity.py` answers "how much should this client count?". Nothing consumed
the answer: no buffer held updates, no aggregate was ever formed, and no
aggregated weight ever reached a policy. This module is that missing half, and
it is deliberately small -- the arithmetic all lives in `similarity.py`, which
is already tested.

MERGING IS NOT A WEIGHTED MEAN
------------------------------
Averaging the parameters of independently trained networks is usually a bad
idea, for two reasons that need opposite fixes, and `federation/robust.py`
holds both:

  * hidden units that have drifted out of correspondence are averaged
    coordinate-against-wrong-coordinate -- fixed by permuting each peer into
    the target's basis first, which is free because it does not change any
    peer's function;
  * one bad client moves a mean arbitrarily far, because a mean has a
    breakdown point of zero -- fixed by the weighted geometric median, and
    by clipping each deviation to a radius around the target's current model.

Both defaults are on. `rule='mean'` and `align=False` remain available as
ablations, because "the robust rule helped" is a claim that needs the
unprotected number next to it.

Two preconditions still hold and are not tuning knobs: every client starts
from ONE initialisation, and local drift between exchanges is bounded. A run
with clients started from different inits is not a harder version of this
experiment; it is a different one, and it will mostly measure permutation
mismatch that no aggregation rule can undo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hfmarl.federation.robust import (
    align_to,
    centered_clip,
    coordinate_median,
    weighted_geometric_median,
)
from hfmarl.federation.similarity import ClientUpdate, aggregation_weights
from hfmarl.physics.dimensionless import DimensionlessState


@dataclass
class FedBuffServer:
    """Asynchronous buffer, one slot per client per channel.

    A new update from a device REPLACES that device's slot rather than queueing
    behind it: a client that has been running all week should not get a louder
    vote than one that fired twice, and FedAvg's sample-count weighting is the
    arbitrary rule SPEC.md §4b exists to replace. Age is expressed through the
    staleness factor instead, which is what makes the buffer asynchronous
    rather than merely unordered.
    """

    bandwidth: float = 1.0
    use_similarity: bool = True
    use_sample_count: bool = False
    use_safety: bool = True
    include_self: bool = True
    # A weighted mean has a breakdown point of zero: one arbitrarily bad
    # client moves it arbitrarily far. The default is the robust rule; see
    # federation/robust.py for why it is the geometric median and not Krum
    # or a trimmed mean at this client count.
    rule: str = "geomedian"  # geomedian | mean | median
    align: bool = True       # permute hidden units before merging
    clip_factor: float | None = 2.0
    shape: tuple[int, int, int] | None = None  # obs_dim, hidden, act_dim
    last_clipped: int = 0
    last_rejected_all: bool = False
    last_error: str = ""
    # CUMULATIVE, because the `last_` fields describe one call. An arm whose
    # aggregate was empty for the first twenty rounds and fine on the
    # twenty-first reports last_rejected_all=False, which is true and
    # useless: the arm spent its whole pretraining as a set of isolated
    # clients. These count how often that happened.
    rejected_rounds: int = 0
    error_rounds: int = 0
    # Rounds where every similarity kernel underflowed and the weights fell
    # back to uniform over the admissible peers. Counted only when
    # use_similarity is on, because for the uniform arm it is not a fallback.
    uniform_fallback_rounds: int = 0
    # channel -> device -> most recent update
    _slots: dict[str, dict[str, ClientUpdate]] = field(default_factory=dict)
    last_weights: dict[str, np.ndarray] = field(default_factory=dict)

    def publish(self, update: ClientUpdate) -> None:
        self._slots.setdefault(update.cluster, {})[update.device] = update

    def buffered(self, cluster: str) -> list[ClientUpdate]:
        return list(self._slots.get(cluster, {}).values())

    def aggregate_for(
        self,
        target: str,
        target_state: DimensionlessState,
        cluster: str,
        current_round: int,
        current_config_epoch: int = 0,
        reference: np.ndarray | None = None,
        target_region=None,
    ) -> np.ndarray | None:
        """Personalised aggregate for one device, or None if nothing to mix.

        `reference` is the target's CURRENT parameters. Alignment permutes
        peers into its basis and clipping is centred on it, so passing it
        is what makes both defences work; without it they fall back to the
        highest-weighted peer and to no clipping respectively.

        Personalised, not global: the weighting is computed relative to where
        THIS device operates, so ITER and TCV receive different mixtures of the
        same buffer. A single global model would throw away the entire point of
        weighting by dimensionless distance.
        """
        updates = self.buffered(cluster)
        if not self.include_self:
            updates = [u for u in updates if u.device != target]
        if not updates:
            return None

        diag: dict = {}
        w = aggregation_weights(
            updates,
            diagnostics=diag,
            target_state=target_state,
            current_round=current_round,
            current_config_epoch=current_config_epoch,
            bandwidth=self.bandwidth,
            target_region=target_region,
            use_similarity=self.use_similarity,
            use_sample_count=self.use_sample_count,
            use_safety=self.use_safety,
        )
        # A similarity arm whose kernel underflowed IS the uniform arm, and
        # the returned weights are the only place that ever showed it.
        if diag.get("uniform_fallback") and self.use_similarity:
            self.uniform_fallback_rounds += 1
        self.last_weights[target] = np.asarray(
            [(u.device, wi) for u, wi in zip(updates, w)], dtype=object
        )
        if not np.all(np.isfinite(w)):
            # SELF-AUDIT. Distinct from rejection, and it used to be
            # indistinguishable: NaN weights made `any(w > 0)` False, so
            # a numerical fault was reported as 'no admissible peer' and
            # federation went quiet for the round.
            self.last_rejected_all = False
            self.error_rounds += 1
            self.last_error = (
                "non-finite aggregation weights -- a peer's operating region or "
                "state could not be encoded; federation did not run this round")
            return None
        self.last_error = ""
        if not np.any(w > 0):
            # Physics-informed rejection disqualified every buffered peer:
            # outside the regime where similarity holds, or produced by a
            # campaign that spent most of its shots crossing limits. No
            # aggregate is the correct answer -- averaging them anyway is
            # what the rejection exists to prevent.
            self.last_rejected_all = True
            self.rejected_rounds += 1
            return None
        self.last_rejected_all = False

        # AUDIT #3. Zeroing a weight is not removing a client. The
        # coordinate median ignores weights entirely, and the clipping
        # radius is a median over the rows it is given -- so a rejected
        # peer still moved both. Measured: with weights [1, 0, 0] the
        # coordinate median returned the REJECTED peer's value.
        # Inadmissible rows are dropped here, before anything reads the
        # stack; their zero weights stay in `last_weights` for the record.
        # This also drops a peer whose similarity kernel UNDERFLOWED to
        # exactly zero, which is deliberate: a peer contributing nothing
        # to the weighted mean must not be allowed to move the coordinate
        # median or the clipping radius either. `last_weights` keeps the
        # distinction between 'rejected' and 'too far to matter'.
        keep = w > 0
        updates = [u for u, k in zip(updates, keep) if k]
        w = w[keep]
        total = w.sum()
        w = w / total if total > 0 else np.full(len(updates), 1.0 / len(updates))
        stack = np.stack([u.weights["flat"] for u in updates])

        # 1. Align. Permuting hidden units leaves each client's function
        #    untouched, so this cannot cost anything -- but without it the
        #    mean averages coordinates that do not correspond.
        if self.align and self.shape is not None and stack.shape[0] > 1:
            obs_dim, hidden, act_dim = self.shape
            anchor = (np.asarray(reference, float) if reference is not None
                      else stack[int(np.argmax(w))])
            stack = np.stack([
                align_to(anchor, row, obs_dim, hidden, act_dim)
                for row in stack
            ])

        # 2. Clip, if there is something to centre on. Bounds how far any
        #    one client can move the result, and keeps the weighting.
        self.last_clipped = 0
        if self.clip_factor and reference is not None and stack.shape[0] > 1:
            stack, self.last_clipped = centered_clip(
                stack, np.asarray(reference, float), self.clip_factor)

        # 3. Combine.
        if self.rule == "mean":
            return np.tensordot(w, stack, axes=(0, 0))
        if self.rule == "median":
            return coordinate_median(stack)
        if self.rule == "geomedian":
            return weighted_geometric_median(stack, w)
        raise ValueError(
            f"unknown aggregation rule {self.rule!r}; "
            "expected one of: mean, median, geomedian")

    def weight_table(self) -> dict[str, dict[str, float]]:
        """What each client contributed to each target, for the record."""
        out: dict[str, dict[str, float]] = {}
        for target, rows in self.last_weights.items():
            out[target] = {str(d): float(w) for d, w in rows}
        return out
