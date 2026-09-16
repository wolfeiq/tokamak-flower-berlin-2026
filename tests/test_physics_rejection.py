"""Refusing an update the physics says not to trust.

The generic robustness already in place -- alignment, geometric median, centred
clipping -- bounds how far a bad peer can drag the merge. None of it can refuse
a peer, and none of it knows anything about plasmas. These do both.

The case that motivated them is measured, not hypothetical. In the catastrophe
run, federating tcv_like with peers that never meet a limit raised TCV's own
violation rate from 5.3% to 24.4%. iter_like and sparc_like cannot reach their
beta_N limits at all, so their policies drive hard with impunity, and averaging
that confidence in is how it travels to a machine that violates on step 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmarl.devices.registry import encoded_states
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import (
    ClientUpdate,
    aggregation_weights,
    regime_valid,
    safety_factor,
)
from hfmarl.physics.dimensionless import DimensionlessState

STATES = encoded_states()
TARGET = STATES["tcv_like"]


def update(device="diiid_like", violation_rate=0.0, state=None, rnd=1):
    return ClientUpdate(
        device=device, cluster="thermal",
        weights={"flat": np.zeros(4)},
        state=state if state is not None else STATES[device],
        n_samples=10, round_produced=rnd, violation_rate=violation_rate,
    )


# -- the regime gate ------------------------------------------------------


def test_every_device_in_the_set_is_currently_admissible():
    """The gate is armed, not active: all four sit deep in the banana regime.
    If this ever fails, a device drifted and weight sharing stopped being
    justified -- which is the point of checking."""
    for name, s in STATES.items():
        assert regime_valid(s), f"{name} at nu*={s.nu_star}"


def test_a_peer_outside_the_banana_regime_is_refused_not_downweighted():
    """nu* > 1 is Pfirsch-Schlueter. The transport physics differs
    qualitatively, so Connor-Taylor gives no reason to share weights at all.
    A Gaussian kernel would say 'far, count less'; this says 'no'."""
    collisional = DimensionlessState(rho_star=6e-3, nu_star=2.5, beta_N=2.0,
                                     q95=3.5, mach=0.0)
    ups = [update("diiid_like"), update("iter_like", state=collisional)]
    w = aggregation_weights(ups, TARGET, current_round=1)
    assert w[1] == 0.0
    assert w[0] == pytest.approx(1.0)


def test_when_no_peer_is_admissible_the_weights_are_zero_not_uniform():
    """The caller must be able to tell 'nobody may be used' from 'everybody
    counts equally'. A uniform fallback here would average in precisely the
    updates the physics just disqualified."""
    collisional = DimensionlessState(rho_star=6e-3, nu_star=2.5, beta_N=2.0,
                                     q95=3.5, mach=0.0)
    ups = [update("diiid_like", state=collisional),
           update("iter_like", state=collisional)]
    w = aggregation_weights(ups, TARGET, current_round=1)
    assert np.allclose(w, 0.0)


def test_the_server_returns_no_aggregate_rather_than_a_zero_policy():
    """A zero vector IS a policy, and a client would adopt it and unlearn."""
    collisional = DimensionlessState(rho_star=6e-3, nu_star=2.5, beta_N=2.0,
                                     q95=3.5, mach=0.0)
    s = FedBuffServer(bandwidth=1.0, align=False, clip_factor=None)
    for d in ("diiid_like", "iter_like"):
        s.publish(update(d, state=collisional))
    assert s.aggregate_for("tcv_like", TARGET, "thermal", 1) is None
    assert s.last_rejected_all is True


# -- the violation gate ---------------------------------------------------


def test_a_disruptive_campaign_counts_for_less_than_a_clean_one():
    ups = [update("diiid_like", violation_rate=0.0),
           update("sparc_like", violation_rate=0.4)]
    w = aggregation_weights(ups, TARGET, current_round=1, use_similarity=False)
    assert w[0] > w[1]


def test_a_mostly_disruptive_campaign_is_refused_outright():
    """Past the threshold it is not 'weaker evidence', it is a machine that
    spent its campaign learning how to cross a limit."""
    ups = [update("diiid_like", violation_rate=0.0),
           update("sparc_like", violation_rate=0.8)]
    w = aggregation_weights(ups, TARGET, current_round=1)
    assert w[1] == 0.0
    assert w[0] == pytest.approx(1.0)


def test_safety_factor_has_the_right_endpoints():
    assert safety_factor(update(violation_rate=0.0)) == pytest.approx(1.0)
    assert safety_factor(update(violation_rate=1.0)) == pytest.approx(0.0)
    assert safety_factor(update(violation_rate=0.25)) == pytest.approx(0.75)


def test_rejection_can_be_switched_off_for_the_ablation():
    """Every defence in this repo has to be removable, or 'it helped' is not a
    claim anyone can check."""
    ups = [update("diiid_like", violation_rate=0.0),
           update("sparc_like", violation_rate=0.9)]
    w = aggregation_weights(ups, TARGET, current_round=1, use_safety=False)
    assert w[1] > 0.0


def test_weights_still_sum_to_one_with_rejection_active():
    ups = [update("diiid_like", violation_rate=0.1),
           update("iter_like", violation_rate=0.3),
           update("sparc_like", violation_rate=0.9)]
    w = aggregation_weights(ups, TARGET, current_round=1)
    assert np.isclose(w.sum(), 1.0)
    assert w[2] == 0.0
