"""Operating limits -- the basis of the Phase 6 catastrophe claim."""

import pytest

from hfmarl.envs.limits import DEFAULT_LIMITS, Limit, LimitSet

SAFE = {"greenwald_fraction": 0.5, "beta_N": 1.8, "q95": 3.5}
NEAR = {"greenwald_fraction": 0.9, "beta_N": 2.7, "q95": 2.3}
OVER = {"greenwald_fraction": 1.05, "beta_N": 3.2, "q95": 1.8}


def test_safe_state_has_no_violations():
    r = LimitSet().evaluate(SAFE)
    assert not r.any_violated and r.worst_margin > 1.0


def test_near_limit_is_not_yet_a_violation_but_has_small_margin():
    """The soft band is what gives the policy a gradient before the cliff."""
    r = LimitSet().evaluate(NEAR)
    assert not r.any_violated
    assert 0.0 < r.worst_margin < 1.0


def test_over_limit_flags_every_breach():
    r = LimitSet().evaluate(OVER)
    assert set(r.violations) == {"greenwald_fraction", "beta_N", "q95"}
    assert r.worst_margin < 0.0


def test_margin_is_one_at_soft_and_zero_at_hard():
    lim = Limit("x", soft=0.8, hard=1.0, upper=True)
    assert lim.margin(0.8) == pytest.approx(1.0)
    assert lim.margin(1.0) == pytest.approx(0.0)


def test_lower_limit_direction_is_handled():
    """q95 is a LOWER limit -- falling below is the violation."""
    q = next(l for l in DEFAULT_LIMITS if l.name == "q95")
    assert not q.upper
    assert q.violated(1.9)
    assert not q.violated(2.5)
    assert q.margin(2.5) == pytest.approx(1.0)


def test_missing_quantity_is_skipped_not_defaulted_safe():
    """Defaulting an absent safety quantity is how a controller looks safe wrongly."""
    r = LimitSet().evaluate({"beta_N": 1.0})
    assert "greenwald_fraction" not in r.margins
    assert set(r.values) == {"beta_N"}


def test_empty_observation_gives_neutral_margin():
    r = LimitSet().evaluate({})
    assert r.worst_margin == 1.0 and not r.any_violated


def test_restricted_envelope_is_strictly_tighter():
    """Phase 6 needs device A to never reach the real limit."""
    base, tight = LimitSet(), LimitSet().restricted(0.5)
    for b, t in zip(base.limits, tight.limits):
        assert t.hard != b.hard
        if b.upper:
            assert t.hard < b.hard
        else:
            assert t.hard > b.hard


def test_restricted_envelope_violates_earlier():
    base, tight = LimitSet(), LimitSet().restricted(0.3)
    assert not base.evaluate(NEAR).any_violated
    assert tight.evaluate(NEAR).any_violated


def test_restricted_rejects_bad_fraction():
    with pytest.raises(ValueError):
        LimitSet().restricted(0.0)
    with pytest.raises(ValueError):
        LimitSet().restricted(1.5)


def test_inconsistent_limit_definition_raises():
    with pytest.raises(ValueError):
        Limit("bad", soft=1.0, hard=0.5, upper=True).margin(0.7)


def test_report_str_mentions_violations():
    assert "VIOLATED" in str(LimitSet().evaluate(OVER))
    assert "VIOLATED" not in str(LimitSet().evaluate(SAFE))
