"""Which reach-rate differences six seeds can actually separate.

PROTOCOL.md 4 makes the reach rate at a fixed budget the primary endpoint, and
at six seeds per arm it is nearly blind: two-sided Fisher exact separates only
0/6 from >=5/6, and 1/6 from 6/6. If the baseline reaches competence in two or
more seeds, nothing in that fold can be distinguished from it whatever it
does.

That is why `sparc_like` is the fold that matters -- it is the one device
where the transferred conventional controller scores 0%, so it is the one
place a baseline near 0/6 is plausible.

The arithmetic is pinned here so the pre-registered decision rule in
FINDINGS.md cannot drift when the numbers arrive.
"""
from math import comb

import pytest


def fisher_exact_2x2(a: int, b: int, n: int = 6) -> float:
    """Two-sided p for reach a/n against b/n, hypergeometric on the margins."""
    total, k = 2 * n, a + b

    def p(x: int) -> float:
        return (comb(n, x) * comb(n, k - x) / comb(total, k)
                if 0 <= k - x <= n else 0.0)

    obs = p(a)
    return sum(p(x) for x in range(n + 1) if p(x) <= obs + 1e-12)


def smallest_separable(baseline: int, n: int = 6) -> int | None:
    for m in range(baseline + 1, n + 1):
        if fisher_exact_2x2(baseline, m, n) < 0.05:
            return m
    return None


@pytest.mark.parametrize("baseline,expected", [
    (0, 5),      # a total failure separates from a near-total success
    (1, 6),      # and only from a total one
    (2, None),   # from here up, nothing is separable at six seeds
    (3, None),
    (4, None),
    (5, None),
])
def test_the_separable_gaps_at_six_seeds(baseline, expected):
    assert smallest_separable(baseline) == expected


def test_the_headline_case_is_significant():
    """sparc_like: the transferred conventional controller scores 0%.

    If a learned arm reaches five or six seeds out of six there, that is the
    one pre-registered comparison this seed count can actually make.
    """
    assert fisher_exact_2x2(0, 5) < 0.05
    assert fisher_exact_2x2(0, 6) < 0.01


def test_the_effect_that_looked_like_a_signal_is_not_one():
    """iter_like's 4/6 against 6/6 was the clearest-looking result in the
    project and it cannot be distinguished from chance at this seed count."""
    assert fisher_exact_2x2(4, 6) > 0.2


def test_doubling_the_seeds_helps_but_only_for_enormous_effects():
    """Twelve seeds do separate a half-succeeding baseline -- from a PERFECT
    method and nothing less.

    I asserted the opposite first and the test caught it, which is the point
    of writing the arithmetic down rather than reasoning about it: at n=12 a
    6/12 baseline needs 12/12, and even 3/12 needs 9/12. Neither is an effect
    size this project has ever seen.
    """
    assert smallest_separable(6, n=12) == 12
    assert fisher_exact_2x2(3, 9, n=12) < 0.05
    assert fisher_exact_2x2(6, 11, n=12) > 0.05


def test_the_test_is_symmetric_and_bounded():
    assert fisher_exact_2x2(3, 3) == pytest.approx(1.0)
    for a in range(7):
        for b in range(7):
            p = fisher_exact_2x2(a, b)
            assert 0.0 < p <= 1.0 + 1e-12
            assert p == pytest.approx(fisher_exact_2x2(b, a))
