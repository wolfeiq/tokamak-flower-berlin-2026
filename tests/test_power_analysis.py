"""The power calculation, which decides what a null result means.

Run before reading a fold, not after. If six seeds cannot resolve the
differences the arms actually show, then "all five arms are indistinguishable"
is a statement about the experiment rather than about federation -- and the
measured answer is that they cannot: median seed-to-seed sd 11.5 shots gives a
detection floor of 18.7 shots at n=6, against observed differences of 3 to 35.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "analyse_power.py").read_text(encoding="utf-8")


def mdd(sd, n, paired=False):
    """Minimum detectable difference, as the script computes it."""
    return 2.80 * sd * (np.sqrt(1.0 / n) if paired else np.sqrt(2.0 / n))


def test_the_script_is_runnable_from_anywhere():
    """It carried an absolute path to my home directory when it was a
    scratchpad probe."""
    assert ("C:" + chr(92) + "Users") not in SRC
    assert "Path(__file__).resolve().parents[1]" in SRC


def test_the_detection_floor_falls_as_the_root_of_n():
    """Halving the floor costs four times the seeds, which is the whole
    reason more seeds is not the answer here."""
    assert mdd(11.5, 24) == pytest.approx(mdd(11.5, 6) / 2, rel=1e-9)


def test_the_measured_spread_puts_the_observed_effects_below_the_floor():
    """The finding, as a regression on the arithmetic.

    Median sd 11.5 shots over fifteen (fold, arm) cells; the differences the
    folds show are 3 to 35 shots.
    """
    floor = mdd(11.5, 6)
    assert 18 < floor < 19
    for observed in (3.3, 5.0, 10.0, 15.0):
        assert observed < floor          # cannot be resolved
    assert 35.0 > floor                  # the one that could be, in principle


def test_pairing_only_helps_when_the_differences_are_tighter():
    """Measured, they are not: sd(diff) came out LARGER than sd(raw) on most
    comparisons, so common random numbers buys nothing here."""
    sd_raw, sd_diff = 18.3, 40.4        # single_source vs scratch, tcv fold
    assert mdd(sd_diff, 6, paired=True) > mdd(sd_raw, 6)


def test_a_reach_rate_difference_of_four_in_six_is_not_resolvable():
    """PROTOCOL.md 4 asks for reach rates, and at six seeds they are the
    weakest of the three metrics -- only an all-or-nothing effect shows."""
    from math import comb

    def fisher_two_sided(a, b, n=6):
        """P(|difference| >= observed) under the null, by enumeration."""
        obs = abs(a - b)
        tot = 0.0
        for x in range(n + 1):
            for y in range(n + 1):
                p = comb(n, x) * comb(n, y) / (2.0 ** (2 * n))
                if abs(x - y) >= obs:
                    tot += p
        return tot

    assert fisher_two_sided(4, 6) > 0.2      # the iter_like effect: invisible
    assert fisher_two_sided(0, 6) < 0.01     # all-or-nothing: visible
