"""The conventional arm's gain, chosen where the protocol says it must be.

PROTOCOL.md 2 refuses hyperparameter tuning on the held-out device for every
arm. The conventional arm has been exempt from that in practice: every number
reported for it came from `--kp-sweep` and quoted the best gain, which is
tuning on the evaluation device.

`brink` made the difference visible. DIII-D's best gain is 0.9; on TCV that
scores 90% where TCV's own best (0.6) scores 100%. So the classical baseline
has the cold-start problem federation claims to solve, and the transfer column
is what says so.

The selection is pure arithmetic over a sweep that has already happened, so it
is tested here without firing anything.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "exp_conventional.py").read_text(encoding="utf-8")


def choose_from_sources(per_gain, name):
    """The selection rule, mirrored from exp_conventional.main.

    Kept in step with the script by `test_the_script_still_uses_this_rule`.
    """
    others = [d for d in per_gain if d != name]
    scored = []
    for g in sorted(per_gain[name], key=lambda g: (g is None, g)):
        js = [per_gain[d][g][0] for d in others if g in per_gain[d]]
        es = [per_gain[d][g][1] for d in others if g in per_gain[d]]
        if js:
            scored.append((float(np.mean(js)), -float(np.mean(es)), g))
    return max(scored)[2] if scored else None


def test_the_brink_numbers_reproduce_the_transfer_penalty():
    """The measured case, as a regression on the rule itself.

    Both devices reach 100% at several gains, so joint success ties and the
    error breaks it -- which on DIII-D picks 0.9, and 0.9 is the gain that
    costs TCV ten points.
    """
    per_gain = {
        "diiid_like": {0.05: (1.0, 0.1588), 0.6: (1.0, 0.0491),
                       0.9: (1.0, 0.0362)},
        "tcv_like": {0.05: (1.0, 0.0981), 0.6: (1.0, 0.0290),
                     0.9: (0.90, 0.0811)},
    }
    assert choose_from_sources(per_gain, "tcv_like") == 0.9
    assert per_gain["tcv_like"][0.9][0] == 0.90   # what transfer costs
    assert per_gain["tcv_like"][0.6][0] == 1.00   # what tuning here would buy


def test_a_gain_that_fails_on_the_sources_is_not_chosen():
    """Joint success decides; the error is only a tie-break."""
    per_gain = {
        "a": {0.1: (1.0, 0.09), 0.9: (0.5, 0.01)},
        "b": {0.1: (1.0, 0.08), 0.9: (0.5, 0.01)},
        "c": {0.1: (0.7, 0.20), 0.9: (0.7, 0.20)},
    }
    assert choose_from_sources(per_gain, "c") == 0.1


def test_a_gain_missing_from_a_source_does_not_crash_the_average():
    per_gain = {
        "a": {0.1: (1.0, 0.09)},
        "b": {0.1: (1.0, 0.08), 0.9: (1.0, 0.01)},
        "c": {0.1: (0.7, 0.20), 0.9: (0.9, 0.05)},
    }
    # 0.9 is only measured on b, and that is enough to rank it.
    assert choose_from_sources(per_gain, "c") == 0.9


def test_the_script_still_uses_this_rule():
    """Guards the mirrored arithmetic above against drifting from the script."""
    assert "sources pick kp" in SRC
    assert "transfer_rows" in SRC
    assert '"transfer": transfer_rows' in SRC
    # joint success first, mean error as the tie-break, ties to the lower error
    assert "scored.append((float(np.mean(js)), -float(np.mean(es)), g))" in SRC
    assert "_, _, g_src = max(scored)" in SRC


def test_the_transfer_column_costs_no_extra_shots():
    """It reads the sweep that already ran; if it fired shots it would be a
    second experiment wearing the first one's budget."""
    block = SRC[SRC.index("transfer_rows = []"):SRC.index('say("=" * 74)')]
    for forbidden in ("fire_shot", "ToraxDeviceEnv", "RunLog("):
        assert forbidden not in block, forbidden


def test_the_tie_break_is_in_units_of_each_device_tolerance():
    """Absolute beta_N errors are not commensurable across devices.

    DIII-D's tolerance on `moderate` is 0.2401 and SPARC's is 0.0091 -- a
    factor of 26. An unnormalised mean error would let DIII-D's numbers break
    every tie for every device, which is the mistake the band-fraction
    normalisation exists to prevent, made one level up.
    """
    assert "med / tol if tol else" in SRC
    # and the header says so, because a ranking rule nobody can see is a
    # ranking rule nobody can check
    assert "in units of each device's own tolerance" in SRC


def test_normalisation_changes_which_gain_the_sources_pick():
    """A worked case, so the fix is not just a comment.

    Two sources tie on joint success. Unnormalised, the one with the larger
    absolute errors decides; normalised, each source speaks in its own units.
    """
    # Absolute errors: device A has a loose tolerance and big numbers,
    # device B a tight one and small numbers. A prefers gain 0.9, B prefers
    # 0.1, and B's preference is the stronger one in its own units.
    tol = {"a": 0.24, "b": 0.01}
    absolute = {
        "a": {0.1: (1.0, 0.120), 0.9: (1.0, 0.060)},   # A: 0.9 halves its error
        "b": {0.1: (1.0, 0.004), 0.9: (1.0, 0.009)},   # B: 0.9 more than doubles
        "target": {0.1: (1.0, 0.5), 0.9: (0.0, 0.9)},
    }
    normalised = {d: {g: (j, e / tol[d]) for g, (j, e) in rows.items()}
                  for d, rows in absolute.items() if d in tol}
    normalised["target"] = absolute["target"]

    # Unnormalised: mean error at 0.9 is (0.060+0.009)/2 = 0.0345 against
    # (0.120+0.004)/2 = 0.062 -> picks 0.9, which fails the target outright.
    assert choose_from_sources(absolute, "target") == 0.9
    # Normalised: 0.9 costs B 0.9 tolerances against A's saving of 0.25,
    # so the sources pick 0.1 -- and the target passes.
    assert choose_from_sources(normalised, "target") == 0.1
