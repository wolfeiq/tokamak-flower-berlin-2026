"""Which devices can actually cross a limit, and what depends on that.

Measured by driving each device from zero to full thermal command on `easy`
and taking the worst margin over the episode:

    iter_like   +4.17     sparc_like  +5.04
    diiid_like  -0.38     tcv_like    -7.48

So containment is a live constraint on two of four devices and vacuous on the
other two. Two things follow, and both are tested here: the catastrophe
experiment must refuse a target with nothing to avoid, and nobody should quote
the joint endpoint's safety half on ITER or SPARC without saying that.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hfmarl.devices.registry import (  # noqa: E402
    BETA_N_MARGIN_AT_FULL_COMMAND,
    DEVICES,
    can_violate,
)


def test_every_device_has_a_measured_reachability():
    """A device with no entry is a device nobody drove to full command."""
    assert set(BETA_N_MARGIN_AT_FULL_COMMAND) == set(DEVICES)


def test_an_unmeasured_device_raises_rather_than_defaulting():
    """Defaulting to 'safe' here would silently pass the interlock below.

    The whole point is to catch an experiment that cannot test its claim; a
    missing measurement must stop it, not wave it through.
    """
    with pytest.raises(KeyError, match="no measured limit reachability"):
        can_violate("machine_that_does_not_exist")


@pytest.mark.parametrize("name,expected", [
    ("iter_like", False),
    ("sparc_like", False),
    ("diiid_like", True),
    ("tcv_like", True),
])
def test_reachability_matches_the_measurement(name, expected):
    assert can_violate(name) is expected
    assert (BETA_N_MARGIN_AT_FULL_COMMAND[name] < 0) is expected


def test_the_catastrophe_experiment_refuses_a_target_with_no_catastrophe():
    """`--target iter_like` was the DEFAULT, and would have scored perfectly.

    On a device that cannot reach a limit, "avoided a regime it had never
    entered" is satisfied by a controller that does nothing.
    """
    import exp_catastrophe

    argv = sys.argv
    try:
        sys.argv = ["exp_catastrophe.py", "--target", "iter_like"]
        assert exp_catastrophe.main() == 2
    finally:
        sys.argv = argv


def test_the_catastrophe_default_target_can_have_one():
    """The default is what gets run when nobody thinks about it."""
    import argparse
    import exp_catastrophe

    src = Path(exp_catastrophe.__file__).read_text(encoding="utf-8")
    ap = argparse.ArgumentParser()
    # Parse the default out of the source rather than running main(), which
    # would fire 150 shots per device.
    line = next(ln for ln in src.splitlines() if '"--target"' in ln)
    default = line.split('default="')[1].split('"')[0]
    assert can_violate(default), (
        f"exp_catastrophe defaults to {default!r}, which cannot cross a limit")
    del ap
