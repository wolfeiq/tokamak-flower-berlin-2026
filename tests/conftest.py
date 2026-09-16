"""Shared fixtures.

The `fake_torax` fixture lets us test the actuator ramp logic -- the single
most important correctness property in the repo -- without installing TORAX.
It stubs only `torax.experimental.TimeVaryingScalarUpdate`, which is the one
TORAX symbol `ActuatorBank.build_updates` touches.
"""

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class FakeUpdate:
    time: np.ndarray
    value: np.ndarray


@pytest.fixture
def fake_torax(monkeypatch):
    """Install a minimal stub of torax.experimental for the duration of a test."""
    torax_mod = types.ModuleType("torax")
    exp_mod = types.ModuleType("torax.experimental")
    exp_mod.TimeVaryingScalarUpdate = FakeUpdate
    torax_mod.experimental = exp_mod
    monkeypatch.setitem(sys.modules, "torax", torax_mod)
    monkeypatch.setitem(sys.modules, "torax.experimental", exp_mod)
    return exp_mod
