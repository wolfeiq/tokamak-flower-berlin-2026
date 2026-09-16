"""Small, independently testable boundary for optional TORAX 1.4.3 snapshots."""

import numpy as np


def fraction_to_action(fraction):
    """Convert a fraction of an actuator's range to its [-1, 1] action."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Heating fraction must be in [0, 1]")
    return 2.0 * fraction - 1.0


def electron_source(state):
    """Read the actual dict-of-JAX-arrays contract, including ion/electron exchange.

    This is a net source/sink, not a delivered auxiliary-power measurement.
    A transient snapshot still requires a storage term for a physical balance.
    """
    sources = state.core_sources
    terms = sources.T_e
    if not isinstance(terms, dict) or not terms:
        raise ValueError("TORAX electron source dictionary is empty")
    expected = np.asarray(state.core_profiles.T_e.value).shape
    arrays = [np.asarray(value, dtype=float) for value in terms.values()]
    exchange = np.asarray(sources.qei.qei_coef, dtype=float) * (
        np.asarray(state.core_profiles.T_i.value)
        - np.asarray(state.core_profiles.T_e.value)
    )
    arrays.append(exchange)
    if len(expected) != 1 or any(
        a.shape != expected or not np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("TORAX source profile grids or values are inconsistent")
    return np.sum(arrays, axis=0)
