"""The forward solve must satisfy the equation it claims to solve.

Both bugs found while writing this module were silent: a convergence tolerance
with an absolute floor larger than T itself (T is in joules, order 1e-16) exited
the iteration immediately and froze chi at its floor, and a damped fixed point
stalled on the stiff branch. Neither raised; both produced plausible-looking
profiles. The residual test is what catches that class of failure.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from thermal_investigation import profiles as P
from thermal_investigation._closure import chi_from_power_balance

DEVICES = json.loads(
    (Path(P.__file__).with_name("_devices.json")).read_text()
)
WINDOW = slice(10, 90)


def solved(site, fraction=1.0):
    return DEVICES[site], P.simulate(DEVICES[site], fraction)


@pytest.mark.parametrize("site,fraction", [("A", 1.0), ("B", 1.0), ("C", 0.001)])
def test_solution_conserves_flux(site, fraction):
    """n*chi*dT/dr must equal the flux conservation alone demands."""
    _, c = solved(site, fraction)
    r, T, n, q, chi = c["r"], c["T_e"], c["n_e"], c["q"], c["chi_true"]
    step = r[1:] - r[:-1]
    grad = (T[:-1] - T[1:]) / step
    q_mid = 0.5 * (q[:-1] + q[1:])
    n_mid = 0.5 * (n[:-1] + n[1:])
    residual = np.abs(n_mid * chi[:-1] * grad - q_mid) / np.maximum(np.abs(q_mid), 1e-30)
    assert np.median(residual[3:]) < 1e-9


@pytest.mark.parametrize("site", ["A", "B"])
def test_power_balance_recovers_the_transport_that_was_solved_for(site):
    """Given the true source, the estimator must return the true chi."""
    _, c = solved(site)
    got = chi_from_power_balance(c["r"], c["n_e"], c["T_e"], c["source"])
    ratio = float(np.nanmedian(got.chi)) / float(np.nanmedian(c["chi_true"][WINDOW]))
    assert 0.8 < ratio < 1.25


@pytest.mark.parametrize("site", ["A", "B"])
def test_overstated_source_inflates_apparent_transport(site):
    """The demo's entire ambiguity, as a number rather than an assertion."""
    _, c = solved(site)
    inflated = chi_from_power_balance(
        c["r"], c["n_e"], c["T_e"], c["source"] * P.__dict__.get("_FACTOR", 1.6)
    )
    truth = chi_from_power_balance(c["r"], c["n_e"], c["T_e"], c["source"])
    assert np.nanmedian(inflated.chi) > 1.4 * np.nanmedian(truth.chi)


def test_devices_differ_in_transport_not_just_in_labels():
    """A high-field compact machine must not transport like a low-field one."""
    a_dev, a = solved("A")
    b_dev, b = solved("B")
    assert b_dev["B_0"] > a_dev["B_0"]
    # Gyro-Bohm: chi scales as 1/B^2, so the high-field device must be stiffer.
    assert b["chi_gb"] < a["chi_gb"] / 2.0


def test_weakly_heated_profile_stays_below_the_critical_gradient():
    """C is unidentifiable because it is flat, not because it is labelled flat."""
    dev, c = solved("C", 0.001)
    r_over_lt = dev["R_major"] * np.abs(np.gradient(c["T_e"], c["r"])) / np.maximum(c["T_e"], 1e-30)
    assert np.median(r_over_lt[WINDOW]) < P.CRITICAL_R_OVER_LT


def test_stiffness_actually_reaches_the_solution():
    """Guards the frozen-chi bug: raising stiffness must change the profile.

    Compared in keV with an explicit relative tolerance. pytest.approx carries a
    default ABSOLUTE tolerance of 1e-12, which silently swallows any comparison
    of joule-scale temperatures -- the same mistake that froze chi in the solver.
    """
    before = P.STIFFNESS
    try:
        P.STIFFNESS = 1.5
        soft = P.simulate(DEVICES["A"], 1.0)["T_e_keV"][0]
        P.STIFFNESS = 60.0
        stiff = P.simulate(DEVICES["A"], 1.0)["T_e_keV"][0]
    finally:
        P.STIFFNESS = before
    # Stiffer transport pins the profile closer to the critical gradient.
    assert stiff < soft
    assert abs(stiff - soft) / soft > 0.05


def test_ground_truth_never_reaches_a_released_product():
    from thermal_investigation.core import analyse_site

    for site in ("A", "B", "C"):
        blob = json.dumps(analyse_site(site))
        assert "chi_true" not in blob and "chi_gb" not in blob
