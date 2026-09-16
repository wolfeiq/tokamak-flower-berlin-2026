"""Identification: closure-free chi(rho), the inverse PINN, and the CRLB screen.

All of this runs without TORAX -- the identification maths is verified against
a manufactured solution before it ever meets simulator output.
"""

import numpy as np
import pytest

from hfmarl.identification.closure import (
    ChiEstimate,
    chi_from_flux,
    chi_from_power_balance,
    heat_flux_from_balance,
    manufactured_case,
    relative_error,
)
from hfmarl.identification.crlb import (
    crlb,
    fisher_from_jacobian,
    jacobian_by_finite_differences,
)
from hfmarl.identification.pinn import ScalarMLP, identify_chi_pinn


def bump(r, base=0.5, height=2.0, centre=0.5, width=0.15):
    return base + height * np.exp(-(((r - centre) / width) ** 2))


# -- the manufactured solution -------------------------------------------


def test_manufactured_case_is_self_consistent():
    """The source must be the one conservation demands for the chosen chi."""
    c = manufactured_case(bump)
    for k in ("rho", "n_e", "T_e", "source", "chi_true"):
        assert k in c and np.all(np.isfinite(c[k]))
    assert c["T_e"][0] > c["T_e"][-1], "temperature should be peaked"


def test_power_balance_recovers_chi_on_clean_data():
    """Generated from the DIFFERENTIAL form, recovered from the INTEGRAL form
    -- different numerical operations, so agreement is a real check."""
    c = manufactured_case(bump, n_rho=201)
    est = chi_from_power_balance(c["rho"], c["n_e"], c["T_e"], c["source"])
    w = (c["rho"] >= 0.05) & (c["rho"] <= 0.95)
    assert relative_error(est.chi[w], c["chi_true"][w]) < 0.02


def test_mid_radius_anomaly_is_located_correctly():
    """The whole point: find WHERE transport is anomalous, not just that it is."""
    c = manufactured_case(lambda r: bump(r, centre=0.7), n_rho=201)
    est = chi_from_power_balance(c["rho"], c["n_e"], c["T_e"], c["source"])
    w = (c["rho"] >= 0.05) & (c["rho"] <= 0.95)
    peak = c["rho"][w][np.nanargmax(est.chi[w])]
    assert abs(peak - 0.7) < 0.05


def test_flux_vanishes_on_the_magnetic_axis():
    """No heat crosses rho = 0; it is a boundary condition, not a fit."""
    c = manufactured_case(bump)
    q = heat_flux_from_balance(c["rho"], c["n_e"], c["T_e"], c["source"])
    assert q[0, 0] == pytest.approx(0.0)


def test_flux_is_outward_for_a_peaked_plasma():
    c = manufactured_case(bump)
    q = heat_flux_from_balance(c["rho"], c["n_e"], c["T_e"], c["source"])[0]
    assert np.nanmean(q[5:-5]) > 0


def test_chi_is_nan_where_the_gradient_is_flat():
    """A flat profile carries no information about transport. Returning a
    number there would invent one."""
    rho = np.linspace(0, 1, 51)
    flat_T = np.full_like(rho, 5.0)
    q = np.ones((1, rho.size))
    chi = chi_from_flux(rho, q, np.ones_like(rho), flat_T)
    assert np.all(np.isnan(chi))


def test_estimate_reports_its_own_masked_fraction():
    c = manufactured_case(bump)
    est = chi_from_power_balance(c["rho"], c["n_e"], c["T_e"], c["source"])
    assert isinstance(est, ChiEstimate)
    assert 0.0 <= est.valid_fraction <= 1.0
    assert isinstance(est.masked_summary(), str)


def test_rejects_non_monotonic_radius():
    with pytest.raises(ValueError):
        heat_flux_from_balance(np.array([0.0, 0.5, 0.2]), np.ones(3), np.ones(3),
                               np.ones(3))


def test_rejects_mismatched_shapes():
    c = manufactured_case(bump, n_rho=51)
    with pytest.raises(ValueError):
        heat_flux_from_balance(c["rho"], np.ones(10), c["T_e"], c["source"])


def test_scales_with_source_magnitude():
    """Double the heating at fixed profiles and chi must double."""
    c = manufactured_case(bump, n_rho=201)
    a = chi_from_power_balance(c["rho"], c["n_e"], c["T_e"], c["source"])
    b = chi_from_power_balance(c["rho"], c["n_e"], c["T_e"], 2 * c["source"])
    w = (c["rho"] >= 0.2) & (c["rho"] <= 0.8)
    assert np.nanmean(b.chi[w] / a.chi[w]) == pytest.approx(2.0, rel=1e-6)


# -- the network ---------------------------------------------------------


def test_mlp_derivative_is_analytic_and_correct():
    """The residual depends on dT/drho; a wrong derivative is a silent
    physics error, not a crash."""
    net = ScalarMLP(8, seed=0)
    x = np.linspace(0, 1, 200)
    y, dy = net.value_and_grad_x(x)
    fd = np.gradient(y, x)
    assert np.max(np.abs(dy[3:-3] - fd[3:-3])) / np.max(np.abs(fd)) < 1e-3


def test_mlp_flat_roundtrip():
    net = ScalarMLP(6, seed=1)
    v = net.get_flat().copy()
    net.set_flat(np.zeros_like(v))
    net.set_flat(v)
    assert np.allclose(net.get_flat(), v)


def test_mlp_is_small():
    """SPEC.md §8: small networks throughout."""
    assert ScalarMLP(16).n_params < 100


def test_pinn_chi_is_positive_by_construction():
    """A negative transport coefficient is unphysical; softplus enforces it so
    the optimiser cannot satisfy the residual in a nonsense way."""
    c = manufactured_case(bump, n_rho=61)
    q = heat_flux_from_balance(c["rho"], c["n_e"], c["T_e"], c["source"])[0]
    r = identify_chi_pinn(c["rho"], c["T_e"], c["n_e"], c["rho"] * q,
                          iterations=30, use_jax=False, seed=0)
    assert np.all(r.chi > 0)


def test_pinn_reduces_its_loss():
    c = manufactured_case(bump, n_rho=41)
    q = heat_flux_from_balance(c["rho"], c["n_e"], c["T_e"], c["source"])[0]
    r = identify_chi_pinn(c["rho"], c["T_e"], c["n_e"], c["rho"] * q,
                          iterations=120, lr=1e-2, use_jax=False, seed=0)
    assert r.history[-1]["loss"] < r.history[0]["loss"]


def test_pinn_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        identify_chi_pinn(np.linspace(0, 1, 10), np.ones(9), np.ones(10),
                          np.ones(10), use_jax=False)


def test_pinn_rejects_too_few_points():
    with pytest.raises(ValueError):
        identify_chi_pinn(np.linspace(0, 1, 3), np.ones(3), np.ones(3),
                          np.ones(3), use_jax=False)


# -- identifiability -----------------------------------------------------


def test_crlb_falls_with_more_data():
    x = np.linspace(0, 1, 50)
    J1 = np.column_stack([x, x**2])
    J2 = np.column_stack([np.tile(x, 4), np.tile(x, 4) ** 2])
    a = crlb(J1, 0.01, ["a", "b"], np.array([1.0, 1.0]))
    b = crlb(J2, 0.01, ["a", "b"], np.array([1.0, 1.0]))
    assert np.all(b.crlb_std < a.crlb_std)


def test_crlb_rises_with_noise():
    x = np.linspace(0, 1, 50)
    J = np.column_stack([x, x**2])
    lo = crlb(J, 0.01, ["a", "b"], np.array([1.0, 1.0]))
    hi = crlb(J, 0.10, ["a", "b"], np.array([1.0, 1.0]))
    assert np.all(hi.crlb_std > lo.crlb_std)


def test_degenerate_pair_is_flagged_even_though_its_crlb_looks_fine():
    """The per-parameter bound LIES here -- with a pseudo-inverse it reports a
    small std for two parameters that are the same parameter. Only the
    correlation and rank checks catch it, which is why both are reported."""
    x = np.linspace(0, 1, 80)
    J = np.column_stack([x, np.exp(-x), np.exp(-x) * 1.0001])
    rep = crlb(J, 0.01, ["chi_mult", "tau_dev", "conf_dev"], np.ones(3))
    a, b, r = rep.worst_pair()
    assert {a, b} == {"tau_dev", "conf_dev"}
    assert r > 0.95
    assert rep.rank_deficient
    assert "WARNING" in rep.summary()


def test_exactly_duplicated_column_is_rank_deficient():
    x = np.linspace(0, 1, 40)
    J = np.column_stack([x, x])
    assert crlb(J, 0.01, ["a", "b"], np.ones(2)).rank_deficient


def test_independent_parameters_are_not_flagged():
    x = np.linspace(0, 1, 80)
    J = np.column_stack([np.ones_like(x), x, x**2])
    rep = crlb(J, 0.01, ["a", "b", "c"], np.ones(3))
    assert not rep.rank_deficient
    assert "NOT identifiable" not in rep.summary()


def test_fisher_matrix_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(40, 3))
    F = fisher_from_jacobian(J, 0.05)
    assert np.allclose(F, F.T)
    assert np.all(np.linalg.eigvalsh(F) > -1e-12)


def test_finite_difference_jacobian_matches_an_analytic_one():
    """Used instead of jax.jacfwd because TORAX #2331 makes that silently NaN."""
    A = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])

    def forward(theta):
        return A @ theta

    J = jacobian_by_finite_differences(forward, np.array([1.0, 1.0]))
    assert np.allclose(J, A, atol=1e-6)


def test_crlb_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        crlb(np.ones((10, 2)), 0.01, ["only_one"], np.array([1.0]))


# ---------------------------------------------------------------------------
# Regressions from the audit.
# ---------------------------------------------------------------------------


def _bump_case(noise: float, points: int, seed: int = 0):
    """The manufactured case exp_identify_chi.py uses, with optional noise."""
    import numpy as np

    from hfmarl.identification.closure import manufactured_case

    def bump_chi(rho):
        return 0.5 + 2.0 * np.exp(-(((rho - 0.5) / 0.15) ** 2))

    case = manufactured_case(bump_chi, n_rho=201)
    rng = np.random.default_rng(seed)
    idx = np.unique(np.linspace(5, len(case["rho"]) - 6, points).astype(int))
    rho, n_e = case["rho"][idx], case["n_e"][idx]
    T_e = case["T_e"][idx]
    S = case["source"][idx]
    if noise:
        T_e = T_e * (1 + rng.normal(0, noise, idx.size))
        S = S * (1 + rng.normal(0, noise, idx.size))
    return rho, n_e, T_e, S, case["chi_true"][idx]


def test_chi_gradient_guard_is_relative_not_absolute():
    """A dimensional threshold cannot have a correct default.

    `min_gradient=1e-8` sat ~5 orders of magnitude below the real gradient
    scale, so it masked nothing -- and chi = -q/(n dT/drho) has a denominator
    that noise can drive to near zero. Scaling every input by 1000 (keV -> eV)
    must not change which points are judged identifiable.
    """
    import numpy as np

    from hfmarl.identification.closure import chi_from_power_balance

    rho, n_e, T_e, S, _ = _bump_case(noise=0.03, points=40)
    a = chi_from_power_balance(rho, n_e, T_e, S)
    b = chi_from_power_balance(rho, n_e, T_e * 1000.0, S * 1000.0)
    assert a.valid_fraction == pytest.approx(b.valid_fraction)


def test_noisy_chi_error_is_monotone_in_noise():
    """The bug's signature: more noise scored BETTER.

    chi = -q/(n dT/drho), and one near-zero denominator produces an error
    large enough to set the mean by itself. Whether such a point occurs is
    luck, so the seed-averaged error was not monotone in noise -- with the old
    absolute guard, 5% noise scored 2.41 against 8% noise's 1.82. That
    inversion is what let a guard that never fired pass as "the estimator is
    just noise-sensitive".

    Averaged over seeds, because a single draw is itself non-monotone by luck;
    the claim is about the estimator, not about one realisation.
    """
    import numpy as np

    from hfmarl.identification.closure import (
        chi_from_power_balance,
        relative_error,
    )

    means, worst = [], []
    for noise in (0.0, 0.03, 0.05, 0.08):
        e = [
            relative_error(
                chi_from_power_balance(*_bump_case(noise, 40, seed)[:4]).chi,
                _bump_case(noise, 40, seed)[4],
            )
            for seed in range(12)
        ]
        means.append(float(np.mean(e)))
        worst.append(float(np.max(e)))

    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1)), means
    assert means[0] < 0.05, "clean-data accuracy must not regress"
    # Pre-fix the worst seed hit 6.5 at 5% noise, driven by one bad point.
    assert max(worst) < 3.0, worst


def test_relative_error_summary_exposes_a_single_blowup():
    """Mean far above median is the tell; reporting only the mean hides it."""
    import numpy as np

    from hfmarl.identification.closure import relative_error_summary

    truth = np.ones(20)
    est = np.ones(20)
    est[7] = 101.0  # one point 100x off
    s = relative_error_summary(est, truth)
    assert s["median"] == pytest.approx(0.0)
    assert s["mean"] == pytest.approx(5.0)
    assert s["max"] == pytest.approx(100.0)
    assert s["n"] == 20
