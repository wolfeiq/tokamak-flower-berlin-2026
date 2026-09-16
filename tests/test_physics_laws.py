"""Scaling laws, verified against first principles.

A formula can carry a wrong coefficient and still look plausible. These tests
pin the *exponents and proportionalities* instead, which is what actually
encodes the physics -- a transcription slip anywhere in the chain breaks at
least one of them.
"""

import numpy as np
import pytest
from dataclasses import replace

from hfmarl.devices.registry import all_devices, encoded_states, get
from hfmarl.federation.similarity import (
    ClientUpdate,
    aggregation_weights,
    similarity_distance,
    similarity_weight,
    staleness_factor,
)
from hfmarl.metrics.curves import bootstrap_ci, trailing_mean
from hfmarl.physics.dimensionless import (
    DimensionlessState,
    beta_normalised,
    beta_toroidal,
    coulomb_log_e,
    encode,
    nu_star_e,
    profile_shape_features,
    rho_star,
)

NU_KW = dict(R_major=6.2, epsilon=0.32, Z_eff=1.5)


# -- rho* = sqrt(m_i T_e)/(e B a) ---------------------------------------


def test_rho_star_scales_as_sqrt_ion_mass():
    b = float(rho_star(10.0, 5.0, 1.0, A_i=1.0))
    assert float(rho_star(10.0, 5.0, 1.0, A_i=4.0)) == pytest.approx(2 * b)


def test_rho_star_scales_as_sqrt_temperature():
    b = float(rho_star(10.0, 5.0, 1.0))
    assert float(rho_star(40.0, 5.0, 1.0)) == pytest.approx(2 * b)


def test_rho_star_inverse_in_field_and_minor_radius():
    b = float(rho_star(10.0, 5.0, 1.0))
    assert float(rho_star(10.0, 10.0, 1.0)) == pytest.approx(b / 2)
    assert float(rho_star(10.0, 5.0, 2.0)) == pytest.approx(b / 2)


# -- nu*_e = 6.921e-18 q R n Z_eff lnA / (eps^1.5 T^2) ------------------


def test_nu_star_linear_in_q():
    n0 = float(nu_star_e(1e20, 10.0, 3.0, **NU_KW))
    assert float(nu_star_e(1e20, 10.0, 6.0, **NU_KW)) == pytest.approx(2 * n0)


def test_nu_star_linear_in_major_radius():
    n0 = float(nu_star_e(1e20, 10.0, 3.0, **NU_KW))
    kw = dict(NU_KW, R_major=12.4)
    assert float(nu_star_e(1e20, 10.0, 3.0, **kw)) == pytest.approx(2 * n0)


def test_nu_star_linear_in_z_eff():
    n0 = float(nu_star_e(1e20, 10.0, 3.0, **NU_KW))
    kw = dict(NU_KW, Z_eff=3.0)
    assert float(nu_star_e(1e20, 10.0, 3.0, **kw)) == pytest.approx(2 * n0)


def test_nu_star_scales_as_eps_to_the_minus_three_halves():
    lo = float(nu_star_e(1e20, 10.0, 3.0, R_major=6.2, epsilon=0.1, Z_eff=1.5))
    hi = float(nu_star_e(1e20, 10.0, 3.0, R_major=6.2, epsilon=0.4, Z_eff=1.5))
    assert hi == pytest.approx(lo / 8.0)  # (4)^-1.5


def test_nu_star_scales_as_T_squared_inverse_up_to_the_coulomb_log():
    n0 = float(nu_star_e(1e20, 10.0, 3.0, **NU_KW))
    n1 = float(nu_star_e(1e20, 20.0, 3.0, **NU_KW))
    ln_ratio = float(coulomb_log_e(1e20, 20.0) / coulomb_log_e(1e20, 10.0))
    assert n1 / n0 == pytest.approx(0.25 * ln_ratio, rel=1e-9)


def test_nu_star_rises_with_density_faster_than_linearly_is_false():
    """n enters linearly; lnA falls slowly with n, so the ratio is just under 2."""
    n0 = float(nu_star_e(1e20, 10.0, 3.0, **NU_KW))
    n1 = float(nu_star_e(2e20, 10.0, 3.0, **NU_KW))
    assert 1.8 < n1 / n0 < 2.0


# -- Coulomb logarithm ---------------------------------------------------


def test_coulomb_log_falls_with_density_and_rises_with_temperature():
    assert coulomb_log_e(1e21, 10.0) < coulomb_log_e(1e19, 10.0)
    assert coulomb_log_e(1e20, 20.0) > coulomb_log_e(1e20, 5.0)


# -- beta ----------------------------------------------------------------


def test_beta_scales_with_density_temperature_and_inverse_field_squared():
    b = float(beta_toroidal(1e20, 5.0, 5.0, 3.0))
    assert float(beta_toroidal(2e20, 5.0, 5.0, 3.0)) == pytest.approx(2 * b)
    assert float(beta_toroidal(1e20, 10.0, 10.0, 3.0)) == pytest.approx(2 * b)
    assert float(beta_toroidal(1e20, 5.0, 5.0, 6.0)) == pytest.approx(b / 4)


def test_beta_N_is_inverse_in_plasma_current():
    b = beta_normalised(0.02, 2.0, 5.3, 10e6)
    assert beta_normalised(0.02, 2.0, 5.3, 20e6) == pytest.approx(b / 2)


# -- Greenwald -----------------------------------------------------------


def test_greenwald_depends_only_on_current_and_minor_radius():
    d = get("iter_like")
    assert replace(d, B_0=99.0).greenwald_density == pytest.approx(d.greenwald_density)
    assert replace(d, R_major=99.0).greenwald_density == pytest.approx(d.greenwald_density)
    assert replace(d, elongation=9.0).greenwald_density == pytest.approx(d.greenwald_density)


def test_greenwald_scales_inversely_with_area():
    d = get("iter_like")
    assert replace(d, a_minor=2 * d.a_minor).greenwald_density == pytest.approx(
        d.greenwald_density / 4
    )


# -- profile shape -------------------------------------------------------


def test_flat_profile_has_unit_peaking_zero_gradient_unit_edge():
    f = profile_shape_features(np.ones(51), np.linspace(0, 1, 51))
    assert np.allclose(f, [1.0, 0.0, 1.0], atol=1e-6)


def test_peaked_profile_is_detected():
    rho = np.linspace(0, 1, 51)
    f = profile_shape_features(1 - 0.9 * rho**2, rho)
    assert f[0] > 1.0 and f[2] < 1.0 and f[1] < 0.0


# -- the encoder as a whole ----------------------------------------------


def test_hotter_plasma_raises_rho_star_and_lowers_collisionality():
    d = get("iter_like")
    kw = dict(q95=3.2, B_0=d.B_0, R_major=d.R_major, a_minor=d.a_minor,
              Ip=d.Ip_nominal)
    lo = encode(T_e_keV=4.0, T_i_keV=4.0, n_e=0.7e20, **kw)
    hi = encode(T_e_keV=16.0, T_i_keV=16.0, n_e=0.7e20, **kw)
    assert hi.rho_star > lo.rho_star
    assert hi.nu_star < lo.nu_star


# -- the similarity metric ----------------------------------------------


def test_metric_is_translation_invariant_in_log_space():
    """A common rescaling of rho* must not change a distance -- that is what
    makes the metric about ratios rather than absolute size."""
    a = DimensionlessState(1e-3, 0.02, 2.0, 3.0)
    b = DimensionlessState(3e-3, 0.03, 2.2, 3.1)
    a10 = DimensionlessState(1e-2, 0.02, 2.0, 3.0)
    b10 = DimensionlessState(3e-2, 0.03, 2.2, 3.1)
    assert similarity_distance(a, b) == pytest.approx(similarity_distance(a10, b10))


def test_weight_is_strictly_decreasing_in_distance():
    w = [similarity_weight(x) for x in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(w[i] > w[i + 1] for i in range(len(w) - 1))


def test_staleness_matches_the_fedbuff_form():
    """FedBuff (Nguyen et al., AISTATS 2022): 1/sqrt(1+tau)."""
    u = ClientUpdate("d", "thermal", {}, DimensionlessState(1e-3, 0.02, 2.0, 3.0), 10, 0)
    for tau in (0, 1, 3, 15):
        assert staleness_factor(u, tau, 0) == pytest.approx((1 + tau) ** -0.5)


def test_aggregation_weights_normalise_and_favour_the_nearer_peer():
    a = DimensionlessState(1e-3, 0.02, 2.0, 3.0)
    near = DimensionlessState(3e-3, 0.03, 2.2, 3.1)
    far = DimensionlessState(1e-2, 0.09, 1.5, 3.5)
    ups = [ClientUpdate("near", "thermal", {}, near, 10, 0),
           ClientUpdate("far", "thermal", {}, far, 10, 0)]
    w = aggregation_weights(ups, a, current_round=0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]


# -- statistics ----------------------------------------------------------


def test_trailing_mean_window_one_is_the_identity():
    x = np.arange(10.0)
    assert np.allclose(trailing_mean(x, 1), x)


def test_trailing_mean_full_window_is_the_running_mean():
    x = np.arange(10.0)
    assert np.allclose(trailing_mean(x, 10), np.cumsum(x) / np.arange(1, 11))


def test_bootstrap_of_a_constant_is_that_constant():
    lo, hi = bootstrap_ci(np.full(20, 5.0))
    assert lo == hi == 5.0


# -- cross-device sanity -------------------------------------------------


@pytest.mark.parametrize("device", [d.name for d in all_devices()])
def test_nominal_operating_points_are_physically_plausible(device):
    s = encoded_states()[device]
    assert 1e-4 < s.rho_star < 5e-2, "rho* outside any tokamak's range"
    assert 0.0 < s.nu_star < 1.0, "should be in the banana regime"
    assert 0.5 < s.beta_N < 4.0, "beta_N outside the operable band"
    assert 2.0 < s.q95 < 6.0, "q outside the operable band"


def test_rho_star_ordering_tracks_machine_size():
    s = encoded_states()
    assert s["iter_like"].rho_star < s["diiid_like"].rho_star < s["tcv_like"].rho_star


# -- robustness of the primary metric to outliers ------------------------


def _curve(seed, n=400, rate=60.0, outlier_at=None, outlier=None):
    from hfmarl.metrics.log import RunLog, ShotRecord

    rng = np.random.default_rng(seed)
    r = RunLog("centralised", "d", seed)
    for i in range(n):
        r.add(ShotRecord(shot=i, reward=float(-10 * np.exp(-i / rate) + rng.normal(0, 0.2)),
                         steps=1))
    if outlier_at is not None:
        r.shots[outlier_at] = ShotRecord(shot=outlier_at, reward=outlier, steps=1)
    return r


@pytest.mark.parametrize("outlier", [-1e3, -1e5, -1e12])
def test_one_bad_shot_cannot_move_the_threshold(outlier):
    """Bug: the threshold anchored on min() over raw rewards. Per-step cost is
    quadratic in tracking error, so one excursion could be orders of magnitude
    out -- a single -1e5 shot dragged the threshold from -2.0 to -20000, at
    which point every condition cleared it on shot 1 and shots-to-threshold
    collapsed from ~104 to 1. The primary metric read as an instant win for
    everything."""
    from hfmarl.metrics.curves import threshold_from_reference

    clean = threshold_from_reference([_curve(s) for s in range(3)])
    dirty = threshold_from_reference(
        [_curve(0, outlier_at=200, outlier=outlier)] + [_curve(s) for s in (1, 2)]
    )
    assert dirty == pytest.approx(clean, rel=0.05)


def test_shots_to_threshold_is_unchanged_by_an_outlier():
    from hfmarl.metrics.curves import shots_to_threshold, threshold_from_reference

    runs = [_curve(s) for s in range(3)]
    clean_t = threshold_from_reference(runs)
    dirty_t = threshold_from_reference(
        [_curve(0, outlier_at=200, outlier=-1e9)] + [_curve(s) for s in (1, 2)]
    )
    assert shots_to_threshold(runs, clean_t).median == pytest.approx(
        shots_to_threshold(runs, dirty_t).median
    )


def test_plateau_is_robust_to_a_blown_up_tail_shot():
    """Bug: the mean within a seed took the plateau from -0.03 to -12500."""
    from hfmarl.metrics.curves import asymptotic_performance
    from hfmarl.metrics.log import ShotRecord

    clean = asymptotic_performance([_curve(0)])[0]
    dirty_run = _curve(0)
    dirty_run.shots[-1] = ShotRecord(shot=399, reward=-1e6, steps=1)
    dirty = asymptotic_performance([dirty_run])[0]
    assert abs(dirty - clean) < 0.5


def test_plateau_ignores_non_finite_shots():
    from hfmarl.metrics.curves import asymptotic_performance
    from hfmarl.metrics.log import ShotRecord

    run = _curve(0)
    run.shots[-1] = ShotRecord(shot=399, reward=float("nan"), steps=1)
    assert np.isfinite(asymptotic_performance([run])[0])


def test_threshold_rejects_a_reference_that_never_improves():
    """Interpolating between a start and a plateau is meaningless if the run
    got no better; a silent negative threshold would be worse than an error."""
    from hfmarl.metrics.curves import threshold_from_reference
    from hfmarl.metrics.log import RunLog, ShotRecord

    flat = RunLog("centralised", "d", 0)
    for i in range(100):
        flat.add(ShotRecord(shot=i, reward=-5.0, steps=1))
    with pytest.raises(ValueError, match="does not improve"):
        threshold_from_reference([flat])


def test_threshold_lies_between_start_and_plateau():
    from hfmarl.metrics.curves import asymptotic_performance, threshold_from_reference

    runs = [_curve(s) for s in range(3)]
    t = threshold_from_reference(runs, fraction=0.8)
    plateau = asymptotic_performance(runs)[0]
    start = float(np.median([np.median(r.rewards()[:25]) for r in runs]))
    assert start < t < plateau


# -- the per-step cost floor --------------------------------------------


def test_per_step_cost_is_bounded():
    """One excursion must not swamp the episode return and every statistic
    downstream of it."""
    from hfmarl.envs.torax_env import ToraxDeviceEnv

    env = ToraxDeviceEnv(get("iter_like"), task="easy")

    class Extreme:
        beta_N = 1e6
        q95 = 3.0
        fgw_n_e_line_avg = 0.5

    env.core.post = Extreme()
    cost = env._control_cost(env.limits.evaluate(env.core.limit_inputs()))
    assert cost >= -env.violation_penalty


def test_cost_floor_preserves_the_no_free_exit_guarantee():
    """The floor must stay strictly above -violation_penalty, or terminating
    early becomes exactly as cheap as finishing."""
    from hfmarl.envs.torax_env import ToraxDeviceEnv

    env = ToraxDeviceEnv(get("iter_like"), task="hard")

    class Extreme:
        beta_N = 1e6
        q95 = 3.0
        fgw_n_e_line_avg = 0.5

    env.core.post = Extreme()
    per = env._control_cost(env.limits.evaluate(env.core.limit_inputs()))
    assert per + env.violation_penalty > 0
