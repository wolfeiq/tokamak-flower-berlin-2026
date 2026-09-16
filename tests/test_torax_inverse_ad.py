"""The differentiable path through TORAX. Needs a live TORAX; skipped without it.

Three separate bugs sat on this path and each masked the next, so each has a
test here. All three were invisible to the rest of the suite because nothing
outside the gate scripts ever differentiated through TORAX.
"""

import numpy as np
import pytest

torax = pytest.importorskip("torax")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.torax

from hfmarl.devices.registry import get  # noqa: E402
from hfmarl.envs.torax_config import build_config  # noqa: E402
from hfmarl.identification.torax_inverse import (  # noqa: E402
    CANDIDATE_UNKNOWNS,
    _updater_for_paths,
    build_loss,
)


@pytest.fixture(scope="module")
def cfg():
    return build_config(get("iter_like"), t_final=1.0)


def test_updater_wraps_each_leaf_in_the_type_torax_demands(cfg):
    """A bare float is refused for TimeVarying* leaves -- it must be wrapped.

    Z_eff is a TimeVaryingArray, so passing a float raised
        ValueError: To replace a `TimeVaryingArray` use a
        `TimeVaryingArrayUpdate`, got <class 'numpy.float64'> instead.
    and the inverse path could not run at all.
    """
    from torax.experimental import make_step_fn

    step_fn = make_step_fn(torax.ToraxConfig.from_dict(cfg))
    paths = list(CANDIDATE_UNKNOWNS)
    wrap = _updater_for_paths(step_fn.runtime_params_provider, paths)

    # The real check: TORAX itself accepts every wrapped value.
    step_fn.runtime_params_provider.update_provider_from_mapping(
        {p: wrap[p](CANDIDATE_UNKNOWNS[p][0]) for p in paths}
    )


def test_time_varying_array_update_has_a_1d_rho_norm(cfg):
    """`value` is (t, rhon) but `rho_norm` is (rhon,), not (1, rhon)."""
    from torax._src.torax_pydantic import interpolated_param_2d as p2
    from torax.experimental import make_step_fn

    step_fn = make_step_fn(torax.ToraxConfig.from_dict(cfg))
    wrap = _updater_for_paths(step_fn.runtime_params_provider,
                              ["plasma_composition.Z_eff"])
    upd = wrap["plasma_composition.Z_eff"](1.6)

    assert isinstance(upd, p2.TimeVaryingArrayUpdate)
    assert upd.rho_norm.ndim == 1
    assert upd.value.shape[1] == upd.rho_norm.shape[0]


def test_gradients_are_finite_not_nan(cfg):
    """The payoff test: a NaN gradient reads exactly like an unidentifiable
    parameter, so this is the one that keeps `gate_identify` honest.

    Two causes, both fixed: reverse-mode AD cannot traverse TORAX's
    `lax.while_loop` at all, and `numerics.adaptive_dt=True` makes the tangent
    of the solver-retry branch NaN.
    """
    names = list(CANDIDATE_UNKNOWNS)
    truth = np.array([CANDIDATE_UNKNOWNS[n][0] for n in names])

    _, simulate = build_loss(cfg, names, {"T_e": np.zeros((2, 25))}, 0.5, 2)
    obs = {k: np.asarray(v) for k, v in simulate(truth).items()}
    assert np.isfinite(obs["T_e"]).all()

    loss_and_grad, _ = build_loss(cfg, names, obs, 0.5, 2)
    value, grad = loss_and_grad(truth * 1.1)
    grad = np.asarray(grad)

    assert np.isfinite(value)
    assert np.isfinite(grad).all(), f"NaN gradient for {[n for n, g in zip(names, grad) if not np.isfinite(g)]}"
    assert np.abs(grad).max() > 0, "every gradient is exactly zero"


def test_build_loss_forces_adaptive_dt_off(cfg):
    """It must not depend on the caller remembering, and must not mutate theirs."""
    cfg_in = dict(cfg)
    cfg_in["numerics"] = {**cfg_in["numerics"], "adaptive_dt": True}
    before = dict(cfg_in["numerics"])

    build_loss(cfg_in, ["plasma_composition.Z_eff"],
               {"T_e": np.zeros((1, 25))}, 0.5, 1)

    assert cfg_in["numerics"] == before, "build_loss mutated the caller's config"
