"""Policy networks and the federated payload claim (SPEC.md §8)."""

import numpy as np
import pytest

from hfmarl.agents.policy import MLPPolicy, make_policy


def test_action_is_bounded():
    p = make_policy(7, 3)
    for scale in (0.0, 1.0, 1e3):
        a = p.act(np.full(7, scale))
        assert a.shape == (3,) and np.all(np.abs(a) <= 1.0)


def test_flat_roundtrip_is_exact():
    p = make_policy(7, 2, seed=3)
    f = p.get_flat().copy()
    p.set_flat(np.zeros_like(f))
    p.set_flat(f)
    assert np.allclose(p.get_flat(), f)


def test_set_flat_rejects_wrong_size():
    p = make_policy(7, 2)
    with pytest.raises(ValueError):
        p.set_flat(np.zeros(p.n_params + 1))


def test_payload_is_kilobytes_not_megabytes():
    """SPEC.md §8: state the payload size to remove the bandwidth objection."""
    p = make_policy(7, 3, hidden=16)
    assert p.payload_bytes() < 4096, "policy is too large for the ms control premise"
    assert p.n_params < 1000


def test_state_dict_roundtrip():
    p = make_policy(7, 2, seed=1)
    q = make_policy(7, 2, seed=2)
    assert not np.allclose(p.act(np.ones(7)), q.act(np.ones(7)))
    q.load_state_dict(p.as_state_dict())
    assert np.allclose(p.act(np.ones(7)), q.act(np.ones(7)))


def test_state_dict_rejects_mismatched_architecture():
    """Role-matched federation requires identical shapes within a cluster."""
    p = make_policy(7, 2, hidden=16)
    q = make_policy(7, 2, hidden=8)
    with pytest.raises(ValueError, match="does not match"):
        q.load_state_dict(p.as_state_dict())


def test_state_dict_rejects_missing_keys():
    p = make_policy(7, 2)
    sd = p.as_state_dict()
    del sd["W2"]
    with pytest.raises(KeyError):
        p.load_state_dict(sd)


def test_same_seed_gives_same_policy():
    """SPEC.md §8: seed everything."""
    a, b = make_policy(5, 2, seed=11), make_policy(5, 2, seed=11)
    assert np.allclose(a.get_flat(), b.get_flat())


def test_phase1_payload_matches_the_number_in_the_readme():
    """README states an exact payload; pin it so the two cannot drift apart.

    The previous test only asserted `< 4096`, so when `_observe()` grew from 7
    to 9 entries the README's "162 parameters / 648 bytes" quietly became
    wrong while the suite stayed green. SPEC.md 8 asks for this figure to be
    measured rather than asserted, which only holds if it is checked.
    """
    from hfmarl.devices.registry import get
    from hfmarl.envs.actuators import bank_from_device
    from hfmarl.envs.torax_env import ToraxDeviceEnv

    env = ToraxDeviceEnv(get("iter_like"), task="easy")
    obs_dim = env._observe().size
    act_dim = len(bank_from_device(get("iter_like"), ("thermal",)))

    p = make_policy(obs_dim, act_dim)
    assert (obs_dim, act_dim) == (9, 2)
    assert p.n_params == 194
    assert p.payload_bytes() == 776
