"""Small policy networks.

SPEC.md §8 is emphatic: small networks throughout. Anything large contradicts
the millisecond control-loop premise -- PACMAN's deployed controllers infer in
0.2-15 ms -- and federated payloads should be kilobytes, which removes the
standard bandwidth objection to federated RL. ``payload_bytes`` exists so that
claim can be stated as a measured number rather than asserted.

Pure NumPy, deliberately. A tanh MLP with one hidden layer of 16 units is a
few hundred parameters; putting a deep-learning framework behind that would
add a dependency, a GPU context competing with TORAX's, and no benefit. It
also keeps the federation code framework-agnostic: an update is a dict of
arrays, and the aggregator neither knows nor cares what produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLPPolicy:
    """One-hidden-layer tanh MLP mapping observation -> action in [-1, 1].

    The output is tanh-bounded, so every actuator's command lives on the same
    scale regardless of its physical units. ``ActuatorSpec.from_unit`` maps it
    onto the device's engineering envelope. This is what lets a thermal policy
    trained on one device be applied to another at all.
    """

    obs_dim: int
    act_dim: int
    hidden: int = 16
    W1: np.ndarray = None  # type: ignore[assignment]
    b1: np.ndarray = None  # type: ignore[assignment]
    W2: np.ndarray = None  # type: ignore[assignment]
    b2: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.W1 is None:
            rng = np.random.default_rng(0)
            # Xavier-ish init; scale matters little at this size but keeps the
            # initial policy from saturating tanh.
            self.W1 = rng.normal(0, 1 / np.sqrt(self.obs_dim), (self.obs_dim, self.hidden))
            self.b1 = np.zeros(self.hidden)
            self.W2 = rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, self.act_dim))
            self.b2 = np.zeros(self.act_dim)

    def act(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(np.asarray(obs, float) @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    # -- flat vector view, for CEM and for federation ---------------------

    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat(self, v: np.ndarray) -> None:
        v = np.asarray(v, float)
        if v.size != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {v.size}")
        i = 0
        for name, shape in self._shapes():
            n = int(np.prod(shape))
            setattr(self, name, v[i : i + n].reshape(shape))
            i += n

    def _shapes(self):
        return [
            ("W1", (self.obs_dim, self.hidden)),
            ("b1", (self.hidden,)),
            ("W2", (self.hidden, self.act_dim)),
            ("b2", (self.act_dim,)),
        ]

    @property
    def n_params(self) -> int:
        return sum(int(np.prod(s)) for _, s in self._shapes())

    def payload_bytes(self, dtype=np.float32) -> int:
        """Size of one federated update on the wire.

        Answers the bandwidth objection in SPEC.md §8 with a number. float32
        is what would actually be transmitted; float64 doubles it and buys
        nothing for a policy delta.
        """
        return self.n_params * np.dtype(dtype).itemsize

    def as_state_dict(self) -> dict[str, np.ndarray]:
        """What gets federated. Keys are stable across devices by construction."""
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def load_state_dict(self, sd: dict[str, np.ndarray]) -> None:
        for k in ("W1", "b1", "W2", "b2"):
            if k not in sd:
                raise KeyError(f"state dict missing {k!r}")
            cur = getattr(self, k)
            if sd[k].shape != cur.shape:
                raise ValueError(
                    f"{k}: shape {sd[k].shape} does not match {cur.shape}. "
                    "Role-matched federation requires identical architectures "
                    "within a cluster (SPEC.md §2)."
                )
            setattr(self, k, np.array(sd[k], float))


def make_policy(obs_dim: int, act_dim: int, hidden: int = 16, seed: int = 0) -> MLPPolicy:
    rng = np.random.default_rng(seed)
    p = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
    p.W1 = rng.normal(0, 1 / np.sqrt(obs_dim), (obs_dim, hidden))
    p.b1 = np.zeros(hidden)
    p.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, act_dim))
    p.b2 = np.zeros(act_dim)
    return p
