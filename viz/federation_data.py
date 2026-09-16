"""The federation graph, computed from the real aggregation code.

Everything this module reports is produced by running the modules the
experiment runs: ``physics.dimensionless.encode`` for the operating points,
``federation.similarity`` for the distances and kernel weights, and
``federation.server.FedBuffServer`` for the personalised aggregate --
including its admissibility gates, staleness factor, alignment, centred
clipping and geometric median.

The point of doing it this way rather than hard-coding pretty numbers is that
the picture is then falsifiable. If the bandwidth is set so that every peer is
ignored, the scene goes dark and the diagnostics say ``uniform_fallback`` or
``rejected_all`` -- the same two silent failure modes ``describe_device_set``
warns about, made visible instead of discovered after a training run.

STANDING CAVEAT, inherited and not fixable here: ``OPERATING_POINTS`` is still
the hand-written nominal table, not TORAX output (README "Still open"). Every
distance below is therefore a distance between tabulated points, not between
measured operating regions. ``StateRegion`` exists precisely because that
distinction bit once; the app labels it rather than hiding it.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from hfmarl.agents.policy import make_policy
from hfmarl.devices import registry
from hfmarl.federation.server import FedBuffServer
from hfmarl.federation.similarity import (
    ClientUpdate,
    similarity_distance,
    similarity_weight,
    staleness_factor,
    suggest_bandwidth,
)
from hfmarl.physics.dimensionless import DimensionlessState

# The three federation channels. Role-matched aggregation means thermal only
# ever mixes with thermal (SPEC.md section 2), which is why these are three
# separate hubs in the scene and not one server with three cables into it.
CLUSTERS: tuple[str, ...] = ("thermal", "particle", "current")

CLUSTER_META: dict[str, dict[str, str]] = {
    "thermal": {
        "label": "Thermal",
        "color": "#ff7b3d",
        "blurb": (
            "Auxiliary heating. Two live actuators; ICRH has no independent "
            "TORAX 1.4.3 source and is excluded rather than aliased onto "
            "generic_heat."
        ),
    },
    "particle": {
        "label": "Particle",
        "color": "#2dd4bf",
        "blurb": "Fuelling. One live actuator, the gas puff.",
    },
    "current": {
        "label": "Current",
        "color": "#a371f7",
        "blurb": (
            "Plasma current. Ip starts at its nominal flat-top value, not at "
            "the lower bound -- otherwise step 1 drops the current by 90%."
        ),
    },
}

DEVICE_META: dict[str, dict[str, str]] = {
    "iter_like": {"label": "ITER-like", "badge": "Giant / low-rho*", "color": "#58a6ff"},
    "sparc_like": {"label": "SPARC-like", "badge": "Compact / high-field", "color": "#f778ba"},
    "diiid_like": {"label": "DIII-D-like", "badge": "Medium / low-A", "color": "#3fb950"},
    "tcv_like": {"label": "TCV-like", "badge": "Small / highly shaped", "color": "#d29922"},
}

# Where the machine each client is modelled on actually stands. This is the
# honest version of "the federation is geographically dispersed": these are
# real coordinates, not a decorative scatter, and they span two continents and
# nine time zones -- which is the latency and data-sovereignty story the
# federated setup exists to answer.
#
# ``offset_deg`` is a CARTOGRAPHIC offset only, in (lon, lat) degrees. Cadarache
# and Lausanne are ~340 km apart, so at world scale the ITER-like and TCV-like
# stations would be drawn on top of each other. The scene therefore pins the
# true coordinate and draws the station body offset from it with a leader line
# back to the pin -- the standard fix for label collision. ``lat``/``lon`` stay
# true and are what the tooltip reports.
SITES: dict[str, dict[str, object]] = {
    "iter_like": {
        "lab": "ITER Organization",
        "place": "Cadarache, France",
        "lat": 43.7085,
        "lon": 5.7561,
        "offset_deg": (-5.0, -6.0),
    },
    "sparc_like": {
        "lab": "Commonwealth Fusion Systems",
        "place": "Devens, Massachusetts, USA",
        "lat": 42.5376,
        "lon": -71.6070,
        "offset_deg": (2.0, 6.0),
    },
    "diiid_like": {
        "lab": "General Atomics",
        "place": "San Diego, California, USA",
        "lat": 32.8944,
        "lon": -117.2350,
        "offset_deg": (-4.0, -5.0),
    },
    "tcv_like": {
        "lab": "Swiss Plasma Center, EPFL",
        "place": "Lausanne, Switzerland",
        "lat": 46.5191,
        "lon": 6.5668,
        "offset_deg": (6.0, 5.0),
    },
}

# The aggregation server has no physical home in the repository, so the scene
# does not invent a data centre for it: it floats over Greenland,
# with a drop line to the map so the
# position is read as "nowhere in particular" rather than "here".
SERVER_SITE: dict[str, float] = {"lat": 72.0, "lon": -42.0}

# The Phase-1 policy shape, from README "The federated payload is 776 bytes":
# 9-dim observation, 16 hidden units. The action dimension is the number of
# LIVE actuators in the cluster, so each channel federates a differently-shaped
# network -- which is fine, because channels never mix.
OBS_DIM = 9
HIDDEN = 16


@dataclass(frozen=True)
class AgentNode:
    """One actuator agent: a box in the scene, a row in the action vector."""

    name: str
    cluster: str
    lo: float
    hi: float
    units: str
    available: bool
    torax_path: str


@dataclass
class ChannelLink:
    """One device's participation in one federation channel."""

    cluster: str
    weight: float        # normalised contribution to the target's aggregate
    kernel: float        # raw similarity kernel, before staleness and safety
    staleness: float     # (1+tau)^-alpha * epoch_penalty^n_epochs
    admissible: bool
    is_self: bool
    # Not in the buffer at all, as opposed to in it and refused. The only way
    # this happens today is ``include_self=False``, and the two must not look
    # alike: one is a configuration choice, the other is the physics gate
    # firing.
    excluded: bool = False
    agents: list[AgentNode] = field(default_factory=list)


@dataclass
class DeviceNode:
    key: str
    label: str
    badge: str
    color: str
    note: str
    R_major: float
    a_minor: float
    B_0: float
    elongation: float
    Ip_MA: float
    aspect_ratio: float
    q_cyl: float
    state: dict[str, float]
    distance: float          # dimensionless distance to the target device
    round_age: int
    config_epoch_lag: int
    violation_rate: float
    n_samples: int
    links: dict[str, ChannelLink]

    @property
    def scale(self) -> float:
        """Visual size.

        R^0.45 keeps TCV visible next to ITER without lying about which one is
        bigger: the ordering is preserved, the ratio is compressed, and the
        stat table carries the real metres.
        """
        return float((self.R_major / 6.2) ** 0.45)


@dataclass
class ChannelDiagnostics:
    """What the server actually did, per channel."""

    cluster: str
    act_dim: int
    n_params: int
    payload_bytes: int
    aggregated: bool
    clipped: int
    uniform_fallback: bool
    rejected_all: bool
    error: str
    aggregate_shift: float   # ||aggregate - the target's own weights||_2
    effective_peers: float   # 1 / sum(w^2): how many devices actually spoke


@dataclass
class FederationGraph:
    target: str
    bandwidth: float
    suggested_bandwidth: float
    use_similarity: bool
    include_self: bool
    use_safety: bool
    use_sample_count: bool
    rule: str
    align: bool
    current_round: int
    devices: list[DeviceNode]
    diagnostics: dict[str, ChannelDiagnostics]
    distance_matrix: dict[str, dict[str, float]]
    # Which devices actually published this round. Kept so the participation
    # sweep can rebuild this graph's settings without the caller passing them
    # in twice.
    participants: tuple[str, ...] = ()
    clip_factor: float | None = 2.0

    def device(self, key: str) -> DeviceNode:
        for d in self.devices:
            if d.key == key:
                return d
        raise KeyError(key)


def median_bandwidth() -> float:
    """The median-heuristic bandwidth over the device set.

    Separate from ``build_graph`` so the UI can offer it as a default without
    running a whole aggregation round to find out what it is.
    """
    states = registry.encoded_states()
    return float(suggest_bandwidth([states[n] for n in states]))


def _agents_for(device, cluster: str) -> list[AgentNode]:
    """Every actuator in the cluster, INCLUDING the unavailable ones.

    ``actuators_for`` hides ``available=False`` by default, which is right for
    the action space and wrong for a diagram: ICRH being defined-but-excluded
    is a deliberate decision documented in ``_standard_actuators``, and a
    picture that silently omitted it would show a thermal cluster that had
    always had exactly two agents.
    """
    return [
        AgentNode(
            name=a.name,
            cluster=a.cluster,
            lo=a.lo,
            hi=a.hi,
            units=a.units,
            available=a.available,
            torax_path=a.torax_path,
        )
        for a in device.actuators_for(cluster, include_unavailable=True)
    ]


def _act_dim(device, cluster: str) -> int:
    return len(device.actuators_for(cluster))


def build_graph(
    target: str = "iter_like",
    bandwidth: float | None = None,
    *,
    use_similarity: bool = True,
    include_self: bool = True,
    use_safety: bool = True,
    use_sample_count: bool = False,
    rule: str = "geomedian",
    align: bool = True,
    clip_factor: float | None = 2.0,
    current_round: int = 12,
    round_age: dict[str, int] | None = None,
    config_epoch_lag: dict[str, int] | None = None,
    violation_rate: dict[str, float] | None = None,
    n_samples: dict[str, int] | None = None,
    participants: Iterable[str] | None = None,
) -> FederationGraph:
    """Run one personalised aggregation round and report everything it did.

    ``target`` is the device receiving the aggregate. The weighting is computed
    relative to where THAT device operates, so ITER and TCV get different
    mixtures of the same buffer -- which is the point of SPEC.md section 4b and
    the reason there is no single global model to draw.

    ``participants`` is the set of devices that actually published an update.
    ``None`` means all of them. A device left out does not reach the buffer at
    all, so it comes back ``excluded=True`` with zero weight: a
    client that stayed silent is a different thing from one the physics gate
    refused, and the scene draws them differently. The target still receives
    the aggregate whether or not it contributed to it.
    """
    states: dict[str, DimensionlessState] = registry.encoded_states()
    names = list(states)
    if target not in states:
        raise KeyError(f"unknown target device {target!r}; have {sorted(states)}")

    suggested = suggest_bandwidth([states[n] for n in names])
    bw = suggested if bandwidth is None else float(bandwidth)

    round_age = round_age or {}
    config_epoch_lag = config_epoch_lag or {}
    violation_rate = violation_rate or {}
    n_samples = n_samples or {}
    sending = set(names) if participants is None else {
        k for k in participants if k in states
    }

    # One update per (device, channel). The policy weights are a stand-in --
    # this app does not train -- but the SHAPES are the real ones, so the
    # payload sizes, the alignment permutation and the clipping radius are all
    # computed on vectors of the size that would actually be transmitted.
    updates: dict[str, list[ClientUpdate]] = {c: [] for c in CLUSTERS}
    flats: dict[tuple[str, str], np.ndarray] = {}
    for seed, name in enumerate(names):
        dev = registry.get(name)
        age = int(round_age.get(name, 0))
        lag = int(config_epoch_lag.get(name, 0))
        for ci, cluster in enumerate(CLUSTERS):
            act_dim = _act_dim(dev, cluster)
            if act_dim == 0:
                continue
            policy = make_policy(OBS_DIM, act_dim, HIDDEN, seed=100 * seed + ci)
            flat = policy.get_flat()
            # Always recorded, even for a silent device: the target's own flat
            # is the `reference` the aggregate is compared against, and the
            # target is allowed to receive without sending.
            flats[(name, cluster)] = flat
            if name not in sending:
                continue
            updates[cluster].append(
                ClientUpdate(
                    device=name,
                    cluster=cluster,
                    weights={"flat": flat},
                    state=states[name],
                    n_samples=int(n_samples.get(name, 40)),
                    round_produced=current_round - age,
                    # The server sits at config epoch 0 here, so a peer that
                    # changed its wall or heating since is BEHIND. Expressed as
                    # a lag so the control reads "epochs behind", not
                    # "absolute epoch".
                    config_epoch=-lag,
                    violation_rate=float(violation_rate.get(name, 0.0)),
                )
            )

    diagnostics: dict[str, ChannelDiagnostics] = {}
    weights: dict[str, dict[str, float]] = {c: {} for c in CLUSTERS}

    for cluster in CLUSTERS:
        buf = updates[cluster]
        act_dim = _act_dim(registry.get(target), cluster)
        n_params = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * act_dim + act_dim
        if not buf:
            # An empty buffer is not an empty channel. A channel with no
            # actuators anywhere has nothing to report and is skipped; a
            # channel where every client merely stayed silent still exists,
            # still has a payload size, and simply formed no aggregate. The
            # per-channel card has to distinguish those, or switching every
            # station off would claim the actuators had disappeared.
            if any(_act_dim(registry.get(n), cluster) for n in names):
                diagnostics[cluster] = ChannelDiagnostics(
                    cluster=cluster,
                    act_dim=act_dim,
                    n_params=n_params,
                    payload_bytes=n_params * 4,
                    aggregated=False,
                    clipped=0,
                    uniform_fallback=False,
                    rejected_all=False,
                    error="",
                    aggregate_shift=float("nan"),
                    effective_peers=0.0,
                )
            continue
        server = FedBuffServer(
            bandwidth=bw,
            use_similarity=use_similarity,
            use_sample_count=use_sample_count,
            use_safety=use_safety,
            include_self=include_self,
            rule=rule,
            align=align,
            clip_factor=clip_factor,
            shape=(OBS_DIM, HIDDEN, act_dim),
        )
        for u in buf:
            server.publish(u)

        reference = flats[(target, cluster)]
        aggregate = server.aggregate_for(
            target=target,
            target_state=states[target],
            cluster=cluster,
            current_round=current_round,
            current_config_epoch=0,
            reference=reference,
        )
        table = server.weight_table().get(target, {})
        weights[cluster] = {k: float(v) for k, v in table.items()}

        w = np.array(list(weights[cluster].values()), dtype=float)
        sq = float(np.sum(w**2))
        eff = float(1.0 / sq) if sq > 0 else 0.0
        shift = (
            float(np.linalg.norm(np.asarray(aggregate, float) - reference))
            if aggregate is not None
            else float("nan")
        )
        diagnostics[cluster] = ChannelDiagnostics(
            cluster=cluster,
            act_dim=act_dim,
            n_params=n_params,
            payload_bytes=n_params * 4,  # float32 is what goes on the wire
            aggregated=aggregate is not None,
            clipped=int(server.last_clipped),
            uniform_fallback=server.uniform_fallback_rounds > 0,
            rejected_all=bool(server.last_rejected_all),
            error=str(server.last_error),
            aggregate_shift=shift,
            effective_peers=eff,
        )

    # Per-device nodes. The two factors that go into a weight are reported
    # separately, because a peer can be near and stale or far and fresh, and
    # the product alone cannot tell you which.
    nodes: list[DeviceNode] = []
    for name in names:
        dev = registry.get(name)
        meta = DEVICE_META.get(name, {"label": name, "badge": "", "color": "#8b949e"})
        d = similarity_distance(states[name], states[target])
        links: dict[str, ChannelLink] = {}
        for cluster in CLUSTERS:
            agents = _agents_for(dev, cluster)
            if not agents:
                continue
            w_i = weights[cluster].get(name, 0.0)
            match = [u for u in updates[cluster] if u.device == name]
            stale = staleness_factor(match[0], current_round, 0) if match else 0.0
            in_buffer = name in weights[cluster]
            links[cluster] = ChannelLink(
                cluster=cluster,
                weight=w_i,
                kernel=similarity_weight(d, bw) if use_similarity else 1.0,
                staleness=float(stale),
                # A zero weight from a device that IS in the buffer means the
                # admissibility gate fired, or the kernel underflowed to
                # exactly zero. The server keeps the row either way.
                admissible=in_buffer and w_i > 0.0,
                is_self=(name == target),
                excluded=not in_buffer,
                agents=agents,
            )
        nodes.append(
            DeviceNode(
                key=name,
                label=meta["label"],
                badge=meta["badge"],
                color=meta["color"],
                note=dev.note,
                R_major=dev.R_major,
                a_minor=dev.a_minor,
                B_0=dev.B_0,
                elongation=dev.elongation,
                Ip_MA=dev.Ip_nominal / 1e6,
                aspect_ratio=dev.aspect_ratio,
                q_cyl=dev.q_cylindrical,
                state=states[name].as_dict(),
                distance=float(d),
                round_age=int(round_age.get(name, 0)),
                config_epoch_lag=int(config_epoch_lag.get(name, 0)),
                violation_rate=float(violation_rate.get(name, 0.0)),
                n_samples=int(n_samples.get(name, 40)),
                links=links,
            )
        )

    dmat = {
        a: {b: float(similarity_distance(states[a], states[b])) for b in names}
        for a in names
    }

    return FederationGraph(
        target=target,
        bandwidth=bw,
        suggested_bandwidth=float(suggested),
        use_similarity=use_similarity,
        include_self=include_self,
        use_safety=use_safety,
        use_sample_count=use_sample_count,
        rule=rule,
        align=align,
        current_round=current_round,
        devices=nodes,
        diagnostics=diagnostics,
        distance_matrix=dmat,
        participants=tuple(n for n in names if n in sending),
        clip_factor=clip_factor,
    )


def _finite(x: float) -> float | None:
    """NaN and infinity become ``null``.

    The payload is injected as a JavaScript object literal, where a bare ``NaN``
    happens to parse because it is a global -- but it would stop parsing the
    moment anything ran it through ``JSON.parse``, and an aggregate shift is
    NaN exactly when no aggregate formed, which is a case the scene has to read.
    """
    v = float(x)
    return v if np.isfinite(v) else None


def participation_mask(keys: Iterable[str], sending: Iterable[str]) -> str:
    """A subset of clients as a bit string in device order, e.g. ``"1011"``.

    The browser builds the same string from its own toggle state and uses it as
    a dictionary key, so the two sides agree on the ordering by construction.
    """
    s = set(sending)
    return "".join("1" if k in s else "0" for k in keys)


def _participation_entry(graph: FederationGraph) -> dict:
    return {
        "clusters": {
            c: {
                "aggregated": bool(d.aggregated),
                "effectivePeers": _finite(d.effective_peers),
                "shift": _finite(d.aggregate_shift),
                "clipped": int(d.clipped),
            }
            for c, d in graph.diagnostics.items()
        },
        "devices": {
            dev.key: {
                c: {
                    "weight": float(link.weight),
                    "admissible": bool(link.admissible),
                    "excluded": bool(link.excluded),
                }
                for c, link in dev.links.items()
            }
            for dev in graph.devices
        },
    }


def participation_sweep(graph: FederationGraph) -> dict[str, dict]:
    """Every possible set of sending clients, with the weights it really gives.

    Clicking a station in the scene has to change the arithmetic, not just stop
    an animation. The alternative -- freezing the packets while the weight
    labels keep their old values -- would draw a client that is visibly silent
    and still counted, which is precisely the kind of quietly wrong picture
    this module exists to avoid.

    A recomputation needs Python, and the scene is an iframe with no way to
    call back into Streamlit without a full rerun. With four devices there are
    only sixteen subsets and one round costs about 3 ms, so the whole space is
    simply precomputed here and the browser looks the answer up. Every number
    the scene can display therefore still comes out of ``FedBuffServer``.

    Keyed by :func:`participation_mask` over ``[d.key for d in graph.devices]``.
    """
    keys = [d.key for d in graph.devices]
    kw = dict(
        target=graph.target,
        bandwidth=graph.bandwidth,
        use_similarity=graph.use_similarity,
        include_self=graph.include_self,
        use_safety=graph.use_safety,
        use_sample_count=graph.use_sample_count,
        rule=graph.rule,
        align=graph.align,
        clip_factor=graph.clip_factor,
        current_round=graph.current_round,
        round_age={d.key: d.round_age for d in graph.devices},
        config_epoch_lag={d.key: d.config_epoch_lag for d in graph.devices},
        violation_rate={d.key: d.violation_rate for d in graph.devices},
        n_samples={d.key: d.n_samples for d in graph.devices},
    )
    out: dict[str, dict] = {}
    for r in range(len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            out[participation_mask(keys, combo)] = _participation_entry(
                build_graph(participants=combo, **kw)
            )
    return out


def scene_payload(graph: FederationGraph, *, participation: dict | None = None) -> dict:
    """The graph, flattened into the JSON the Three.js scene consumes."""
    devices = []
    for d in graph.devices:
        site = SITES.get(
            d.key, {"lab": "", "place": "", "lat": 0.0, "lon": 0.0, "offset_deg": (0.0, 0.0)}
        )
        off = site["offset_deg"]
        devices.append(
            {
                "key": d.key,
                "label": d.label,
                "badge": d.badge,
                "color": d.color,
                # True coordinates, plus the purely cartographic nudge that
                # stops Cadarache and Lausanne overlapping at world scale.
                "lat": float(site["lat"]),
                "lon": float(site["lon"]),
                "offsetLon": float(off[0]),
                "offsetLat": float(off[1]),
                "lab": site["lab"],
                "place": site["place"],
                "scale": d.scale,
                # The real shaping, so each station is drawn as the machine it
                # is rather than a generic ring: the digital-twin coil profile
                # is a function of inverse aspect ratio and elongation.
                "epsilon": float(d.a_minor / d.R_major),
                "distance": d.distance,
                "isTarget": d.key == graph.target,
                "R": d.R_major,
                "a": d.a_minor,
                "B0": d.B_0,
                "kappa": d.elongation,
                "Ip": d.Ip_MA,
                "state": d.state,
                "roundAge": d.round_age,
                "violationRate": d.violation_rate,
                "coils": [
                    {
                        "cluster": c,
                        "label": CLUSTER_META[c]["label"],
                        "color": CLUSTER_META[c]["color"],
                        "weight": d.links[c].weight,
                        "kernel": d.links[c].kernel,
                        "staleness": d.links[c].staleness,
                        "admissible": d.links[c].admissible,
                        "excluded": d.links[c].excluded,
                        "isSelf": d.links[c].is_self,
                        "agents": [
                            {
                                "name": a.name,
                                "available": a.available,
                                "hi": a.hi,
                                "units": a.units,
                            }
                            for a in d.links[c].agents
                        ],
                    }
                    for c in CLUSTERS
                    if c in d.links
                ],
            }
        )
    keys = [d.key for d in graph.devices]
    return {
        "target": graph.target,
        "bandwidth": graph.bandwidth,
        "useSimilarity": graph.use_similarity,
        "rule": graph.rule,
        "round": graph.current_round,
        "clusters": [
            {
                "key": c,
                "label": CLUSTER_META[c]["label"],
                "color": CLUSTER_META[c]["color"],
                "payloadBytes": (
                    graph.diagnostics[c].payload_bytes if c in graph.diagnostics else 0
                ),
                "aggregated": (
                    graph.diagnostics[c].aggregated if c in graph.diagnostics else False
                ),
            }
            for c in CLUSTERS
        ],
        "devices": devices,
        "server": {"lat": SERVER_SITE["lat"], "lon": SERVER_SITE["lon"]},
        # Which clients the scene starts with switched on, and the whole
        # precomputed subset space it can move through. Empty by default: the
        # scene opens with every client silent, so the first thing a viewer
        # does is choose who takes part.
        "sending": [],
        "deviceOrder": keys,
        "participation": participation if participation is not None else participation_sweep(graph),
    }
