"""Streamlit front end for tokamak-flower-berlin-2026.

Two views over the same four devices:

  * **Federation (3D)** -- how the agents work together. Four device stations,
    digital-twin TF coils with three channel colors, agent boxes on the coils,
    and role-matched links into a three-channel aggregation server. Every
    weight in the picture is computed by the repository's own
    ``FedBuffServer``, not by the visualiser.
  * **Digital twin (3D + CAD)** -- one device at a time, with the procedural
    reactor assembly and ISO-10303-21 STEP / STL export.

The federation view is the one that shows the claim; the digital twin view is
the one that shows the machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HTML_DIR = PROJECT_ROOT / "assets" / "3d" / "html"
MODELS_DIR = PROJECT_ROOT / "assets" / "3d" / "models"

st.set_page_config(
    page_title="Tokamak Federation -- 3D Digital Twin & Agent Federation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Device Metadata (digital-twin view). Machine numbers mirror
# hfmarl/devices/registry.py; P_aux names the physical system each device's
# `aux_heat` channel actually is -- ICRF on SPARC, NBI everywhere else.
DEVICES_INFO = {
    "iter_like": {
        "name": "ITER-like",
        "badge": "Giant / Low-ρ*",
        "r0": 6.20,
        "a": 2.00,
        "aspect_ratio": 3.10,
        "elongation": 1.72,
        "b0": 5.3,
        "ip": 10.5,
        "p_aux": "33 MW (NBI) + 20 MW (ECRH)",
        "role": "Large, low-rho_star, low-nu_star burning plasma regime.",
        "html_file": "iter_like.html",
        "stl_file": "iter_like.stl",
        "step_file": "iter_like.step",
    },
    "sparc_like": {
        "name": "SPARC-like",
        "badge": "Ultra-Compact / High-Field",
        "r0": 1.85,
        "a": 0.57,
        "aspect_ratio": 3.25,
        "elongation": 1.97,
        "b0": 12.2,
        "ip": 8.7,
        "p_aux": "25 MW (ICRF) + 5 MW (ECRH)",
        "role": "High-field HTS tokamak reaching burning plasma dimensionless physics at 1/3 scale.",
        "html_file": "sparc_like.html",
        "stl_file": "sparc_like.stl",
        "step_file": "sparc_like.step",
    },
    "diiid_like": {
        "name": "DIII-D-like",
        "badge": "Medium / Low Aspect Ratio",
        "r0": 1.67,
        "a": 0.67,
        "aspect_ratio": 2.49,
        "elongation": 1.80,
        "b0": 2.0,
        "ip": 1.5,
        "p_aux": "20 MW (NBI) + 6 MW (ECRH)",
        "role": "PACMAN benchmark reference device for RL tokamak control.",
        "html_file": "diiid_like.html",
        "stl_file": "diiid_like.stl",
        "step_file": "diiid_like.step",
    },
    "tcv_like": {
        "name": "TCV-like",
        "badge": "Small / Highly Shaped",
        "r0": 0.88,
        "a": 0.25,
        "aspect_ratio": 3.52,
        "elongation": 1.50,
        "b0": 1.44,
        "ip": 0.25,
        "p_aux": "1.3 MW (NBI) + 4.5 MW (ECRH)",
        "role": "EPFL research tokamak, highly shaped, fast ECRH response.",
        "html_file": "tcv_like.html",
        "stl_file": "tcv_like.stl",
        "step_file": "tcv_like.step",
    },
}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #8b949e;
        margin-bottom: 20px;
    }
    .spec-card {
        background: rgba(22, 27, 34, 0.95);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    .cad-box {
        margin-top: 18px;
        padding: 16px;
        border-radius: 8px;
        background: rgba(31, 111, 235, 0.08);
        border: 1px solid rgba(56, 139, 253, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# View: federation
# ---------------------------------------------------------------------------


def render_federation() -> None:
    import pandas as pd

    from viz import federation_data as fd
    from viz.federation_scene import build_scene_html

    device_keys = list(fd.DEVICE_META)

    with st.sidebar:
        st.markdown("#### **Aggregation target**")
        target = st.selectbox(
            "Device receiving the personalised aggregate",
            options=device_keys,
            format_func=lambda k: fd.DEVICE_META[k]["label"],
            index=0,
            help=(
                "Weighting is computed relative to where THIS device operates, "
                "so ITER and TCV receive different mixtures of the same buffer. "
                "There is no single global model."
            ),
        )

        st.markdown("#### **Similarity kernel**")
        # The median pairwise distance over the device set -- computed, not
        # chosen, which is what makes the default defensible.
        suggested = fd.median_bandwidth()
        auto_bw = st.checkbox(
            f"Use the median-heuristic bandwidth ({suggested:.3f})", value=True
        )
        # The default is clamped into the slider's range rather than assumed to
        # fall inside it: the median heuristic is computed from the device set,
        # and a device set with more spread would push it past 4.0 and raise.
        bw_default = float(min(max(round(suggested, 2), 0.10), 4.00))
        bandwidth = (
            suggested
            if auto_bw
            else st.slider("Bandwidth h", 0.10, 4.00, bw_default, 0.05)
        )
        use_similarity = st.checkbox(
            "Similarity weighting (method)",
            value=True,
            help="Off reproduces baseline 2: uniform, role-matched federation.",
        )
        use_safety = st.checkbox(
            "Physics + safety gates",
            value=True,
            help=(
                "Regime validity (nu* < 1) and violation-rate admissibility, "
                "plus the graded safety down-weight."
            ),
        )
        include_self = st.checkbox("Include the target's own update", value=True)
        use_sample_count = st.checkbox(
            "Also weight by sample count (FedAvg)",
            value=False,
            help="Off by default: sample count is the arbitrary rule being replaced.",
        )

        st.markdown("#### **Aggregation rule**")
        rule = st.radio(
            "Rule",
            options=["geomedian", "mean", "median"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        align = st.checkbox("Align hidden units before merging", value=True)
        clip_on = st.checkbox("Centred clipping (factor 2.0)", value=True)

        with st.expander("Per-device update state"):
            st.caption(
                "Staleness and safety are what turn a near peer into a weak one. "
                "Push a device past a 50% violation rate and the admissibility "
                "gate refuses it outright -- the link goes grey in the scene."
            )
            current_round = st.slider("Current server round", 1, 60, 12)
            round_age: dict[str, int] = {}
            violation_rate: dict[str, float] = {}
            config_lag: dict[str, int] = {}
            # Two per row, not three: the sidebar is ~330px wide and three
            # number inputs across it wrap their steppers into unusable stubs.
            for k in device_keys:
                st.markdown(f"**{fd.DEVICE_META[k]['label']}**")
                c1, c2 = st.columns(2)
                round_age[k] = c1.number_input(
                    "update age", 0, 50, 0, key=f"age_{k}",
                    help="rounds elapsed since this update was produced",
                )
                violation_rate[k] = c2.number_input(
                    "violation rate", 0.0, 1.0, 0.0, 0.05, key=f"viol_{k}",
                    help="fraction of its shots that crossed a limit; >0.5 is refused",
                )
                config_lag[k] = st.number_input(
                    "config epochs behind", 0, 5, 0, key=f"epoch_{k}",
                    help=(
                        "Wall, divertor or heating changes since. Each epoch "
                        "halves the weight -- staleness that wall-clock time "
                        "cannot express."
                    ),
                )

    graph = fd.build_graph(
        target=target,
        bandwidth=bandwidth,
        use_similarity=use_similarity,
        include_self=include_self,
        use_safety=use_safety,
        use_sample_count=use_sample_count,
        rule=rule,
        align=align,
        clip_factor=2.0 if clip_on else None,
        current_round=current_round,
        round_age=round_age,
        config_epoch_lag=config_lag,
        violation_rate=violation_rate,
    )

    st.markdown(
        '<div class="main-title">A shared policy across four tokamaks</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Select machines to watch their policies flow through '
        "the federation. Explore each contribution, channel by channel.</div>",
        unsafe_allow_html=True,
    )

    components.html(
        build_scene_html(fd.scene_payload(graph, participation=fd.participation_sweep(graph))),
        height=720,
        scrolling=False,
    )
    st.caption(
        "The scene starts with all senders off. Click models to recompute the scene's "
        "weights and effective peers. The tables and metrics below describe the "
        "all-client reference round; sidebar changes reset scene selection. "
        "Site pins show host labs; displaced models have leader lines. "
        "The server's Greenland position is schematic."
    )

    # ---- headline numbers -------------------------------------------------
    total_payload = sum(d.payload_bytes for d in graph.diagnostics.values())
    thermal = graph.diagnostics.get("thermal")
    # Refused is not the same as absent: a self-link that was never buffered
    # (include_self off) is a configuration choice, not the gate firing.
    n_refused = sum(
        1
        for d in graph.devices
        for link in d.links.values()
        if not link.admissible and not link.excluded
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Payload per client per round",
        f"{total_payload} B",
        help="All three channels, float32 on the wire. Measured from the real shapes.",
    )
    m2.metric(
        "Effective peers (thermal)",
        f"{thermal.effective_peers:.2f}" if thermal else "--",
        help=(
            "1 / sum(w^2). Equals the client count when the weighting is uniform "
            "and falls toward 1 as one device dominates."
        ),
    )
    m3.metric(
        "Bandwidth h",
        f"{graph.bandwidth:.3f}",
        delta=f"{graph.bandwidth - graph.suggested_bandwidth:+.3f} vs median heuristic",
        delta_color="off",
    )
    m4.metric(
        "Links refused by the gate",
        n_refused,
        help="Admissibility is asked before weighting, and answered separately.",
    )

    # ---- warnings the scene would otherwise only imply --------------------
    off_diag = [
        link.kernel
        for d in graph.devices
        if d.key != graph.target
        for link in list(d.links.values())[:1]
    ]
    if use_similarity and off_diag:
        if max(off_diag) < 0.05:
            st.error(
                "Every peer is effectively ignored at this bandwidth. Federation "
                "does nothing and baseline 3 collapses onto baseline 1 -- one of "
                "the two silent failure modes `describe_device_set` warns about."
            )
        elif min(off_diag) > 0.90:
            st.warning(
                "All peers are weighted near-equally. Baseline 3 collapses onto "
                "baseline 2 and the method's contribution becomes untestable. "
                "Decrease the bandwidth."
            )
    for cluster, diag in graph.diagnostics.items():
        if diag.error:
            st.error(f"**{cluster}**: {diag.error}")
        elif diag.rejected_all:
            st.warning(
                f"**{cluster}**: every buffered peer was disqualified, so no "
                "aggregate was formed. That is the correct answer, not a bug -- "
                "averaging them anyway is what the rejection exists to prevent."
            )
        elif diag.uniform_fallback and use_similarity:
            st.warning(
                f"**{cluster}**: every similarity kernel underflowed and the "
                "weights fell back to uniform over the admissible peers. The "
                "similarity arm has just become the uniform arm."
            )

    # ---- tables -----------------------------------------------------------
    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown("#### Aggregation weights into **%s**" % fd.DEVICE_META[target]["label"])
        rows = []
        for d in graph.devices:
            row = {"device": d.label, "distance": d.distance}
            for c in fd.CLUSTERS:
                link = d.links.get(c)
                row[c] = link.weight if link else float("nan")
            row["staleness"] = (
                list(d.links.values())[0].staleness if d.links else float("nan")
            )
            rows.append(row)
        df = pd.DataFrame(rows).set_index("device")
        st.dataframe(
            df.style.format(
                {
                    "distance": "{:.3f}",
                    "staleness": "{:.3f}",
                    **{c: "{:.1%}" for c in fd.CLUSTERS},
                },
                na_rep="--",
            ).background_gradient(cmap="viridis", subset=list(fd.CLUSTERS)),
            width="stretch",
        )
        st.caption(
            "`distance` is the weighted Euclidean distance in (rho*, nu*, beta_N, "
            "q95) -- log-scaled where the parameter spans decades. `staleness` is "
            "the FedBuff factor with the physics-informed config-epoch penalty on "
            "top. The columns are the normalised weights the server actually used."
        )

    with right:
        st.markdown("#### Pairwise dimensionless distance")
        dm = pd.DataFrame(graph.distance_matrix)
        dm.index = [fd.DEVICE_META[k]["label"] for k in dm.index]
        dm.columns = [fd.DEVICE_META[k]["label"] for k in dm.columns]
        st.dataframe(
            dm.style.format("{:.3f}").background_gradient(cmap="magma_r"),
            width="stretch",
        )
        st.caption(
            "SPARC-like sits adjacent to ITER-like despite being a third the "
            "size, because its field is 2.3x higher. That is the Connor-Taylor "
            "overlap Phase 6 needs, and it is physics rather than an artefact of "
            "how the configs were chosen."
        )

    # ---- per-channel diagnostics -----------------------------------------
    st.markdown("#### What the server did, per channel")
    cols = st.columns(len(fd.CLUSTERS))
    for col, cluster in zip(cols, fd.CLUSTERS):
        diag = graph.diagnostics.get(cluster)
        meta = fd.CLUSTER_META[cluster]
        with col:
            st.markdown(
                f"<div class='spec-card' style='padding:14px;'>"
                f"<div style='color:{meta['color']};font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.6px;font-size:0.78rem;'>"
                f"{meta['label']}</div>"
                f"<div style='font-size:0.74rem;color:#8b949e;margin:6px 0 10px 0;'>"
                f"{meta['blurb']}</div>"
                + (
                    f"<div style='font-size:0.78rem;line-height:1.7;'>"
                    f"live agents <b>{diag.act_dim}</b><br>"
                    f"parameters <b>{diag.n_params}</b> "
                    f"(<b>{diag.payload_bytes} B</b> float32)<br>"
                    f"aggregate formed <b>{'yes' if diag.aggregated else 'no'}</b><br>"
                    f"clients clipped <b>{diag.clipped}</b><br>"
                    f"effective peers <b>{diag.effective_peers:.2f}</b><br>"
                    f"||aggregate - own|| <b>"
                    f"{diag.aggregate_shift:.3f}</b>"
                    f"</div>"
                    if diag
                    else "<div style='color:#f85149;'>no agents in this channel</div>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    with st.expander("How to read the scene, and what it is not claiming"):
        st.markdown(
            """
**Each station is one device.** Its size is `R_major^0.45` -- the ordering is
true, the ratio is compressed so TCV-like stays visible next to ITER-like. The
real metres are in the hover card and in the digital-twin view.

**Sixteen D-shaped TF coils use the digital-twin profile.** Their three colors
represent the logical federation channels (`Actuator.cluster`): `thermal`,
`particle`, `current`; they do not claim a physical magnet-to-actuator mapping.
A box riding on a coil is one actuator agent. The agents connect to a shared
bus because the team is **flat** -- the three-level hierarchy
(per-actuator agents under cluster heads under a device head) was removed on
2026-09-15, and the `H` in `hfmarl` is now historical.

**`icrh` is drawn hollow and dark** on every thermal coil. TORAX 1.4.3 has two
independently drivable auxiliary heat sources, not three, so ICRH is defined
and marked `available=False` rather than quietly aliased onto `generic_heat` --
which would make two agents write the same config path and turn a controller
bug into something that looks like a physics result.

**The three hub rings are not joined to each other.** Role matching means
thermal never aggregates with particle or current; `aggregation_weights` raises
if it is handed mixed clusters. A column joining them would draw exactly the
coupling the design forbids.

**A grey, broken link was refused, not down-weighted.** Admissibility and
weighting are two different questions asked in order: may this update be
aggregated at all, and if so how much should it count. Collapsing them would
let a peer whose physics disqualifies it still contribute, just less.

---

**What this view is not.** It is not a training run. The policy *weights* in
the buffer are seeded stand-ins; what is real is every shape, distance, kernel,
staleness factor, admissibility decision and normalised weight, all computed by
`hfmarl.federation`. And the operating points behind the distances are still
`OPERATING_POINTS` -- the hand-written nominal table, not TORAX output. That
substitution is on the RUNBOOK's "Before Phase 5" list and has not been done,
so the distances are between tabulated points, not measured operating regions.
            """
        )


# ---------------------------------------------------------------------------
# View: digital twin
# ---------------------------------------------------------------------------


def render_digital_twin() -> None:
    with st.sidebar:
        selected_key = st.selectbox(
            "Select Tokamak Device:",
            options=list(DEVICES_INFO.keys()),
            format_func=lambda k: f"{DEVICES_INFO[k]['name']} ({DEVICES_INFO[k]['badge']})",
            index=0,
        )

        st.markdown("---")
        st.markdown("#### **Available Tokamak Models**")
        for k, d in DEVICES_INFO.items():
            prefix = "\U0001f449 **" if k == selected_key else "- "
            suffix = "**" if k == selected_key else ""
            st.markdown(f"{prefix}{d['name']} ({d['badge']}){suffix}")

    dev = DEVICES_INFO[selected_key]

    st.markdown(
        f'<div class="main-title">⚛️ {dev["name"]} — 3D Reactor Model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Interactive 3D CAD visualization with concentric '
        "magnetic coils, vacuum vessel, and ISO-10303-21 STEP export.</div>",
        unsafe_allow_html=True,
    )

    col_3d, col_info = st.columns([2.4, 1.1])

    with col_3d:
        html_path = HTML_DIR / dev["html_file"]
        if html_path.exists():
            components.html(html_path.read_text(encoding="utf-8"), height=650, scrolling=False)
        else:
            st.error(f"3D view not found at: {html_path}")

    with col_info:
        st.markdown(
            f"""
            <div class="spec-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <h3 style="margin: 0; font-size: 1.3rem; color: #f0f6fc;">{dev['name']}</h3>
                    <span style="background: rgba(88, 166, 255, 0.15); border: 1px solid #58a6ff; color: #58a6ff; font-size: 0.70rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Tokamak</span>
                </div>
                <div style="display: inline-block; background: rgba(139, 148, 158, 0.15); border: 1px solid #30363d; color: #8b949e; font-size: 0.72rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-bottom: 10px;">{dev['badge']}</div>
                <p style="font-size: 0.80rem; color: #8b949e; line-height: 1.4; margin-bottom: 14px;">{dev['role']}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; border-top: 1px solid #30363d; padding-top: 12px;">
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Major Radius R₀</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #58a6ff;">{dev['r0']:.2f} m</div>
                    </div>
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Minor Radius a</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{dev['a']:.2f} m</div>
                    </div>
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Aspect Ratio A</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{dev['aspect_ratio']:.2f}</div>
                    </div>
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Elongation κ</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{dev['elongation']:.2f}</div>
                    </div>
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Toroidal Field B₀</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #58a6ff;">{dev['b0']:.1f} T</div>
                    </div>
                    <div style="background: rgba(13, 17, 23, 0.6); padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5);">
                        <div style="font-size: 0.68rem; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px;">Plasma Current Iₚ</div>
                        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{dev['ip']:.2f} MA</div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.78rem; color: #8b949e; line-height: 1.4;">
                    <b>Aux Heating:</b> {dev['p_aux']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### **\U0001f4e5 Export CAD Model**")
        step_path = MODELS_DIR / dev["step_file"]
        stl_path = MODELS_DIR / dev["stl_file"]

        if step_path.exists():
            st.download_button(
                label=f"\U0001f4be Download {dev['name']} (.step)",
                data=step_path.read_bytes(),
                file_name=f"{selected_key}.step",
                mime="application/step",
                width="stretch",
            )
        else:
            st.warning(f"STEP file not found: {step_path.name}")

        if stl_path.exists():
            st.download_button(
                label="\U0001f4e6 Download Mesh (.stl)",
                data=stl_path.read_bytes(),
                file_name=f"{selected_key}.stl",
                mime="model/stl",
                width="stretch",
            )

        st.caption(
            "Standard ISO-10303-21 STEP (AP203 Faceted B-Rep) format, compatible "
            "with SolidWorks, FreeCAD, Autodesk Fusion, Siemens NX, and Blender."
        )


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### **Tokamak Federation**")
    st.markdown("`tokamak-flower-berlin-2026`")
    st.markdown("---")
    view = st.radio(
        "View",
        options=["Federation (3D)", "Digital twin (3D + CAD)"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

if view.startswith("Federation"):
    render_federation()
else:
    render_digital_twin()
