"""Streamlit 3D Tokamak Visualizer & CAD Exporter.

Project: tokamak-flower-berlin-2026
Interactive 3D reactor models with parametric CAD STEP / STL export:
  1. ITER-like (Giant burning-plasma facility)
  2. SPARC-like (Compact high-field HTS tokamak)
  3. DIII-D-like (Medium aspect ratio benchmark facility)
  4. TCV-like (Highly shaped research tokamak)
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="Tokamak 3D Digital Twin & CAD Export",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
HTML_DIR = PROJECT_ROOT / "assets" / "3d" / "html"
MODELS_DIR = PROJECT_ROOT / "assets" / "3d" / "models"

# Device Metadata
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

# Custom Styling
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

# Sidebar - Device Selector & Machine Overview
with st.sidebar:
    st.image("https://flower.ai/images/flower-logo.svg", width=180)
    st.markdown("### **Tokamak 3D Digital Twin**")
    st.markdown("`tokamak-flower-berlin-2026`")
    st.markdown("---")

    selected_key = st.selectbox(
        "Select Tokamak Device:",
        options=list(DEVICES_INFO.keys()),
        format_func=lambda k: f"{DEVICES_INFO[k]['name']} ({DEVICES_INFO[k]['badge']})",
        index=0,
    )

    st.markdown("---")
    st.markdown("#### **Available Tokamak Models**")
    for k, d in DEVICES_INFO.items():
        prefix = "👉 **" if k == selected_key else "- "
        suffix = "**" if k == selected_key else ""
        st.markdown(f"{prefix}{d['name']} ({d['badge']}){suffix}")

dev = DEVICES_INFO[selected_key]

# Header
st.markdown(f'<div class="main-title">⚛️ {dev["name"]} — 3D Reactor Model</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Interactive 3D CAD visualization with concentric magnetic coils, vacuum vessel, and ISO-10303-21 STEP export.</div>',
    unsafe_allow_html=True,
)

# Main Two-Column Layout
col_3d, col_info = st.columns([2.4, 1.1])

with col_3d:
    html_path = HTML_DIR / dev["html_file"]
    if html_path.exists():
        html_code = html_path.read_text(encoding="utf-8")
        components.html(html_code, height=650, scrolling=False)
    else:
        st.error(f"3D view not found at: {html_path}")

with col_info:
    # Specifications Card
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

    # CAD Export Block
    st.markdown("#### **📥 Export CAD Model**")
    step_path = MODELS_DIR / dev["step_file"]
    stl_path = MODELS_DIR / dev["stl_file"]

    if step_path.exists():
        step_bytes = step_path.read_bytes()
        st.download_button(
            label=f"💾 Download {dev['name']} (.step)",
            data=step_bytes,
            file_name=f"{selected_key}.step",
            mime="application/step",
            use_container_width=True,
        )
    else:
        st.warning(f"STEP file not found: {step_path.name}")

    if stl_path.exists():
        stl_bytes = stl_path.read_bytes()
        st.download_button(
            label=f"📦 Download Mesh (.stl)",
            data=stl_bytes,
            file_name=f"{selected_key}.stl",
            mime="model/stl",
            use_container_width=True,
        )

    st.caption("Standard ISO-10303-21 STEP (AP203 Faceted B-Rep) format, compatible with SolidWorks, FreeCAD, Autodesk Fusion, Siemens NX, and Blender.")
