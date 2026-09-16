# Visual & 3D CAD Architecture Documentation

**Project:** `tokamak-flower-berlin-2026`  
**Hackathon:** Flower Hackathon Berlin 2026 — Heterogeneous Federated Multi-Agent RL for Tokamak Control  
**File:** `visual-additions.md`  
**Author:** Antigravity AI Assistant & Tokamak Federation Engineering Team  

---

## Executive Summary

This document details the complete 3D visual and CAD architecture implemented for the project. The application features interactive WebGL 3D models of 4 heterogeneous research tokamaks with procedural reactor hardware (vacuum vessel cutaways, D-shaped Toroidal Field coils, Central Solenoid), concentric alignment without clipping, and direct ISO-10303-21 STEP CAD export for engineering workflows.

---

## Part 1: Visual & Frontend Systems

### 1.1 Individual Tokamak 3D Reactor Assemblies
* **Files:**
  * [`assets/3d/html/iter_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/iter_like.html)
  * [`assets/3d/html/sparc_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/sparc_like.html)
  * [`assets/3d/html/diiid_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/diiid_like.html)
  * [`assets/3d/html/tcv_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/tcv_like.html)
* **Components Modeled per Tokamak:**
  1. **Internal Plasma Core (STL Mesh):** High-resolution tessellated D-shaped elongated torus exported from CadQuery/Paramak, decoded via `THREE.STLLoader`. Features persistent machine-specific emission glow and dynamic breathing pulse shader (`Math.sin(t) * 0.15 + 0.85`).
  2. **Vacuum Vessel Outer Cutaway Shell:** A $180^\circ$ semi-circular toroidal cutaway shell snugly covering the back sector of the plasma and leaving the front open. Styled with semi-transparent brushed stainless steel (`metalness: 0.88, roughness: 0.22, opacity: 0.32`).
  3. **Toroidal Field (TF) Magnet Coils:** 16 discrete D-shaped planar coils encircling the vacuum vessel with widened outer clearance ($R_\text{out} \approx 1.42 (R_0 + a)$) to ensure **zero clipping** through the vessel outside wall. Rendered in burnished copper/bronze (`color: 0xcd7f32, metalness: 0.85`).
  4. **Central Solenoid (CS):** Massive central cylindrical magnet column at $R=0$ standing inside the torus donut hole, detailed with 7 discrete copper winding rings (`color: 0xb87333`).
* **Concentric Alignment & Origin Fix:**
  * Eliminated bounding-box centering (`geometry.center()`). Raw CadQuery/Paramak STL coordinate origins coincide with the machine axis $(0, 0, 0)$, ensuring zero clipping and concentric alignment between Central Solenoid, TF coils, vacuum vessel, and the D-shaped plasma ring.
* **Viewport Controls & Non-Obtrusive Layout:**
  * **Zero Canvas Obstruction:** The client statistics HUD overlay was moved completely outside the WebGL canvas into the right-hand dashboard sidebar (`col_info`), leaving the 3D viewport 100% unobstructed.
  * **Auto-Rotation Off by Default:** Viewports start with auto-rotation disabled by default (`autoRotate = false`), allowing immediate inspection without camera motion.
  * **Layer Toggle Controls:** Floating bottom-right toggle bar:
    * `Plasma` (show/hide plasma core)
    * `Vessel` (show/hide vacuum vessel cutaway)
    * `TF Coils` (show/hide magnetic field coils)
    * `Solenoid` (show/hide central solenoid column)
    * `Wireframe` (toggle structural mesh wireframe)
    * `Toggle Rotation` & `Reset View`

### 1.2 Streamlit 3D Dashboard & CAD Export UI
* **File:** [`app.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/app.py)
* **Architecture:**
  * Focused exclusively on the **Individual 3D Tokamak Digital Twin**.
  * Sidebar tokamak selector (ITER-like, SPARC-like, DIII-D-like, TCV-like).
  * Responsive 2-column layout:
    * **Left (2.4 width):** Full-height 3D WebGL interactive canvas embedding the selected tokamak reactor assembly.
    * **Right (1.1 width):**
      - Reactor engineering parameters card ($R_0, a, A, \kappa, B_0, I_p, P_\text{aux}$).
      - **CAD Model Export Block:** Direct one-click download buttons for standard **ISO-10303-21 STEP (.step)** and tessellated mesh **STL (.stl)** files. Compatible with SolidWorks, FreeCAD, Autodesk Fusion, Siemens NX, and Blender.

---

## Part 2: CAD Export Architecture

* **Files:**
  * Converter: [`scripts/export_stl_to_step.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/scripts/export_stl_to_step.py)
  * Generated CAD Models:
    * [`assets/3d/models/iter_like.step`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/iter_like.step) (~524 KB)
    * [`assets/3d/models/sparc_like.step`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/sparc_like.step) (~565 KB)
    * [`assets/3d/models/diiid_like.step`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/diiid_like.step) (~596 KB)
    * [`assets/3d/models/tcv_like.step`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/tcv_like.step) (~510 KB)
* **STEP Protocol Standard:**
  * Fully compliant with ISO-10303-21 (AP203 `CONFIG_CONTROL_DESIGN`).
  * Employs `FACETED_BREP` and `CLOSED_SHELL` entity topology linking unique `CARTESIAN_POINT` arrays into planar faceted faces with defined orientation bounds.
  * Allows CAD/CAM engineers to import the exact tokamak geometry into standard parametric CAD software without third-party converters.

---

## Part 3: Cross-Coupling Impact Matrix

| Parameter | Code Location | Direct 3D Visual & CAD Coupling | Action Required if Changed |
| :--- | :--- | :--- | :--- |
| **Major Radius $R_0$** | `registry.py: Device.R_major` | 3D STL mesh toroidal scaling, camera distance, grid floor size, Central Solenoid placement radius | Re-run `python scripts/generate_3d_models.py --html-only` to update camera and viewport bounds. |
| **Minor Radius $a$** | `registry.py: Device.a_minor` | 3D STL minor radius, TF coil Bézier loops, vacuum vessel minor diameter, grid floor height | Re-run `generate_3d_models.py`. Ensure vacuum vessel scale doesn't clip the plasma surface. |
| **Elongation $\kappa$** | `registry.py: Device.elongation` | 3D STL mesh vertical elongation, vacuum vessel vertical scale (`vesselMesh.scale.z = kappa`), Central Solenoid height | Re-run `generate_3d_models.py`. Check vertical clearance of TF coils. |
| **Triangularity $\delta$** | `registry.py: Device.triangularity` | CAD profile sharpness in Paramak, outer contour of STL and STEP mesh | Requires full CAD re-meshing (`python scripts/generate_3d_models.py` with CadQuery/Paramak). |
| **Toroidal Field $B_0$** | `registry.py: Device.B_0` | Stat card badge "Toroidal Field B0 [T]" | Automatically formatted in `app.py`. |
| **Plasma Current $I_p$** | `registry.py: Device.Ip_nominal` | Stat card badge "Plasma Current Ip [MA]" | Automatically formatted in `app.py`. |

---

## Part 4: Complete File Inventory

| Category | Relative Path | Purpose |
| :--- | :--- | :--- |
| **Frontend** | [`app.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/app.py) | Streamlit individual 3D tokamak visualizer & STEP export app |
| **3D Viewport** | [`assets/3d/html/iter_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/iter_like.html) | ITER reactor viewport (plasma, vessel, TF coils, solenoid) |
| **3D Viewport** | [`assets/3d/html/sparc_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/sparc_like.html) | SPARC reactor viewport (plasma, vessel, TF coils, solenoid) |
| **3D Viewport** | [`assets/3d/html/diiid_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/diiid_like.html) | DIII-D reactor viewport (plasma, vessel, TF coils, solenoid) |
| **3D Viewport** | [`assets/3d/html/tcv_like.html`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/html/tcv_like.html) | TCV reactor viewport (plasma, vessel, TF coils, solenoid) |
| **CAD Models** | [`assets/3d/models/*.step`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/) | ISO-10303-21 STEP CAD models (ITER, SPARC, DIII-D, TCV) |
| **Mesh Models** | [`assets/3d/models/*.stl`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/assets/3d/models/) | Tessellated STL binary meshes |
| **CAD Exporter** | [`scripts/export_stl_to_step.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/scripts/export_stl_to_step.py) | Pure Python STL to ISO-10303-21 STEP converter |
| **3D Generator** | [`scripts/generate_3d_models.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/scripts/generate_3d_models.py) | Parametric CAD & Three.js HTML viewport builder |
| **Deploy Tool** | [`scripts/deploy_to_remote.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/scripts/deploy_to_remote.py) | Compressed remote asset deployment and Streamlit daemon manager |

---

## Part 5: Restarting Streamlit on the Remote Host

To restart the Streamlit daemon on the remote server:
```bash
# 1. Kill any existing streamlit instances
pkill -9 -f "streamlit run"

# 2. Launch Streamlit in the background
nohup /home/dmitry/tokamak-flower-berlin-2026/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > /home/dmitry/tokamak-flower-berlin-2026/streamlit.log 2>&1 &
```
