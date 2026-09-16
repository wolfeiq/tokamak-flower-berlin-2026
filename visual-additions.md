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

### 1.2 Federation 3D Scene — how the agents work together
* **Files:**
  * Data layer: [`viz/federation_data.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/viz/federation_data.py)
  * Scene builder: [`viz/federation_scene.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/viz/federation_scene.py)
  * Tests: [`tests/test_federation_view.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/tests/test_federation_view.py)

**The numbers in the picture are computed, not drawn.** Every distance, kernel
value, staleness factor, admissibility decision and normalised weight comes
from running `physics.dimensionless.encode`, `federation.similarity` and
`federation.server.FedBuffServer` — the same modules the experiment runs. Only
the policy *weights* in the buffer are stand-ins, because the app does not
train; the shapes are the real ones, so payload sizes, the alignment
permutation and the clipping radius are all computed on vectors of the size
that would actually be transmitted.

* **Scene vocabulary** (each element is a thing in the repository):

| Element | What it is in the code |
| :--- | :--- |
| Station (4 of them) | One device. Size is $R_0^{0.45}$ — ordering true, ratio compressed so TCV-like stays visible next to ITER-like. |
| **Sixteen D-shaped TF coils** per station | Same profile as the digital twin. Three colors overlay logical `thermal`, `particle`, `current` roles; they do not imply physical magnet wiring. |
| Box on a coil | One actuator agent. `icrh` is drawn hollow and dark because it is `available=False`. |
| Agent connections to a shared bus | The **flat** per-device team. No cluster heads — the three-level hierarchy was removed 2026-09-15. |
| Coloured stream → hub | One role-matched aggregation. Particle density and speed follow the normalised weight `FedBuffServer` assigned. |
| Grey broken link + red label | The admissibility gate **refused** the peer. Distinct from a small weight, because admissibility is a separate question asked first. |
| Absent link | Sender switched off, or not buffered (`include_self=False`). A configuration choice, not a gate firing. |
| Glowing particle streams | Travelling bunches with fading, swirling tails; device → hub density and speed track weight. White streams return to the target only when that channel formed an aggregate. |
| Three hub rings, **not joined** | Role matching. `aggregation_weights` raises on mixed clusters, so no line in the scene may suggest the channels couple. |

* **World map:** simplified vendored Natural Earth coastlines in `viz/world_outline.py`.
  True host-lab pins connect to offset models so nearby European sites stay separate.
  Model size is illustrative; the server's Greenland location is schematic.
  Filled land, subdued graticules and glass panels provide a quieter backdrop.
  Device and hub names use crisp screen-space labels; agent annotations and
  contribution percentages appear on hover or at close zoom. Metallic coils,
  translucent plasma and restrained base rings distinguish the reactor layers.
* **Click to send:** all clients start silent. Click several models to select
  independent senders; click again to stop. Orbit drags do not toggle clients.
  All 16 subsets are precomputed through `FedBuffServer`, so links, tooltips,
  hub state and effective peers update together. Return packets depend on a
  channel having an aggregate, even when its target is silent.
  Animation pause does not change participation. Sidebar changes reset selection;
  tables below the scene show the explicitly labelled all-client reference round.
* **Live controls** (sidebar) — target device, kernel bandwidth (defaulting to
  the median heuristic), similarity on/off (method vs. baseline 2), safety and
  regime gates, self-inclusion, FedAvg sample-count weighting, aggregation rule
  (`geomedian` / `mean` / `median`), hidden-unit alignment, centred clipping,
  and per-device update age / violation rate / config-epoch lag.
* **Failure modes are surfaced, not hidden.** Both silent collapses that
  `describe_device_set` warns about — every peer ignored, or all peers equal —
  raise a banner, as do `uniform_fallback`, `rejected_all` and non-finite
  weights.
* **Standing caveat, shown in the app:** the distances come from
  `OPERATING_POINTS`, still the hand-written nominal table rather than TORAX
  output (README "Still open"). They are distances between tabulated points,
  not between measured operating regions.

### 1.3 Streamlit 3D Dashboard & CAD Export UI
* **File:** [`app.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/app.py)
* **Architecture:**
  * Two views behind a sidebar switch: **Federation (3D)** (default) and
    **Digital twin (3D + CAD)**. The federation view shows the claim; the
    digital-twin view shows the machine.
  * Sidebar tokamak selector (ITER-like, SPARC-like, DIII-D-like, TCV-like).
  * Digital-twin view, responsive 2-column layout:
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
| **Actuator cluster** | `registry.py: Actuator.cluster` | Coil overlay colors, agent boxes and matching server rings | Nothing to re-run — the federation scene reads the registry live. A fourth cluster requires extending the three-channel hub layout. |
| **Actuator availability** | `registry.py: Actuator.available` | Agent box solid vs. hollow; the channel's action dimension and therefore its payload size | Nothing to re-run. If `icrh` ever gains a real TORAX source, the thermal payload stops being 776 B and `test_payload_matches_the_measured_claim` says so. |
| **`OPERATING_POINTS`** | `registry.py` | Every distance, kernel weight, link thickness and packet rate in the federation scene | Nothing to re-run, but re-read the caveat: these are nominal hand-written values, not TORAX output. Replacing them (RUNBOOK "Before Phase 5") moves every link in the picture. |

---

## Part 4: Complete File Inventory

| Category | Relative Path | Purpose |
| :--- | :--- | :--- |
| **Frontend** | [`app.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/app.py) | Streamlit shell: federation 3D view + individual tokamak digital twin & STEP export |
| **Federation data** | [`viz/federation_data.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/viz/federation_data.py) | Runs one personalised aggregation round through the real `FedBuffServer` and reports everything it did |
| **Federation scene** | [`viz/federation_scene.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/viz/federation_scene.py) | Three.js scene: 4 stations x 3 coils, agent boxes, weighted links, 3-channel server |
| **Federation tests** | [`tests/test_federation_view.py`](file:///C:/Users/dima2/IdeaProjects/tokamak-flower-berlin-2026/tests/test_federation_view.py) | Pins the properties a viewer reads off the scene; no TORAX, no GPU |
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
