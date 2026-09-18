# Fusion presentation

This folder is the **presentation**, with a browser deck, speaker script and
interactive 3D visualizations. The working research application is separate in
[`../flower-app/`](../flower-app/README.md). Its controls are not slides.

## Open the presentation

From the repository root:

```powershell
python presentation/serve.py
```

Open <http://127.0.0.1:8788/>. The server uses only the Python standard library.
It serves a fixed set of presentation assets and has no Flower or experiment API.
Use the arrow buttons or left/right keys to navigate. The deck can also be opened
as `index.html` directly. The 3D views require WebGL and internet access to load
their pinned Three.js dependencies.

The application runs independently on <http://127.0.0.1:8787/>. Start it with the
[application instructions](../flower-app/README.md#local-frontend). The deck's
**Open research app** link opens it in another tab. Nothing in the deck submits
a hosted run, resets a ledger or starts an experiment.

## Files and provenance

| File | Purpose |
|---|---|
| `index.html`, `style.css`, `deck.js` | Browser presentation, two clearly labelled blocks |
| [SCRIPT.md](SCRIPT.md) | Seven-slide speaker script, source links and demo cues |
| `visuals/atlas.html` | Geographic federation atlas exported from the existing `viz/` code |
| `visuals/diiid_like.html` | Existing registry-based reactor assembly, copied for portable presentation use |
| `visuals/marl-architecture.png` | User-selected `results/audit/system_design.png`, preserved as an earlier audit snapshot |
| `visuals/flower-agent-architecture.svg` | Editable diagram of the current Flower investigation workflow |
| `build_visuals.py` | Rebuild those exports using the repository's NumPy environment |
| `serve.py` | Independent read-only presentation server |

Regenerate visuals from the repository root with its research environment:

```powershell
.venv/Scripts/python.exe presentation/build_visuals.py
```

The atlas computes all 16 participation combinations through the existing
`FedBuffServer` implementation. It uses synthetic policy updates and nominal
operating points, **not a live training run, measured TORAX trajectories or
connections to the named facilities**. Its thermal, particle and current
channels illustrate control roles, not Flower LLM-agent communication. Click
machines to toggle participation; drag to orbit. The displayed machine sizes
are compressed for visibility, with geographic site pins and display offsets.

The 3D assembly is illustrative geometry. TORAX supplies the separate 1-D
transport experiments; a rendered reactor is not itself a plasma simulator.
The app's default thermal investigation uses a reduced model, explicitly labelled.

## Seven-slide sequence

1. Full-screen tokamak opening, preserved from the previous design.
2. Problem statement: limited local experience, different devices and controlled disclosure.
3. Four existing foundations: Princeton-led RL, PACMAN, cross-device transfer and TORAX.
4. Our Flower LLM agent and TORAX/HFMARL research, with use cases and implementation scope.
5. The interactive geographic map fills the entire slide.
6. Side-by-side architecture comparison: the existing MARL diagram and a new editable Flower-agent diagram. Click either image to inspect it at full resolution.
7. Cold-start results: SPARC-like median comparison and all six paired seed outcomes, with analysis and source-cost qualifications.

`build_results.py PATH_TO_STUDY_DIRECTORY` regenerates the two static SVG graphs
and `visuals/coldstart-results.json` directly from the saved checkpoint and
summary. It excludes policy-selection shots from competence confirmation while
retaining their cost. The plots use the completed study's six matched seeds.

The completed cold-start results are mixed. The current study evaluates thermal
policy search, not a validated end-to-end MARL controller. The map explains the
broader architecture; its animation is not evidence of a learning benefit.
