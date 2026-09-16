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
| [SCRIPT.md](SCRIPT.md) | Spoken LLM-agent script, source links and 3D demo cues |
| `visuals/atlas.html` | Geographic federation atlas exported from the existing `viz/` code |
| `visuals/diiid_like.html` | Existing registry-based reactor assembly, copied for portable presentation use |
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

Block 1 contains the LLM-agent story developed with the user. Block 2 currently
introduces TORAX, the atlas and the experimental question. Detailed cold-start
results and the full block-2 narration require their own evidence review;
animation is not evidence that federation improves learning.
