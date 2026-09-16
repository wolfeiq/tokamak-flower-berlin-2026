# Presentation and application

The presentation lives in [presentation/](presentation/README.md), with a two-block browser deck, [speaker script](presentation/SCRIPT.md), interactive geographic federation atlas and 3D reactor assembly.

The working **Fusion Investigator research app** remains in [flower-app/](flower-app/README.md), with normal page navigation, analysis tools and the hosted Flower investigation controls.

| Open | Command from repository root | URL |
|---|---|---|
| Presentation | `python presentation/serve.py` | <http://127.0.0.1:8788/> |
| Research app | `cd flower-app`, then `uv run python dashboard.py` | <http://127.0.0.1:8787/> |

TORAX control experiments remain in `hfmarl/` and the root scripts. The app's default thermal tools use a reduced model; the presentation identifies the model behind each demonstration. Opening the deck or using its atlas does not start an experiment or hosted run.
