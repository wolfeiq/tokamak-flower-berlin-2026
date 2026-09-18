# Fusion Investigator + THERMAL — Flower AgentApp

**Separate application — not part of the TORAX/RL experiment pipeline.**
This folder contains the Flower agent, local web frontend, site adapters, and its
own tests and dependency lockfile. The repository's `hfmarl/`, `scripts/`,
`configs/`, and root `tests/` belong to the original control experiments.

**This is the working application, not the presentation.** The separate
[`presentation/`](../presentation/README.md) folder contains the talk, speaker
script and interactive 3D atlas. The application uses normal section navigation
and keeps its investigation tools available without advancing slides.

| Component | Purpose | What it establishes |
|---|---|---|
| This app: `flower-app/` | Between-experiment investigation with Flower AgentApp and a browser frontend | Working orchestration, analysis tools, and synthetic validation workflow |
| Original experiment project: repository root | TORAX simulation and cold-start/control experiments | Simulation results subject to the experimental protocol and caveats |

The app's synthetic heating demo is **not TORAX**, does not train PPO/SAC or
federated policies, and is not additional evidence that federation solves cold
start. Its optional checkpoint reader can inspect existing experiment summaries
without changing them. The hosted agent defaults to synthetic sites only.

Version 0.2 combines Fusion Investigator's live frontend, remote adapters and
checkpoint reader with THERMAL's reduced transport solver, facility stewards and
disclosure gateway. There is one AgentApp and one dashboard. See
[merge notes](MERGE.md) for provenance, fixes and validation limits.

A between-experiment assistant built with Flower's **AgentApp harness**. Flower
owns app execution, the model bridge, run-series traces and shared run state.
The app supplies fusion analysis tools and a bounded tool loop, following
[Flower's collaborative-agent tutorial](https://flower.ai/docs/agent/tutorials/build-a-collaborative-agent.html).

## What works

- Read this repository's atomic cold-start checkpoints, including partial runs,
  source costs, paired outcomes, censoring and selection-trial sensitivity.
- Investigate heating-source versus transport ambiguity at `facility-a/b/c`.
  The reduced radial solver uses DIII-D-like, SPARC-like and TCV-like parameters.
  An investigator asks separate facility-steward model contexts for evidence.
  The gateway enforces prerequisites, a five-unit case allowance, cached
  releases and the demo's gradient-based analogy rejection rule.
- Replay evidence decisions, view released radial profiles and request products
  manually in the browser. Import an earlier THERMAL `report.json` locally.
- Follow **How B's shared evidence helps A**: a before/after audit graph of B's
  released transport categories, linked to the supported next measurement at A.
  It follows the selected report and replay position, hides unreleased findings,
  and cites evidence IDs. It illustrates information sharing, not a measured
  advantage over a central agent or a diagnosis transferred to A.
- Select Facility A, B or C through interactive 3D tokamak cards. Click a reactor
  or its labelled button to select it; drag to orbit. Selection synchronizes the
  evidence request, thermal tools and assembly viewer. Geometry comes from the
  existing `assets/3d/html/` exports and requires WebGL plus the pinned Three.js
  CDN dependencies. The labelled buttons remain usable without 3D.
- Explore independent `demo-a/b/c` toy cases. Local fitting ranks reduced heating
  effectiveness against increased heat loss. These are not the same facilities
  or model as the transport investigation above.
- Validate a proposed power scaling with six short heat-balance rollouts under
  parameter perturbations. These are dimensionless toy simulations, **not TORAX**.
- Call independently hosted analysis services over an authenticated HTTP API.
  The supplied local services return summaries rather than raw discharge traces.
- Let a model choose these tools through Flower's runtime, stream its report to
  Flower Chat, retain completed conversation turns, and record tool evidence in
  `context.state["fusion.last_investigation"]` and run logs.

This is an executable research prototype. There are no real facility connections,
physical actuation, controller deployment, new federated-training jobs, or claims
of demonstrated agent benefit. It never starts or modifies the cold-start study.
Actual plant diagnostics need a separately validated site adapter. The demo is a
workflow demonstration, not a realistic tokamak digital twin or causal diagnosis.

## Install and build

### Public Flower Hub release

Public agent identity: `@marykor/fusion-investigator`. The LLM was not trained
or fine-tuned on TORAX data. Default agent evidence comes from the reduced
thermal model; TORAX control training belongs to the separate research project.
The Hub listing serves the agent, while this repository supplies the local 3D UI.

`python scripts/prepare_hub.py` stages an explicit allowlist in
`dist/fusion-investigator/`, using `scripts/HUB_README.md` for the listing.
Review that directory, build it with `flwr build`, then publish it with
`flwr app publish dist/fusion-investigator`. It excludes credentials, reports,
ledgers, experiment data and private/generated thermal fixtures.

### Local frontend

From `flower-app`, launch the dashboard with the app's own environment:

```powershell
.venv/Scripts/python.exe dashboard.py
```

Open <http://127.0.0.1:8787>. The dashboard includes live synthetic partner
diagnostics, an interactive six-rollout validation chart, the hosted agent report,
tool evidence, and runtime logs. The validation sandbox runs the actual local
tools without an LLM and remains separate from the hosted agent's saved report.

The main **Facility evidence** view has three explicitly labeled sources:

- **Local rehearsal:** `Run local demonstration` runs a fixed sequence through
  the real gateway, with no LLM. You can also request products individually.
  Play/rewind only changes the view; it never resets disclosure budgets.
- **Latest hosted agent run:** renders `request_evidence` results from the actual
  agent audit. A pre-merge toy-only run correctly has no THERMAL events.
- **Imported THERMAL report:** reads a local JSON file in the browser, without
  uploading it. This is recorded evidence, not a new live investigation.

Rehearsal ledgers live under `.dashboard/rehearsal/`. Agent/site ledgers use
`FUSION_LEDGER_DIR` (default `.fusion-state/`). They are deliberately independent.

**Run investigation** submits this AgentApp with the actual Flower CLI to
your account's default federation (or the `FUSION_FEDERATION` override), using
your existing `flwr login supergrid` session and model
quota. It polls Flower for the run's status and logs. Each submission is an
independent investigation, using the packaged synthetic sites. Local TORAX
checkpoints are not sent. Only one dashboard-submitted run is active at a time.

The web server binds to loopback, enforces same-origin mutations, and never sends
account credentials to the browser. Run metadata and logs persist in ignored
`.dashboard/`; restart the server to reopen the saved report. If monitoring is
interrupted, follow the saved run link to Flower. Dashboard files and local state
are outside the FAB's include list. This is a local frontend, not a hosted website.

### Python environment

Use this subdirectory as the Flower project. Its environment is separate from the
parent's TORAX environment; do not install these dependencies into the parent.
Run these PowerShell commands from the repository root:

```powershell
cd flower-app
$env:PYTHONUTF8 = "1"
uv sync --extra dev
uv run flwr build
uv run python -m pytest tests
```

The lockfile pins Flower 1.37.0. The FAB includes the app's Python package, the
public synthetic device-parameter snapshot, README and license. Experiment files,
site configurations, profile fixtures, ledgers and credentials are not bundled.

Try the domain tools without a model or API key:

```powershell
uv run fusion-investigator demo
uv run fusion-investigator tool request_evidence '{"site":"facility-a","kind":"context"}'
uv run fusion-investigator --config sites.example.json sites
uv run fusion-investigator --config sites.example.json tool list_studies '{"site":"repository"}'
```

Copy a study identifier from that output to inspect its current saved results:

```powershell
uv run fusion-investigator --config sites.example.json tool inspect_study '{"site":"repository","study":"STUDY_ID"}'
```

The deterministic CLI demo is not an LLM run. It intentionally tests a 1.4 power
scale that does not improve every perturbed scenario, demonstrating that a
candidate can fail validation. The model may propose another bounded candidate.

## Run using the actual Flower harness

Follow Flower's [local SuperLink guide](https://flower.ai/docs/agent/how-to-guides/run-with-local-superlink.html).
In the first terminal, from this folder:

```powershell
$env:FUSION_SITES_CONFIG = (Resolve-Path sites.example.json).Path
# Option A: Flower's model service. Set your key in this terminal.
$env:FLWR_MODEL_API_KEY = "YOUR_FLOWER_MODEL_KEY"
uv run flower-superlink --insecure --disable-runtime-dependency-installation
```

The disabled dependency-installation flag reuses the installed app environment.
Only bind insecure development services to loopback. To use a local model instead,
configure a tool-capable Ollama model as described in Flower's
[Ollama guide](https://flower.ai/docs/agent/how-to-guides/run-with-ollama.html):

```powershell
$env:FLWR_MODEL_API_ENDPOINT = "http://127.0.0.1:11434/v1/responses"
# Start SuperLink after setting the endpoint and FUSION_SITES_CONFIG.
```

Add this named connection to your Flower CLI configuration (`~/.flwr/config.toml`),
preserving any existing connections:

```toml
[superlink.fusion-local]
address = "127.0.0.1:8000"
insecure = true
```

From a second terminal in this folder:

```powershell
uv run flwr run . fusion-local --stream
uv run flwr run . fusion-local --run-config 'agent.input="Inspect the latest repository cold-start study. Does federation help?"' --stream
# If using Ollama, also override the model:
uv run flwr run . fusion-local --run-config 'agent.model="qwen3.5:4b"' --stream
```

For conversational use on SuperGrid, use `flwr chat` and load this app with
`/load .`. Flower 1.37's chat CLI targets SuperGrid; use `flwr run` for local runs.
`agent.max-tool-turns` (1–8) and `agent.max-tool-calls` (1–24) cap work per run.
The app uses Flower's `FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY`; it never
connects directly to a model provider. The provider must support Responses
function calling and streaming. No model keys are included in this project.

To run on SuperGrid, use `flwr login supergrid` then `flwr run . supergrid --stream`.
The default packaged configuration exposes only the synthetic sites. Repository
results require an explicitly configured reachable site service: paths on your
laptop are not available inside a hosted executor. Provision configuration and
credential environment variables in the execution environment before using it.
Publishing this source does not deploy it; hosted runs are submitted separately
through Flower's CLI or the local dashboard.

For a key-free **runtime contract test**, run `uv run python tests/runtime_smoke.py`.
It starts a real local SuperLink, submits this FAB, executes a site tool, and
checks streamed output using a scripted local Responses server. That server is
a test double, not an LLM. Logs go to ignored `smoke-output/`; the script stops
its own processes when finished. An actual model run still needs a provider.

## Independently hosted sites

For the merged transport investigation, serve a site-owned gateway:

```powershell
$env:FUSION_THERMAL_TOKEN = "SET_A_RANDOM_TOKEN_OF_AT_LEAST_24_CHARACTERS"
$env:FUSION_LEDGER_DIR = ".fusion-state/facility-a"
uv run fusion-investigator serve --site facility-a --port 8766 --token-env FUSION_THERMAL_TOKEN
```

`sites.remote.example.json` maps `facility-a` to that authenticated endpoint;
its `request_evidence` and `evidence_history` calls use the same gateway rules as
local execution. A steward's approval never overrides a gateway rejection.

Run the same analysis adapter near a site's data. For a loopback demo:

```powershell
$env:FUSION_SITE_TOKEN = "SET_A_RANDOM_TOKEN_OF_AT_LEAST_24_CHARACTERS"
uv run fusion-investigator serve --site demo-a --port 8765
```

In the SuperLink terminal, set the same token and use:

```powershell
$env:FUSION_SITES_CONFIG = (Resolve-Path sites.remote.example.json).Path
```

Restart SuperLink to inherit updated environment variables. The agent now calls
demo-a via HTTP while demo-b and demo-c remain in-process. To serve real saved
simulation results, select `--config sites.example.json serve --site repository`.
Configure the corresponding remote entry with exactly the same site ID.

The API is `POST /analysis` with bearer authentication and JSON
`{"operation":"diagnose_heating","arguments":{"case_id":"heating-response"}}`.
Only listed operations are accepted. URLs and credentials are operator-configured,
never model-supplied; redirects are refused. Remote hosts require HTTPS (loopback
HTTP is allowed for development). Put the loopback service behind an authenticated
HTTPS gateway for a real cross-site deployment, with institution-managed access,
rate limits and output-release policies. This small server is a development adapter.

Shared model input, run state and logs contain the approved summaries. They are
**not automatically confidential**. Remote adapters are responsible for their own
release policy. The service does not implement differential privacy or secure
aggregation; a Flower workspace does not supply those guarantees by itself.

## Evidence and evaluation

The primary thermal workflow keeps A's diagnosis unresolved: B's independent
source audit suggests what A should measure; C fails an explicit gradient-based
analogy criterion. This is a controlled scenario, not proof that subcritical
gradients are universally unidentifiable in real plasmas. The source discrepancy
and audit availability are scenario assumptions. Model constants are illustrative.

SQLite accounting survives restarts when the operator retains the ledger path.
Ephemeral SuperGrid workers do not guarantee cross-run persistence. Evidence IDs
and cache keys include a fingerprint of the solver, device inputs and configured
fixtures, so a changed simulation cannot reuse an old cached finding. Policy
units are not differential privacy. Default sites share one process; separate
steward contexts are not independent security domains.

The optional `scripts/generate_thermal_fixtures.py` must run with the parent
repository's TORAX environment, separately from the app. Its actuator fractions
and TORAX 1.4.3 source-dictionary extraction have regression tests. No full TORAX
fixture generation is claimed by this merge. Generated snapshots are excluded
from the FAB; `FUSION_THERMAL_FIXTURES` selects a site-owned JSON file. Unverified
steady-state snapshots are allowed to disclose context but the gateway blocks
balance/source conclusions until that assumption is independently checked.

The primary recorded endpoint is two consecutive eligible evaluations completing
all steps with no limit violations and tracking error within tolerance. A separate
sensitivity excludes handover-selection trials from confirmation but charges all
shots. It uses each run's recorded commissioning/selection counts, not hardcoded
shot indices. Source training cost is reported separately. Pending arms stay visible.

Test agent usefulness separately from learning usefulness: fixed workflow versus
agent-directed investigation with identical tools/budgets; local-only versus
cross-site evidence; and pooled versus federated model training when data pooling
is allowed. Use held-out facilities/campaigns, diagnosis accuracy, unsuccessful
validation attempts and analysis cost. This app does not establish those results.

Background sources:

- [Princeton/PPPL PACMAN](https://arxiv.org/abs/2511.08818): local integrated control;
  not proof of federated or multi-agent RL.
- [Offline model-based tokamak RL](https://proceedings.mlr.press/v211/char23a.html):
  historical-data training followed by DIII-D testing.
- [Cross-tokamak transfer](https://www.nature.com/articles/s42005-023-01296-9):
  a motivation for cold-start transfer, not federation superiority.
- [Flower Agent runtime](https://flower.ai/docs/agent/explanations/agentapp-runtime.html):
  AgentApp execution and federated-learning applications are separate components.
