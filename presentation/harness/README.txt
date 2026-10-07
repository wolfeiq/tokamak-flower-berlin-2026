FUSION INVESTIGATOR — FIVE-MINUTE HARNESS SEGMENT

OPEN
  fusion-investigator.pptx   Main six-slide talk, with speaker notes
  index.html                Same talk in the browser, plus optional replay
  presentation.pdf          Six-page static fallback
  speech_for_harness.txt     Spoken script with timings and pointing cues

The main talk uses a static tool-loop diagram and a worked heating example.
No video, model quota, server or network connection is needed to present it.
PowerPoint artwork is exported at 2400 x 1350; notes remain editable text.
Use Presenter View for private PowerPoint notes. Browser notes appear on the
same screen as the slides, so hide them before projecting.

BROWSER CONTROLS
  Right / Space / Page Down: next slide
  Left / Page Up: previous slide
  Home / End: first / last slide
  F: fullscreen
  N: notes
  Slide 5, Inspect gateway replay: optional ten-event evidence replay
  D on slide 5: open replay and advance one event

The optional replay contains actual recorded Gateway.request outputs, using
synthetic cases and fixed requests. It does not show a model-directed run.
Play / Next / event markers change the replay view. Rewind never refunds
disclosure spending. Return to the worked example before continuing the talk.
Print/PDF/PowerPoint always use the static example.

PACING
  1  What it is / who uses it       0:00-0:40
  2  Inputs and outputs            0:40-1:20
  3  Why use an agent?             1:20-2:05
  4  Flower and the tool loop      2:05-3:25
  5  Worked heating example        3:25-4:25
  6  Next measurement / evaluation  4:25-5:00

About 610 spoken words. Aim near 130 words per minute, with brief pauses.
Rehearse with a timer. This is one five-minute part of a twenty-minute talk;
it does not repeat the broader project's background or cold-start results.

TECHNICAL REVIEW — 7 OCTOBER 2026
Suitable as an engineering research prototype presentation to a technical
audience, with the claims below. Real diagnostic validity, production deployment
and demonstrated scientific benefit remain unestablished.

Implemented:
  Flower AgentApp execution, runtime model access and explicit report events.
  App-owned bounded tool loop, structured requests and separate steward contexts.
  Deterministic evidence gateway, prerequisite checks and SQLite accounting.
  Custom local/remote adapters; default synthetic sites share one process.

Shown by the example:
  A remains unresolved under an unverified heating-source assumption.
  B's constructed source audit motivates A's next measurement.
  C fails the chosen demo applicability rule and its balance is withheld.

Still to evaluate:
  Agent versus fixed-workflow quality, cost and robustness on held-out cases.
  Steward benefit; useful cross-site information beyond local evidence.
  Real diagnostic validity and independent institutional deployment.

The gradient cutoff is illustrative, not a universal identifiability limit.
Five disclosure units are policy accounting, not differential privacy.
Flower run-series Context and the gateway SQLite ledger are separate stores.
A retained Context does not guarantee a durable file on a hosted worker.
Report citations are prompted; complete report entailment is not enforced.

REVIEW VALIDATION
  App tests: 69 passed; 1 skipped because local checkpoint data was absent.
  Root identification checks: 24 passed; four PINN tests excluded from this check.
  Real local Flower 1.37 runtime smoke: passed with a scripted model provider.
  Gateway/browser integration: passed, including denials, cache and persistence.
  Browser export: six pages, notes matching speech, projector/laptop sizing.
  PowerPoint builder validates six slides and all six notes in the saved package.
The model test double checks runtime integration, not live-model reasoning.
The broader TORAX control experiment was not rerun for this presentation review.

PRIVATE PREPARATION
  speaker_deep_dive.txt: foundations, code walkthrough, physics derivation,
  limitations, evaluation design and 160 answered questions.
It is ignored in this folder's .gitignore and is not served by the local server.
It remains a local personal file; Git ignore is not filesystem encryption.

OPTIONAL LIVE GATEWAY
From the repository root:
  python -m venv presentation/harness/.venv
  presentation/harness/.venv/Scripts/python.exe -m pip install -r presentation/harness/requirements.txt
  presentation/harness/.venv/Scripts/python.exe presentation/harness/serve.py

Open http://127.0.0.1:8790/, go to slide 5, Inspect gateway replay,
then choose Live Python gateway. Use Custom request to inspect a denial.
This exercises the actual gateway without Flower/model calls.
The server explicitly selects the packaged synthetic model.
Its ledger is presentation/harness/.state/presentation.sqlite3.
Keep that file to retain spending. Repeated runs may return cached releases.

REBUILD AND VERIFY
From the repository root:
  presentation/harness/.venv/Scripts/python.exe presentation/harness/capture_evidence.py
  presentation/harness/.venv/Scripts/python.exe presentation/harness/verify.py
  presentation/harness/.venv/Scripts/python.exe -m pip install -r presentation/harness/powerpoint-requirements.txt
  node presentation/harness/export.mjs --powerpoint
  presentation/harness/.venv/Scripts/python.exe presentation/harness/build_powerpoint.py

Capture regenerates evidence.js using a fresh temporary ledger and validates
the scenario. The case fingerprint includes source bytes, so even changes to
numerical-source comments require a new recording to keep provenance current.

The browser exporter needs Node 22+ and local Edge/Chrome. CHROME_PATH can
override the browser path. No npm dependencies are installed. It generates
the PDF, preview PNGs and .exports/slide-*.png. Close the old PowerPoint if
Windows prevents replacing it. There is no video/ffmpeg build step.

SOURCE MAP
  index.html / slides.css     Slide content and appearance
  slides.js / evidence.js    Navigation and optional captured gateway replay
  speech_for_harness.txt     Spoken script; keep HTML notes synchronized
  export.mjs                Browser checks, screenshots and PDF
  build_powerpoint.py        PowerPoint artwork and native notes
  capture_evidence.py        Fresh evidence recording
  serve.py / verify.py       Optional local gateway and integration check
  ../../flower-app/fusion_agent/agent_app.py       Flower entry point
  ../../flower-app/fusion_agent/orchestrator.py    Model/tool loop and history
  ../../flower-app/fusion_agent/steward.py         Separate model context
  ../../flower-app/fusion_agent/thermal/core.py    Release policy and accounting
  ../../flower-app/fusion_agent/thermal/profiles.py  Reduced forward solver

PLATFORM REFERENCES
  https://flower.ai/docs/agent/explanations/agentapp-runtime.html
  https://flower.ai/docs/agent/how-to-guides/run-with-local-superlink.html

Online docs checked on 7 October 2026 describe newer Flower versions in places.
This app targets 1.37.0; use its source and tested runtime contract for exact
API/CLI behavior rather than silently substituting current documentation.
