FUSION INVESTIGATOR — FIVE-MINUTE PRESENTATION

OPEN THE DECK
Double-click index.html. The six slides, animated visual and embedded demo
work offline. Fonts and all captured evidence are bundled locally.

PRESENT
Arrow keys / Space: next slide. Left arrow: previous slide.
F: fullscreen. N: speaker notes. D: next demo event (on slide 5).
Home / End: first / last slide. The bottom navigation is also clickable.

Slide 5 contains an interactive, ten-event replay of the project's actual
Python gateway outputs. Use Play sequence or Next event. Click any event
marker to inspect that point. Rewind changes the view only.
The mode label always identifies recorded evidence versus live Python calls.
Both are fixed synthetic rehearsals without an LLM.

Open fusion-investigator.pptx in Microsoft PowerPoint for the six-slide talk.
The slide artwork preserves the browser design as high-resolution images;
speaker notes and the demo link are editable native PowerPoint content.
Slide 5 embeds a 37-second MP4 gateway replay: click the video to play it.
Use Slide Show (F5 / Shift+F5) for playback. The video is packaged inside the
PPTX and needs no network connection. The native Open interactive demo link
opens the browser deck alongside the PPTX; keep this folder together.

The presentation.pdf is a static six-page fallback with the demo completed.
Use index.html for animation, interactions, speaker notes and live requests.

OPTIONAL LIVE GATEWAY
From the repository root, in PowerShell:

  python -m venv presentation/harness/.venv
  presentation/harness/.venv/Scripts/python.exe -m pip install -r presentation/harness/requirements.txt
  presentation/harness/.venv/Scripts/python.exe presentation/harness/serve.py

Open http://127.0.0.1:8790 and select Live Python gateway on slide 5.
Next event / Play sequence will call the actual Gateway class in flower-app.
Use Custom request to test prerequisites, raw-log denial and cached releases.
Stop the server with Ctrl+C. No Flower account or model quota is needed.

The ledger is presentation/harness/.state/presentation.sqlite3. Spending persists
when this file is retained. Rewinding never resets it. Repeated live runs
can therefore show cached results from the start. The bundled offline replay
always lets you present the original sequence from its first disclosure.
The live server explicitly selects the same packaged synthetic model as
the recording; it does not load operator-configured TORAX fixtures.

REGENERATE THE RECORDING
  presentation/harness/.venv/Scripts/python.exe presentation/harness/capture_evidence.py

This calls the actual gateway with a fresh temporary ledger, checks its key
outcomes, and writes evidence.js. It exports only released findings and denial
events. The source fingerprint and capture time are included in that file.

FILES
fusion-investigator.pptx PowerPoint with embedded demo video and speaker notes
build_powerpoint.py     Rebuild the PowerPoint from exported slide captures
powerpoint-requirements.txt Optional PowerPoint export dependencies
index.html              Editable slide content and inline speaker notes
slides.css              Visual design, stage scaling and PDF print layout
slides.js               Navigation, animation and embedded demo controls
evidence.js             Recorded gateway outputs, including evidence IDs
speech_for_harness.txt  The talk, with slide cues; around four minutes spoken
presentation.pdf        Static fallback / shareable handout
serve.py                Optional local server using the real gateway
capture_evidence.py     Reproducible offline evidence capture
export.mjs              Browser checks, slide screenshots and PDF export
verify.py               Live server + browser verification
assets/                 Bundled fonts and their license

REBUILD THE PDF / CHECK THE DECK
  node presentation/harness/export.mjs
  presentation/harness/.venv/Scripts/python.exe presentation/harness/verify.py

These use a locally installed Edge or Chrome (override with CHROME_PATH).
The first checks the offline deck; the second uses an isolated temporary
ledger and checks live integration too. Both regenerate the PDF and local
preview images. They require Node 22+ and install no npm dependencies.

REBUILD THE POWERPOINT
  presentation/harness/.venv/Scripts/python.exe -m pip install -r presentation/harness/powerpoint-requirements.txt
  node presentation/harness/export.mjs --powerpoint
  presentation/harness/.venv/Scripts/python.exe presentation/harness/build_powerpoint.py

PACING / THE QUESTIONS THE TALK ANSWERS
1  What it is, and who it is for   0:00–0:50
2  Inputs and outputs             0:50–1:40
3  Why this approach fits         1:40–2:35
4  Why Flower                    2:35–3:25
5  Embedded investigation         3:25–4:40
6  The next experiment            4:40–5:00

The speech's section labels and bracketed cues are not spoken. Press N to
see a reminder for the current slide. Notes appear on the presenting screen,
so hide them before projecting if you do not want the audience to see them.

FACTUAL SCOPE
Synthetic reduced 1-D steady-state transport; illustrative gradient criterion.
A remains unresolved. B's audit suggests A's next measurement. C's balance
is rejected. These are disclosure accounting units, not differential privacy.
The embedded demo executes / replays the gateway, not a hosted model run.
Source: flower-app/fusion_agent/thermal/core.py and dashboard.py::replay.

The architecture comparison explains fit under controlled-disclosure
requirements. It does not claim measured superiority over a fixed workflow.
Flower supplies the execution/model/event/state runtime. The project supplies
scientific tools, bounded tool orchestration, custom remote adapters, and the
Python/SQLite disclosure policy. The embedded demo exercises that policy.

Platform references:
https://flower.ai/docs/agent/explanations/agentapp-runtime.html
https://flower.ai/docs/agent/how-to-guides/run-with-local-superlink.html
