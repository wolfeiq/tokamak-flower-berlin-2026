# Berlin 2026 — presentation plan

Two assets, one visual system. The **deck** (the HFMARL artifact: near-black,
coral, Archivo/IBM Plex Mono) carries the argument; the **live app**
(`flower-app/dashboard.py`) now wears the same skin and carries the proof. Move
between them without a visual seam.

## Shape of the talk (8 minutes + demo)

1. **Deck, slides 1–4 (2 min).** The claim: a device must learn its limits from
   devices that have already crossed them; nothing raw leaves a machine. Land on
   the dimensionless-overlap slide — it is why cross-facility evidence is
   plausible at all.
2. **Cut to the app (4 min).** `uv run python dashboard.py` → 127.0.0.1:8787.
   - *Facility evidence*: press **Run local demonstration**. Narrate the trail:
     A releases context and balance, C's balance is **denied —
     analogy-not-applicable**, budgets tick down and persist. The denial is the
     product: the gateway, not the model, holds the boundary.
   - *Physics playground* — "the simulation, inside the simulation". Same
     solver the facilities run, ported to the browser.
     - Drag **assumed/delivered** from ×1.60 to ×1.00: apparent χ falls onto
       the device's own reference. That slider IS the investigation.
     - Switch to facility C, drag power to 0.1%: the trace goes flat,
       apparent χ reads **unidentifiable**. C's rejection is physics, not a
       hardcoded branch.
     - **Load 3D reactor assembly**: the machine whose transport you just
       solved, from the same registry parameters.
   - *Run investigation* (live, if the room's network cooperates): a real
     Flower AgentApp run in @seroj/workspace; the report cites evidence IDs.
3. **Back to deck (2 min).** The bug-the-gates-found slide (one missing
   normalisation, factor 30 in pressure) — it shows the project catches its own
   errors. Close on Status/Next: claims sized to evidence.

## Fallback ladder for the live run

1. Live: `uv run flwr run . supergrid --stream` (needs login + entitlement).
2. Recorded: `python -m thermal_investigation.fetch_report <run-id> supergrid`
   — dashboard badge says RECORDED FLOWER RUN, which is the honest label.
3. Offline: **Run local demonstration** — badge says local rehearsal, no LLM.
   Never present a lower rung as a higher one; the badges exist so you cannot.

## Claims to make — and not

- SAY: profiles are solved per real device (gyro-Bohm: SPARC-like's 12.2 T ⇒
  ~10× less transport than DIII-D-like); the ambiguity, its resolution at B and
  its rejection at C are all emergent; the FAB excludes ledger, reports, HTML.
- DON'T SAY: TORAX (unless `_fixtures.json` is committed — provenance strings
  in every finding say which), differential privacy (policy units are costs),
  validated diagnosis, or cross-run budgets on SuperGrid (containers are
  ephemeral there).
