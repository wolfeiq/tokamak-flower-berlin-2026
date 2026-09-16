# Berlin 2026 — presentation plan

Two assets, one visual system. The **deck** (the HFMARL artifact: near-black,
coral, Archivo/IBM Plex Mono) carries the argument; the **live app**
(`flower-app/dashboard.py`) now wears the same skin and carries the proof. Move
between them without a visual seam.

## Shape of the talk (8 minutes)

The app IS the deck. `cd flower-app && uv run python dashboard.py` →
127.0.0.1:8787. Arrow keys advance; every slide is live. Nine slides:

| # | Slide | What you do |
| ---: | --- | --- |
| 01 | Shared evidence. Local control. | State the problem: one weak thermal response, three facilities, nothing raw may leave. |
| 02 | Ask the investigator | Submit the brief. It runs for ~2 min on Flower — start it here and let it work while you talk. |
| 03 | Four machines. One aggregation round. | **Load federation atlas.** Click machines to send; the aggregate panel counts them and shows each one's ρ*, ν*, β_N and per-channel weight. Every number is computed by the repo's own FedBuffServer. |
| 04 | Three machines. One question. | **Run local demonstration.** Budgets tick down; C's balance is DENIED. The denial is the product. |
| 05 | What crossed. What was refused. | Profiles and the evidence trail side by side. |
| 06 | The simulation, inside the simulation. | Drag assumed/delivered ×1.60 → ×1.00: A's "anomaly" becomes B's audited resolution. Switch to C, drop power to 0.1%: flat, unidentifiable. |
| 07 | The machine itself. | **Load 3D reactor assembly** for the machine just solved. |
| 08 | Challenge a candidate. | Toy sandbox — explicitly separate from THERMAL. |
| 09 | The advisory, verbatim. | The run from slide 02 has finished. Read its cited evidence IDs. |

Adding a slide is adding a `<section class="slide">` with a kicker and an h2;
`deck.js` registers it in DOM order. Add `class="slide stage"` for a
visualization that should own the viewport.

## Fallback ladder for the live run

1. Live: `uv run flwr run . supergrid --stream` (needs login + entitlement).
2. Recorded: `python -m thermal_investigation.fetch_report <run-id> supergrid`
   — dashboard badge says RECORDED FLOWER RUN, which is the honest label.
3. Offline: **Run local demonstration** — badge says local rehearsal, no LLM.
   Never present a lower rung as a higher one; the badges exist so you cannot.

## Claims to make — and not

- SAY: profiles are solved per real device (gyro-Bohm: SPARC-like's 12.2 T ⇒
  ~10× less transport than DIII-D-like); the ambiguity, its resolution at B and
  its rejection at C are all emergent; the atlas weights come from the real
  aggregator; the FAB excludes ledger, reports and HTML.
- DON'T SAY: TORAX (unless `_fixtures.json` is committed — provenance strings
  in every finding say which), differential privacy (policy units are costs),
  validated diagnosis, or cross-run budgets on SuperGrid (containers are
  ephemeral there).

## The app, separately on Flower

The deck is the local presenter. The AgentApp itself publishes to Flower Hub:

```bash
cd flower-app
uv run flwr build            # seroj.fusion-investigator.0-2-0.<hash>.fab
uv run flwr app publish .
```

`publisher` in `pyproject.toml` must match the Flower account doing the
publish. Anyone can then run it without this repo:

```bash
uv run flwr run <app-id> --stream
```

The dashboard is deliberately NOT part of that bundle: FABs carry no HTML, and
the disclosure ledger must stay on the host that owns it.
