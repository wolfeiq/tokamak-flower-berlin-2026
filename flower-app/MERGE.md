# Combined app

Fusion Investigator 0.2 combines the app in `codex/initial-snapshot` with the
transport-investigation components from `codex/thermal-investigation` at
`44a865f`. The branch is merged into the canonical `flower-app/`; the former
`apps/thermal-investigation/` entry now points here. Experiment code, results
and the separate 3D UI remain separate from the merged application.

## What was retained

- Fusion Investigator: actual Flower AgentApp harness, hosted-run launcher,
  reports, model history, cold-start checkpoint reader, authenticated remote
  sites and the independent toy validation sandbox.
- THERMAL: reduced radial forward solver, source-versus-transport investigation,
  facility-steward model contexts, fixed release catalogue, SQLite accounting,
  prerequisite checks, cached releases and evidence-trail visualization.

THERMAL's numerical `profiles.py` and pure-NumPy `_closure.py` were migrated to
`fusion_agent/thermal/`, along with its public `_devices.json` parameter snapshot.
The closure snapshot came from `hfmarl/identification/closure.py` (SHA256
`1ec9e2aad1e6e01608414839ceb5639a4f1e24645476d1912cb8a833f045026d`).
This package deliberately uses its own copy so hosted and local runs agree.
Device parameters came from the original branch's preparation script and registry.

## Fixes during integration

- Converted optional TORAX heating fractions to the actuator's `[-1, 1]` range:
  0.001 now commands 0.1% rather than 50.05% of that range.
- Read TORAX 1.4.3's actual dictionary of electron source arrays, including
  ion/electron exchange; reject mismatched or non-finite profile grids.
- Block steady-state balance disclosures for unverified transient snapshots.
  Their source sum is not an independent delivered-power measurement.
- Treat incomplete/empty model reports as failure, while recovering text from
  a completed terminal event when the stream supplies no text deltas.
- Version cached evidence by numerical inputs and implementation, preventing
  stale cached findings after a model or fixture change.
- Keep local rehearsal ledgers separate from live agent/site accounting, and
  label toy, reduced-model and recorded-hosted evidence separately in the UI.
- Discard steward prose and malformed/incomplete approvals; enforce policy at
  the site gateway even when the investigator uses a remote HTTP adapter.

## Validation boundaries

Tests cover solver residuals, site differences, source ambiguity, disclosure
budgets, repeated/concurrent releases, stewardship, remote transport, dashboard
requests, heating conversion and final-stream failure handling. The local
SuperLink smoke test packages and executes the app with a scripted Responses
provider; it is a runtime contract check, not a live-model performance result.

Existing SuperGrid reports remain attributable to their earlier app version.
This integration does not itself submit a paid hosted run, run a new cold-start
experiment or establish that federation beats centralized data pooling.
