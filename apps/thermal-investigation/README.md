# THERMAL — cross-facility evidence investigation

A companion Flower AgentApp for HFMARL. An investigator asks facility stewards
for bounded findings, checks applicability, and recommends the next measurement.
No actuator tools or RL-policy changes.

## Reproducible visual demo

From the repository root:

```bash
python scripts/prepare_thermal_investigation.py
cd apps/thermal-investigation
uv sync --extra dev
uv run python -m thermal_investigation.demo
uv run python -m http.server 8765 --bind 127.0.0.1 --directory investigation-output
```

Open http://127.0.0.1:8765. Play the trail or advance one event at a time. The
page labels this mode SCRIPTED REPLAY: no LLM was invoked. Serve ONLY the output
folder, never the repository or adjacent ledger. Restart rewinds the visualisation
without resetting budgets.

## Real Flower agents

Separate environment targeting Flower 1.37.x; root TORAX dependencies are unchanged.
After preparing the numerical module as above:

```bash
uv run flwr build
uv run flwr login supergrid
uv run flwr run . supergrid --stream
```

Flower injects `FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY`. A live run
requires your Flower account and model access. The investigator chooses tools
dynamically; each request invokes a facility steward in a separate model context.
The steward selects a release tool or declines. Only the gateway's response
returns to the investigator; steward prose is never an export channel.

The Flower UI streams the final advisory report. `investigation-output/report.json`
is written ON THE RUNTIME HOST. The HTML dashboard stays in the local checkout
(Flower FABs exclude HTML).

A deployment need not expose an artifact provider, in which case `flwr pull`
fails with `ControlServicer initialized without artifact provider` even though
the run itself completed. The AgentApp therefore fences its sanitized report
into the run log, and this recovers it into a local dashboard:

```bash
uv run python -m thermal_investigation.fetch_report <run-id> supergrid
uv run python -m http.server 8765 --bind 127.0.0.1 --directory investigation-output
```

The log carries exactly what `report.json` already contains: released evidence
and gateway decisions. Raw fixtures, ground truth and steward prose are excluded
before that point and never reach the log. The dashboard labels a recovered run
RECORDED FLOWER RUN rather than SCRIPTED REPLAY. The standalone page can also
open a report with its file picker. It replays recorded runs; it does not
stream live inter-agent traffic.

Live runs need `flwr login supergrid` plus an account entitled to start
Deployment Runtime runs; without that entitlement `flwr run` returns
`Entitlement error. Denied: Starting a run for Deployment Runtime is not allowed.`
A stale token reports `Authentication failed` instead and is fixed by logging in
again. Running against the local SuperLink (`flwr run .`) skips Flower's model
routing and requires `FLWR_MODEL_API_KEY`, optionally with
`FLWR_MODEL_API_ENDPOINT` pointing at any Open Responses-compatible provider.

The agent chooses its own sequence, so a live trail differs from the replay and
between runs: observed runs have requested five to seven products and have
stopped at C's context once the flat profile made the analogy unusable, rather
than spending budget on C's balance as the scripted trail does.

Official documentation:
- https://flower.ai/docs/agent/tutorials/write-your-first-agentapp.html
- https://flower.ai/docs/agent/tutorials/build-a-collaborative-agent.html

`THERMAL_LEDGER` selects a service-owned persistent SQLite path (default
`investigation-state/disclosure.sqlite`). Budgets persist across runs only when
this path is retained. Ephemeral cloud workers without persistent storage do not
provide cross-run accounting. The model cannot set the path, case identity, costs
or allowance. Use separate working directories for live and rehearsal ledgers.

## What is computed

The preparation script copies the repo's `hfmarl/identification/closure.py`
byte-for-byte into the FAB and prints its hash. This keeps TORAX/JAX out of the
agent environment. Regenerate after changing the estimator; the copy is gitignored.

These are manufactured steady-state SPATIAL profiles, not TORAX trajectories,
real incidents, or validated fault diagnoses:

- A and B have usable gradients. The estimator calculates apparent transport
  using a source assumed at 1.6 times the constructed source.
- B has a synthetic independent source audit. Recalculation with that source
  returns transport near the fixture's reference value.
- A lacks that measurement. B's findings cannot establish A's cause.
- C has a flat profile: transport estimation is invalid; further release is denied.

Constants and thresholds are illustrative assumptions, not empirically calibrated
cross-machine criteria. B's audit is a synthetic measurement, not something
inferred from temperature. No `chi_true` or `q_true` arrays enter model contexts
or exported reports. Units are normalised toy units.

## Disclosure controls and limitations

Every product is constructed from explicit fields, enums and fixed coarse bins.
No free-form local notes, arbitrary paths, raw logs, SQL, shell, web or actuator
tools are exposed. Raw arrays never enter a model context. Balance intentionally
releases an 11-point normalised, one-decimal profile: remove it if real facility
policy does not permit shape disclosure.

| Product | Cost | Required evidence |
| --- | ---: | --- |
| context | 1 | None |
| balance | 2 | Comparable context |
| source_check | 2 | Released balance |

Five policy units per fixed case/site. Unknown requests are denied without echoing
their contents. Repeat releases return cached data without a second debit.
Transactions prevent duplicate concurrent debits. Releases and gateway denials
enter an audit trail. Shared JSON contains sanitized boundary decisions only.
Steward refusals are returned to the coordinator but not recorded as releases.
The coordinator has eight tool rounds and twelve facility invocations at most;
each invocation uses one steward model call.

Policy costs are NOT differential privacy. The audit is not tamper proof. This
does not prove every approved finding is necessary or non-identifying. It enforces
a fixed catalogue, prerequisites and a persistent case allowance. Necessity rules
are deliberately narrow and must be reviewed for each real use case.

All facilities share one process in this prototype: application boundaries, not
independent infrastructure security. A compromised host can access all fixtures
or edit the ledger. Production requires separate site-owned services/stores,
authentication, persistent ledgers, rate limits, approved case enrolment and
verifiable provenance. Never package real private data into a shared FAB. Hosted
models see permitted summaries, which still disclose information and require an
approved provider arrangement. The public synthetic fixtures need no secrecy.

## Three-minute demonstration

1. A: apparent transport is elevated. Heating discrepancy or transport change?
2. Consult B; reject C's superficially similar but unidentifiable case.
3. B's source audit changes which evidence is worth collecting at A.
4. Show a raw-log request denied and a repeat request served from cache.
5. End with a specific next measurement and unresolved A diagnosis.

The live agent can choose a different sequence. Scripted replay is rehearsal,
not proof of autonomous performance. Cards show approved knowledge and spending;
the trail shows boundary decisions; profiles show why visual resemblance alone
is insufficient; the report recommends the next measurement.

## Verification and next steps

```bash
uv run pytest tests -q
```

Tests cover numerical ambiguity, flat-profile rejection, invalid/injected requests,
persistence, concurrent duplicates, no ground-truth export, discarded steward
free-text and bounded orchestration. Model decisions are mocked: no claim of
validated LLM diagnostic quality.

Next add a validated adapter for actual simulation profiles: radius, temperature,
density, source basis, time and geometry. Run-summary JSON may lack these arrays.
Keep ground truth inaccessible to agents. Policy-transfer similarity is not yet
a validated metric for incident analogy.

Compare local-only, static peer summaries and adaptive peer questions on held-out
cases at equal resource budgets. Measure useful next measurements, unsupported
causal claims, analogy rejection and disclosure cost. No collaboration advantage
has yet been demonstrated by this prototype.
