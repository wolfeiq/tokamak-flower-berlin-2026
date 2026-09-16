# Cold-start rerun, 16 September 2026

This plan is recorded before the repaired six-seed runs. The earlier runs were
stopped by the user. This is a simulation experiment testing a benefit, not a
promise that federation will succeed.

## Fixed design

- Task: `moderate`; held-out devices: `sparc_like`, then `tcv_like`.
- Seeds: 0–5, paired across arms. Source optimizer seeds are tied to device
  identity, so selecting a source subset does not change its random stream.
- Arms: scratch, one pretrained source, all sources merged once, repeated
  uniform federation, repeated similarity-weighted federation.
- Budget: 80 shots per source, 120 shots on each joiner. Evaluations, five
  initial probing shots, and acceptance checks count against those 120.
- Five upstream calibration shots per joiner were previously used to measure
  the cached device band. Their cost is additional and shared by all arms;
  the rerun does not repeat the calibration sweep.
- Merge: geometric median, alignment enabled, clipping factor 2, reject if
  worse enabled; exchange interval 20, evaluation interval 5; no scrambling.
- Run folds sequentially in one process, with an atomic checkpoint after each
  complete seed. Keep a copy and SHA-256 hashes of the source code.

Command, from the repository root:

```powershell
.venv/Scripts/python.exe -u scripts/exp_coldstart.py --joiners sparc_like,tcv_like --task moderate --seeds 6 --pretrain 80 --join 120 --local-shots 20 --eval-every 5 --select --plot --no-record
```

Resume an interruption with the same arguments plus `--resume <run-directory>`.
The loader refuses changed settings or source files. An incomplete seed is
repeated; complete seeds are not.

## Endpoints and interpretation

The primary endpoint is the proportion of seeds reaching joint competence
within the 120-shot budget. Competence requires two consecutive eligible
incumbent evaluations that complete, cross no limit, and have finite mean
tracking error within the device-specific tolerance. The confirming shot is
the reported cost. It measures progress during learning; it is not a separate
validation of one frozen controller.

The primary contrast is uniform federation versus scratch on SPARC-like,
selected because prior work identified this device as difficult. The TCV-like
fold checks a second plant on which limit violations are possible. Repeated
uniform federation versus one-time merge tests whether repeated exchange adds
value; similarity versus uniform tests weighting. These additional comparisons
are exploratory, with unadjusted p-values explicitly identified.

Report exact binomial reach intervals and exact paired tests by seed. Keep
censored seeds in the denominator. Six pairs have limited resolution: even
five method-only successes and no baseline-only successes give two-sided
paired p=0.0625. Non-significance does not show equivalence, and a minimum
detectable effect is not an upper bound on the true effect.

Plot cumulative competence over all seeds and reach rates with uncertainty,
not just the median among successful seeds. Report all arms, handover adoption,
aggregation rejection/fallback counts, and source/target costs. An improvement
over scratch alone does not isolate repeated federation from warm starting.
Previously saved conventional-controller results use per-shot success and are
context only; their denominator differs from this per-seed reach endpoint.

The scope is 1-D TORAX transport simulation with nominal circular-geometry
devices and manually imposed limits. Any benefit is limited to that setting.

## Fixes before this rerun

- One-time merge now actually aligns hidden units, using the model shape.
- Joint competence excludes unsafe, incomplete, non-finite, and rejected
  handover evaluations; the legacy tracking-only metric remains diagnostic.
- The acceptance comparison for a retired incumbent cannot certify the newly
  installed handover controller.
- Source histories use stable device-based random seeds.
- Handover operating coordinates and regions use the same recent-history
  convention as federated publications; scrambled coordinates and regions move
  together, with explicit fallback when the named source is held out.
- Raw per-seed decisions and disjoint shot accounting are saved. Checkpoints
  and exact source snapshots prevent interrupted work or uncommitted fixes
  from disappearing into a summary labeled only by a Git commit.

Validation before the real-simulator smoke check: 586 passed, 1 skipped,
2 expected failures, and 2 non-strict expected failures that passed. No
unexpected test failures.
