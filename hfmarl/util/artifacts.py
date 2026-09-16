"""Run directories that can be re-analysed, and cannot silently overwrite.

AUDIT #10. The experiment scripts saved a summary and threw the evidence away:
no per-shot records, no seed association for a crossing, no metric provenance,
and no frozen policy. When a metric defect was found -- the shared tolerance
that scored one joiner against another's criterion -- the saved cold-start
results could not be rescored, so an hour of compute became unusable rather
than merely wrong.

The output stem also omitted half the settings it varied (joiner set, clipping
factor, local-shot interval, smoothing window), so two genuinely different runs
wrote to one filename and the second silently replaced the first.

WHAT IS SAVED, AND WHY EACH PIECE
---------------------------------
    manifest.json   every setting, the commit, the environment, the resolved
                    per-device tasks and tolerances. Without the RESOLVED task
                    a result cannot be rescored, because the fractions in a
                    preset mean different absolute values on every device.
    runs.json       every ShotRecord of every run, keyed by (condition, device,
                    seed). This is what makes reanalysis possible at all.
    summary.json    the numbers as computed at the time, so a later reanalysis
                    can be COMPARED against what was originally reported rather
                    than quietly replacing it.

The directory name carries a timestamp and the commit, so nothing overwrites
anything, and a run is identifiable from its path alone.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from hfmarl.metrics.log import RunLog
from hfmarl.util.report import environment_stamp, repo_root


def _git_sha() -> str:
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=repo_root(), capture_output=True, text=True,
                           timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "nogit"


def new_run_dir(experiment: str) -> Path:
    """A fresh, uniquely named directory under results/<experiment>/."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = repo_root() / "results" / experiment / f"{stamp}_{_git_sha()}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _jsonable(obj: Any):
    """Best-effort conversion; a manifest must never fail to save.

    NON-FINITE FLOATS BECOME null. `json.dumps` writes bare `NaN` and
    `Infinity` tokens, which Python reads back happily and which are not
    valid JSON -- so an artifact would be readable by the one tool that
    wrote it and rejected by every other, silently, and only when a run
    happened to produce one. Every non-finite these summaries can contain
    means "no value": a median over seeds that never reached competence, a
    closest error where nothing was evaluated. `null` is JSON's word for
    that.

    This is not hypothetical -- the sparc_like fold running now is the one
    most likely to produce them, because its tolerance may be out of reach
    for every arm.
    """
    import numpy as np

    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        v = obj.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return repr(obj)


def save_run(
    directory: Path,
    manifest: dict,
    runs: dict[str, list[RunLog]] | None = None,
    summary: dict | None = None,
) -> Path:
    """Write the manifest, the raw per-shot records, and the summary."""
    directory.mkdir(parents=True, exist_ok=True)

    full = dict(manifest)
    full.setdefault("commit", _git_sha())
    full.setdefault("environment", environment_stamp().replace("  \n", " | "))
    full.setdefault("saved_at", _dt.datetime.now().isoformat(timespec="seconds"))
    (directory / "manifest.json").write_text(
        json.dumps(_jsonable(full), indent=2, allow_nan=False), encoding="utf-8")

    if runs is not None:
        payload = {
            arm: [r.to_dict() for r in logs] for arm, logs in runs.items()
        }
        (directory / "runs.json").write_text(
            json.dumps(_jsonable(payload), indent=1, allow_nan=False), encoding="utf-8")

    if summary is not None:
        (directory / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")

    return directory
