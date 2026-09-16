"""Site-side, aggregate-only reader of atomically saved cold-start checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

ARMS = (
    "scratch",
    "single_source",
    "handover_merge",
    "federated_uniform",
    "federated_similarity",
)


def child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
        raise ValueError("Invalid study identifier")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Study must remain within the configured results directory")
    return result


def read_checkpoint(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Checkpoint exceeds the 64 MiB analysis limit")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not all(k in payload for k in ("manifest", "runs", "run_metadata")):
        raise ValueError("A current atomic cold-start checkpoint is required")
    return payload, hashlib.sha256(raw).hexdigest()


def list_studies(root: Path) -> dict:
    studies = []
    if root.exists():
        for folder in sorted(root.iterdir(), reverse=True):
            if len(studies) >= 30:
                break
            if not folder.is_dir() or not (folder / "checkpoint.json").is_file():
                continue
            if not folder.resolve().is_relative_to(root.resolve()):
                continue
            if (
                not (folder / "checkpoint.json")
                .resolve()
                .is_relative_to(root.resolve())
            ):
                continue
            try:
                payload, digest = read_checkpoint(folder / "checkpoint.json")
                m = payload["manifest"]
                studies.append(
                    {
                        "study": folder.name,
                        "status": m.get("status"),
                        "task": m.get("task"),
                        "sha256": digest,
                        "completed_runs": sum(map(len, payload["runs"].values())),
                    }
                )
            except (ValueError, OSError):
                studies.append({"study": folder.name, "status": "unreadable"})
    return {"studies": studies, "limit": 30}


def competence(
    run: dict, tolerance: float, steps: int, selection_end: int = 0
) -> int | None:
    streak = 0
    for s in run["shots"]:
        if (
            not s.get("is_evaluation", False)
            or not s.get("evaluation_eligible", True)
            or s["shot"] < selection_end
        ):
            continue
        error = s.get("beta_error")
        ok = (
            isinstance(error, (float, int))
            and math.isfinite(error)
            and error <= tolerance
            and not s.get("violations")
            and not s.get("terminated_early")
            and s["steps"] == steps
        )
        streak = streak + 1 if ok else 0
        if streak == 2:
            return s["shot"] + 1
    return None


def summary(payload: dict, digest: str, study: str) -> dict:
    m, runs, metadata = (payload[k] for k in ("manifest", "runs", "run_metadata"))
    if "joint_competence" not in m.get("metric", ""):
        raise ValueError("Legacy endpoint unsupported; use a current checkpoint")
    expected, budget, steps = m["seeds"], m["join"], m["expected_steps"]
    rows, pairs = [], []
    for device in m["joiners"]:
        times = {}
        for arm in m["arms"]:
            key = f"{device}:{arm}"
            logs = runs.get(key, [])
            base, strict, costs, adopted = {}, {}, [], []
            missing_meta = False
            for run in logs:
                seed = run["seed"]
                if seed in base or len(run["shots"]) != budget:
                    raise ValueError("Duplicate seed or incomplete seed budget")
                if [s["shot"] for s in run["shots"]] != list(range(budget)):
                    raise ValueError("Nonsequential shot accounting")
                meta = metadata.get(key, {}).get(str(seed))
                base[seed] = competence(run, m["tolerances"][device], steps)
                if meta is None:
                    missing_meta = True
                    continue
                end = (
                    meta["joiner_probe_shots"] + meta["joiner_selection_shots"]
                    if meta["joiner_selection_shots"]
                    else 0
                )
                strict[seed] = competence(run, m["tolerances"][device], steps, end)
                costs.append(meta["source_shots"])
                adopted.append(meta.get("handover_adopted"))
            reached = [v for v in base.values() if v is not None]
            confirmed = [v for v in strict.values() if v is not None]
            rows.append(
                {
                    "device": device,
                    "arm": arm,
                    "completed_seeds": len(logs),
                    "expected_seeds": expected,
                    "reached": len(reached),
                    "censored": len(logs) - len(reached),
                    "median_joiner_shots_among_reached": median(reached)
                    if reached
                    else None,
                    "independent_confirmation_median": (
                        median(confirmed) if confirmed and not missing_meta else None
                    ),
                    "independent_confirmation_reached": (
                        len(confirmed) if not missing_meta else None
                    ),
                    "median_source_shots": median(costs) if costs else None,
                    "handover_adopted": sum(v is True for v in adopted),
                    "metadata_complete": not missing_meta,
                }
            )
            times[arm] = (base, strict)
        scratch = times.get("scratch", ({}, {}))
        for arm, measures in times.items():
            if arm == "scratch":
                continue
            for index, label in enumerate(("recorded", "independent_confirmation")):
                a, b = scratch[index], measures[index]
                seeds = sorted(a.keys() & b.keys())
                joint = [s for s in seeds if a[s] is not None and b[s] is not None]
                pairs.append(
                    {
                        "device": device,
                        "arm": arm,
                        "endpoint": label,
                        "paired_seeds": len(seeds),
                        "both_reached": len(joint),
                        "faster": sum(b[s] < a[s] for s in joint),
                        "tied": sum(b[s] == a[s] for s in joint),
                        "slower": sum(b[s] > a[s] for s in joint),
                        "method_only_reached": sum(
                            a[s] is None and b[s] is not None for s in seeds
                        ),
                        "scratch_only_reached": sum(
                            a[s] is not None and b[s] is None for s in seeds
                        ),
                    }
                )
    complete = all(r["completed_seeds"] == expected for r in rows)
    return {
        "study": study,
        "evidence_id": f"checkpoint:{digest}",
        "kind": "TORAX simulation; not real facility measurements",
        "status": m.get("status", "unknown"),
        "coverage_complete": complete,
        "task": m["task"],
        "joiner_budget": budget,
        "rows": rows,
        "paired_against_scratch": pairs,
        "limitations": [
            "Partial arms must not be compared as complete experiments.",
            (
                "Independent confirmation excludes selection trials, retains their cost, "
                "and changes confirmation timing. It is a sensitivity analysis."
            ),
            "Timing medians condition on success; report censoring and paired reach too.",
            "Source training and cached calibration cost are additional to joiner shots.",
            "No pooled-data baseline here; no causal proof of federation superiority.",
            (
                "Thermal transport simulation does not establish disruption avoidance, "
                "real-reactor performance, or Flower Agent effectiveness."
            ),
        ],
    }


def inspect_study(root: Path, study: str) -> dict:
    path = child(root, study) / "checkpoint.json"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Checkpoint escapes configured directory")
    payload, digest = read_checkpoint(path)
    return summary(payload, digest, study)
