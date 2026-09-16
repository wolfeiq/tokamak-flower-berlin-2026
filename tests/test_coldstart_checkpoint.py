"""A failed long experiment must retain complete seeds and resume exactly."""
import json
import sys

import pytest

from hfmarl.metrics.log import RunLog, ShotRecord
from scripts import exp_coldstart as experiment


def test_interrupted_run_resumes_completed_seeds_without_repeating_them(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(experiment, "new_run_dir", lambda _: output)
    monkeypatch.setattr(experiment, "snapshot_source", lambda _: {})
    # Keep the real artifact writer, but no simulator import for its stamp.
    monkeypatch.setattr("hfmarl.util.artifacts.environment_stamp", lambda: "test")
    calls = []
    fail = True

    def run(joiner, incumbents, task, seed, **kwargs):
        calls.append(seed)
        if fail and seed == 1:
            raise RuntimeError("simulated interruption")
        log = RunLog(kwargs["arm"], joiner, seed)
        for shot in range(kwargs["join_shots"]):
            log.add(ShotRecord(shot, 0.0, 30, beta_error=0.0, is_evaluation=True))
        return log, {"received_weights": {}, "handover_adopted": None}

    monkeypatch.setattr(experiment, "run_cold_start", run)
    argv = ["exp_coldstart.py", "--joiners", "tcv_like", "--arms", "scratch",
            "--seeds", "2", "--join", "4", "--no-record"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.main()
    saved = json.loads((output / "checkpoint.json").read_text())
    assert saved["manifest"]["status"] == "running"
    assert [r["seed"] for r in saved["runs"]["tcv_like:scratch"]] == [0]
    # A killed write of the analysis view cannot break the canonical checkpoint.
    (output / "runs.json").write_text("{incomplete")
    fail = False
    monkeypatch.setattr(sys, "argv", argv + ["--resume", str(output)])
    assert experiment.main() == 0
    assert calls == [0, 1, 1]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["results"]["tcv_like"]["competence"]["scratch"]["reach_rate"] == 1.0


def test_resume_refuses_changed_settings_or_code(tmp_path, monkeypatch):
    manifest = {"seeds": 2, "status": "running",
                "source_sha256": {"scripts/exp_coldstart.py": "wrong hash"}}
    monkeypatch.setattr(experiment, "save_run", lambda *args, **kwargs: None)
    experiment.save_checkpoint(tmp_path, manifest, {}, {})
    with pytest.raises(ValueError, match="seeds differs"):
        experiment.load_checkpoint(tmp_path, {"seeds": 3})
    with pytest.raises(ValueError, match="source changed"):
        experiment.load_checkpoint(tmp_path, {"seeds": 2})
