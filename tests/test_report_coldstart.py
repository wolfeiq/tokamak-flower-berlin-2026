"""The fold report, which reads artifacts rather than re-running anything.

Two of my claims died today to a rescore of saved runs: the plateau was
measured on candidate shots, and a "replication" had changed two settings at
once. Recomputing from `runs.json` costs seconds where re-running a fold costs
two hours, so the analysis is a script and the experiment's printed table is a
convenience.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_coldstart as rc  # noqa: E402

from hfmarl.metrics.log import RunLog, ShotRecord  # noqa: E402


def _fold(tmp_path, arms, tol=0.2, joiner="tcv_like"):
    d = tmp_path / "20260916-000000_abc"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "experiment": "coldstart", "task": "moderate", "joiners": [joiner],
        "seeds": 2, "join": 40, "eval_every": 5, "commit": "abc",
        "tolerances": {joiner: tol},
    }), encoding="utf-8")
    payload = {}
    for arm, errs in arms.items():
        log = RunLog(condition=arm, device=joiner, seed=0)
        for i, e in enumerate(errs):
            log.add(ShotRecord(shot=i, reward=-e, beta_error=e, steps=30,
                               is_evaluation=True))
        payload[f"{joiner}:{arm}"] = [log.to_dict()]
    (d / "runs.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_the_arm_comes_from_the_key_not_the_logged_condition(tmp_path):
    """Every saved log once carried `cold_inherit_similarity` as its
    condition, whatever arm produced it. The key is the reliable side."""
    d = _fold(tmp_path, {"scratch": [0.5, 0.1, 0.1],
                         "federated_uniform": [0.9, 0.9, 0.9]})
    man, runs = rc.load(d)
    assert set(runs["tcv_like"]) == {"scratch", "federated_uniform"}
    assert man["task"] == "moderate"


def test_a_half_written_fold_does_not_stop_the_others(tmp_path, capsys):
    good = _fold(tmp_path, {"scratch": [0.1, 0.1, 0.1]})
    bad = tmp_path / "20260916-999999_def"
    bad.mkdir()
    (bad / "runs.json").write_text("{not json", encoding="utf-8")
    sys.argv = ["report_coldstart.py", "--root", str(tmp_path)]
    assert rc.main() == 0
    out = capsys.readouterr().out
    assert good.name in out
    assert "could not read" in out
    assert bad.name in out


def test_no_completed_folds_is_an_error_not_an_empty_report(tmp_path, capsys):
    sys.argv = ["report_coldstart.py", "--root", str(tmp_path)]
    assert rc.main() == 1
    assert "no completed folds" in capsys.readouterr().out


def _conv_dir(tmp_path, task, transfer=True):
    d = tmp_path / f"20260916-000000_{task}"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"task": task}),
                                     encoding="utf-8")
    summ = {"rows": [{"device": "sparc_like", "joint_success": 1.0}]}
    if transfer:
        summ["transfer"] = [{"device": "sparc_like", "own_kp": 0.05,
                             "transferred_kp": 0.6,
                             "joint_transferred": 0.0}]
    (d / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    return d


def test_the_conventional_reference_is_read_not_transcribed(tmp_path):
    """A literal dict copied out of a run's output is a number with no date.

    Re-measure the sweep with a different gain grid and the constant keeps
    reporting the old answer, in a file whose whole purpose is to recompute
    from artifacts rather than trust a remembered value.
    """
    _conv_dir(tmp_path, "moderate")
    table = rc.conventional_reference("moderate", str(tmp_path))
    assert table["sparc_like"]["tuned"] == 1.0
    assert table["sparc_like"]["transferred"] == 0.0
    assert table["sparc_like"]["own_kp"] == 0.05
    assert table["sparc_like"]["transferred_kp"] == 0.6
    assert table["_source"].endswith("moderate")
    assert not hasattr(rc, "CONVENTIONAL_MODERATE")


def test_a_sweep_for_another_task_is_not_borrowed(tmp_path):
    """brink's gains are not moderate's, and a reference from the wrong task
    is worse than none."""
    _conv_dir(tmp_path, "brink")
    assert rc.conventional_reference("moderate", str(tmp_path)) == {}


def test_a_sweep_without_a_transfer_column_cannot_answer_the_question(tmp_path):
    """A single-gain run has no transferred number, and reporting its tuned
    number alone would restate the claim the transfer column corrected."""
    _conv_dir(tmp_path, "moderate", transfer=False)
    assert rc.conventional_reference("moderate", str(tmp_path)) == {}


@pytest.mark.parametrize("arm", rc.ARMS)
def test_the_arm_order_is_the_protocol_ladder(arm):
    assert rc.ARMS.index("scratch") == 0
    assert rc.ARMS.index("federated_similarity") == len(rc.ARMS) - 1
    assert arm in {"scratch", "single_source", "handover_merge",
                   "federated_uniform", "federated_similarity"}


def test_a_summary_with_no_reaching_seed_still_writes_valid_json(tmp_path):
    """The sparc_like fold is the one most likely to produce this.

    If no seed reaches the tolerance, the median is NaN, the closest error
    can be NaN, and the payload carries them straight into save_run. Bare
    NaN would make the artifact unreadable by anything but Python -- in a
    file whose entire purpose is to be re-analysed later.
    """
    import json

    from hfmarl.util.artifacts import save_run

    d = tmp_path / "fold"
    save_run(d, manifest={"task": "moderate"},
             summary={"competence": {"scratch": float("nan")},
                      "closest_evaluated_error": {"scratch": float("nan")},
                      "plateau": {"scratch": float("-inf")}})
    text = (d / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(c)))
    assert parsed["competence"]["scratch"] is None
    assert parsed["plateau"]["scratch"] is None
