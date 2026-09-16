"""Shot accounting and experiment bookkeeping."""

import pytest

from hfmarl.experiments.conditions import CONDITION_ORDER, CONDITIONS, describe, get
from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord


def _run(cond="isolated", dev="iter_like", seed=0, n=10):
    r = RunLog(cond, dev, seed)
    for i in range(n):
        r.add(ShotRecord(shot=i, reward=float(-i), steps=20,
                         violations=("beta_N",) if i < 3 else ()))
    return r


def test_duplicate_run_is_refused():
    """Silently overwriting a run would lose data and skew a seed count."""
    e = ExperimentLog()
    e.add(_run())
    with pytest.raises(ValueError, match="duplicate"):
        e.add(_run())


def test_select_filters_by_condition_and_device():
    e = ExperimentLog()
    for cond in ("isolated", "centralised"):
        for dev in ("iter_like", "tcv_like"):
            e.add(_run(cond, dev, 0))
    assert len(e.select("isolated")) == 2
    assert len(e.select(device="tcv_like")) == 2
    assert len(e.select("isolated", "tcv_like")) == 1


def test_coverage_warns_on_ragged_seed_counts():
    """Unequal seeds make a comparison unfair without looking unfair."""
    e = ExperimentLog()
    e.add(_run("isolated", "iter_like", 0))
    e.add(_run("isolated", "iter_like", 1))
    e.add(_run("centralised", "iter_like", 0))
    assert "WARNING" in e.coverage()


def test_coverage_is_clean_when_balanced():
    e = ExperimentLog()
    for cond in ("isolated", "centralised"):
        for seed in (0, 1):
            e.add(_run(cond, "iter_like", seed))
    assert "WARNING" not in e.coverage()


def test_roundtrip_through_disk(tmp_path):
    e = ExperimentLog("exp")
    e.add(_run("isolated", "iter_like", 0))
    e.add(_run("centralised", "tcv_like", 1))
    p = e.save(tmp_path / "e.json")
    back = ExperimentLog.load(p)
    assert back.name == "exp"
    assert set(back.runs) == set(e.runs)
    orig = e.get("isolated", "iter_like", 0)
    load = back.get("isolated", "iter_like", 0)
    assert load.rewards().tolist() == orig.rewards().tolist()
    assert load.shots[0].violations == ("beta_N",)


def test_condition_registry_matches_plot_order():
    assert set(CONDITIONS) == set(CONDITION_ORDER)


def test_exactly_one_method_and_one_upper_bound():
    roles = [c.role for c in CONDITIONS.values()]
    assert roles.count("method") == 1
    assert roles.count("upper bound") == 1
    assert roles.count("negative control") == 1


def test_naive_fedavg_is_marked_as_the_negative_control():
    assert get("fedavg_naive").role == "negative control"
    assert not get("fedavg_naive").role_matched


def test_only_the_method_uses_similarity_weighting():
    using = [n for n, c in CONDITIONS.items() if c.use_similarity]
    assert using == ["fedbuff_similarity"]


def test_only_centralised_pools_data():
    pooling = [n for n, c in CONDITIONS.items() if c.pooled_data]
    assert pooling == ["centralised"]


def test_describe_mentions_every_condition():
    text = describe()
    for label in CONDITION_ORDER:
        assert label.replace("_", " ") in text.replace("_", " ") or label in text


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        get("magic")


def test_a_saved_log_is_valid_json_even_with_a_failed_shot(tmp_path):
    """Bare `NaN` is not JSON.

    Python writes it and reads it back happily, so an artifact containing one
    is readable by exactly the tool that produced it and rejected by every
    other -- silently, and only when a run happened to fail a solve. A
    solver-failed shot records a NaN tracking error, so this is reachable in
    any run, and the folds running now are the ones most likely to hit it.
    """
    import json

    from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord

    log = ExperimentLog("x")
    r = RunLog(condition="c", device="d", seed=0)
    r.add(ShotRecord(shot=0, reward=float("nan"),
                     beta_error=float("nan"), steps=0))
    log.add(r)
    p = log.save(tmp_path / "e.json")

    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    # strict: reject the non-standard constants the way another tool would
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"non-standard token {c}")))


def test_reloading_brings_the_nan_back(tmp_path):
    """Lossless in the direction that matters: a null is a missing number,
    not a zero, and scoring it as zero would make a failed shot look like a
    perfect one."""
    import math

    from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord

    log = ExperimentLog("x")
    r = RunLog(condition="c", device="d", seed=0)
    r.add(ShotRecord(shot=0, reward=-1.0, beta_error=float("nan"), steps=3))
    log.add(r)
    p = log.save(tmp_path / "e.json")

    back = ExperimentLog.load(p)
    shot = next(iter(back.runs.values())).shots[0]
    assert math.isnan(shot.beta_error)
    assert shot.reward == -1.0
