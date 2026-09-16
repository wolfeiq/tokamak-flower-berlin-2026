"""Figures: they render, and they do not silently hide missing data."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from hfmarl.experiments.conditions import CONDITION_ORDER, LABELS  # noqa: E402
from hfmarl.metrics import plots  # noqa: E402
from hfmarl.metrics.log import ExperimentLog, RunLog, ShotRecord  # noqa: E402


@pytest.fixture
def log():
    rng = np.random.default_rng(0)
    e = ExperimentLog("test")
    rates = {"isolated": 120.0, "fedbuff_uniform": 80.0, "fedbuff_similarity": 50.0,
             "fedavg_naive": 160.0, "centralised": 40.0}
    for cond, rate in rates.items():
        for dev in ("iter_like", "tcv_like"):
            for seed in range(3):
                r = RunLog(cond, dev, seed)
                for i in range(200):
                    r.add(ShotRecord(
                        shot=i, reward=float(-10 * np.exp(-i / rate) + rng.normal(0, .3)),
                        steps=20,
                        violations=("beta_N",) if rng.random() < .4 * np.exp(-i / 40) else (),
                    ))
                e.add(r)
    return e


@pytest.fixture(autouse=True)
def close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def test_every_condition_appears_in_the_legend(log):
    fig = plots.plot_learning_curves(log, "iter_like")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert set(labels) == {LABELS[c] for c in CONDITION_ORDER}


def test_missing_condition_is_annotated_not_hidden(log):
    """A silently dropped condition is a misleading figure, not a partial one."""
    for seed in range(3):
        del log.runs[("centralised", "iter_like", seed)]
    fig = plots.plot_learning_curves(log, "iter_like")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("missing" in t and "centralised" in t for t in texts)


def test_learning_curve_annotates_the_role_matching_check(log):
    fig = plots.plot_learning_curves(log, "iter_like")
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "role" in texts.lower()


def test_shots_to_threshold_renders_a_bar_per_device_and_condition(log):
    fig = plots.plot_shots_to_threshold(log, threshold=-2.0)
    ax = fig.axes[0]
    assert len(ax.patches) == len(CONDITION_ORDER) * len(log.devices)
    assert [t.get_text() for t in ax.get_xticklabels()] == log.devices


def test_censored_bars_are_hatched_and_counted():
    """Never-reached seeds must be visibly distinct, never averaged in."""
    e = ExperimentLog("censored")
    for cond in ("isolated", "fedbuff_similarity"):
        for seed in range(3):
            r = RunLog(cond, "iter_like", seed)
            # isolated never converges; the method does.
            for i in range(100):
                val = -10.0 if cond == "isolated" else float(-10 * np.exp(-i / 20))
                r.add(ShotRecord(shot=i, reward=val, steps=20))
            e.add(r)
    fig = plots.plot_shots_to_threshold(e, threshold=-2.0)
    ax = fig.axes[0]
    assert any(p.get_hatch() for p in ax.patches)
    assert any("censored" in t.get_text() for t in ax.texts)


def test_violations_figure_renders(log):
    fig = plots.plot_violations(log, "iter_like")
    assert fig.axes[0].get_ylabel().startswith("cumulative")


def test_catastrophe_counts_crossings_correctly():
    iso = [np.linspace(2.0, 3.4, 30) for _ in range(4)]   # all cross 3.0
    fed = [np.linspace(2.0, 2.4, 30) for _ in range(5)]   # none do
    fig = plots.plot_catastrophe(iso, fed, limit_value=3.0, soft_value=2.5)
    txt = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "isolated 4/4" in txt and "federated 0/5" in txt


def test_catastrophe_reports_federated_crossings_honestly():
    iso = [np.linspace(2.0, 3.4, 30) for _ in range(3)]
    fed = [np.linspace(2.0, 3.2, 30) for _ in range(3)]  # these cross too
    fig = plots.plot_catastrophe(iso, fed, limit_value=3.0)
    txt = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "federated 3/3" in txt


def test_similarity_ablation_detects_decay():
    d = np.array([0.5, 1.0, 1.5, 2.0])
    b = 2.5 * np.exp(-d)
    fig = plots.plot_similarity_ablation(d, b)
    txt = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "decays with distance" in txt


def test_similarity_ablation_flags_absence_of_decay():
    """A flat relationship means the metric is not capturing transferability."""
    d = np.array([0.5, 1.0, 1.5, 2.0])
    b = np.array([1.0, 1.05, 0.98, 1.2])  # rising slightly
    fig = plots.plot_similarity_ablation(d, b)
    txt = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "unjustified" in txt


def test_cold_start_marks_threshold_crossings(log):
    fig = plots.plot_cold_start(log, "tcv_like", threshold=-2.0)
    assert len(fig.axes[0].lines) >= 2


def test_staleness_renders():
    lags = np.array([0, 5, 10, 20])
    fig = plots.plot_staleness(lags, np.array([-0.4, -0.5, -0.6, -0.8]))
    assert fig.axes[0].get_xlabel().startswith("update lag")


def test_colours_are_consistent_across_figures():
    """The same condition must be the same colour in every figure."""
    assert set(plots.COLOURS) == set(CONDITION_ORDER)
    assert len(set(plots.COLOURS.values())) == len(CONDITION_ORDER)


def test_method_is_drawn_most_prominently():
    assert (plots.STYLES["fedbuff_similarity"]["lw"]
            > plots.STYLES["isolated"]["lw"])


def test_upper_bound_is_dashed_so_it_is_not_read_as_achievable():
    assert plots.STYLES["centralised"].get("ls") == "--"
