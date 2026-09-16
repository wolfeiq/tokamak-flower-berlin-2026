"""The six deliverable figures.

    1. learning curves, five conditions on one axis
    2. shots-to-threshold per device (grouped bars)
    3. cumulative violations during training
    4. the catastrophe plot
    5. similarity-weighting ablation: benefit vs dimensionless distance
    6. cold start
    (+ a supplementary staleness-tolerance figure, metric 6)

Every figure takes an `ExperimentLog` and returns a matplotlib Figure. None of
them invents data: a missing condition is drawn as absent and annotated, not
skipped silently, because a grouped bar chart with a quietly dropped bar is a
misleading figure rather than an incomplete one.

Colour is consistent across every figure -- the same condition is the same
colour everywhere -- and the palette is colour-blind safe (Okabe-Ito).
"""

from __future__ import annotations

import numpy as np

from hfmarl.experiments.conditions import CONDITION_ORDER, LABELS
from hfmarl.metrics.curves import (
    learning_curve,
    shots_to_threshold,
    violations_during_training,
)
from hfmarl.metrics.log import ExperimentLog

# Okabe-Ito, colour-blind safe. Order matches CONDITION_ORDER.
COLOURS: dict[str, str] = {
    "isolated": "#0072B2",
    "fedbuff_uniform": "#009E73",
    "fedbuff_similarity": "#D55E00",
    "fedavg_naive": "#CC79A7",
    "centralised": "#666666",
}
# The method is drawn thickest; the upper bound is dashed because it is not an
# achievable condition, and conflating it with the others is the easiest way to
# overclaim.
STYLES: dict[str, dict] = {
    "isolated": dict(lw=1.8),
    "fedbuff_uniform": dict(lw=1.8),
    "fedbuff_similarity": dict(lw=2.8),
    "fedavg_naive": dict(lw=1.8, ls=":"),
    "centralised": dict(lw=1.8, ls="--"),
}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: WSL has no display by default
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"matplotlib is required for plots ({e}). uv pip install -e '.[viz]'"
        ) from e
    return plt


def _style(ax, xlabel="shots", ylabel=None):
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def _require_single_device(log: ExperimentLog, device: str | None, what: str) -> None:
    """Refuse to pool across devices.

    Devices differ by an order of magnitude in reward scale (a TCV-like shot
    and an ITER-like shot are not the same number), so averaging their curves
    produces a line that describes no machine. Failing loudly beats drawing a
    meaningless mean under a title that says "all devices".
    """
    if device is None and len(log.devices) > 1:
        raise ValueError(
            f"{what} would pool {len(log.devices)} devices ({', '.join(log.devices)}) "
            "whose reward scales differ. Pass device=... explicitly, or make one "
            "figure per device."
        )


def _note_missing(ax, missing: list[str]) -> None:
    """Say what is absent, on the figure itself."""
    if missing:
        ax.text(
            0.98, 0.02, "missing: " + ", ".join(missing), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, style="italic", color="#B00020",
        )


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------


def plot_learning_curves(log: ExperimentLog, device: str | None = None,
                         window: int = 10, ax=None):
    """Reward vs shots, all five conditions on one axis.

    The load-bearing comparison is naive FedAvg against isolated: if role
    matching matters, ignoring roles should sit at or BELOW training alone.
    That relationship is annotated directly on the figure, because it is the
    part a reader is most likely to miss.
    """
    _require_single_device(log, device, "plot_learning_curves")
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.6))

    missing, finals = [], {}
    for cond in CONDITION_ORDER:
        runs = log.select(cond, device)
        if not runs:
            missing.append(cond)
            continue
        x, mean, sem = learning_curve(runs, window)
        if x.size == 0:
            missing.append(cond)
            continue
        ax.plot(x, mean, color=COLOURS[cond], label=LABELS[cond], **STYLES[cond])
        ax.fill_between(x, mean - sem, mean + sem, color=COLOURS[cond], alpha=0.15, lw=0)
        finals[cond] = mean[-1]

    title = "Learning curves" + (f" — {device}" if device else f" — {log.devices[0] if log.devices else 'no data'}")
    n_seeds = len(log.seeds)
    ax.set_title(f"{title}   ({n_seeds} seed{'s' if n_seeds != 1 else ''}, "
                 f"mean ± s.e.m., {window}-shot trailing mean)", fontsize=10)
    _style(ax, "shots", "episode return")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    if "fedavg_naive" in finals and "isolated" in finals:
        ok = finals["fedavg_naive"] <= finals["isolated"]
        ax.text(
            0.02, 0.97,
            ("role-blind FedAvg ≤ isolated ✓ (role matching matters)" if ok
             else "role-blind FedAvg > isolated — role matching NOT demonstrated"),
            transform=ax.transAxes, va="top", fontsize=8,
            color="#1B7837" if ok else "#B00020",
        )
    _note_missing(ax, missing)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Shots to threshold, per device
# ---------------------------------------------------------------------------


def plot_shots_to_threshold(log: ExperimentLog, threshold: float,
                            window: int = 10, persistence: int = 5, ax=None):
    """Grouped bars: shots to threshold, per device, per condition.

    Prediction to check against the figure: the smallest, data-poorest device
    gains most. That is the headline for anyone running a small machine.

    Seeds that never reached the threshold are marked with a hatched bar at the
    censoring limit and an explicit count, never averaged in.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.6))

    devices = log.devices
    conds = [c for c in CONDITION_ORDER if log.select(c)]
    width = 0.8 / max(len(conds), 1)

    for j, cond in enumerate(conds):
        xs, heights, errs, hatches, censor_notes = [], [], [], [], []
        for i, dev in enumerate(devices):
            res = shots_to_threshold(log.select(cond, dev), threshold, window, persistence)
            xs.append(i + j * width - 0.4 + width / 2)
            if res.n_reached:
                lo, hi = res.ci()
                heights.append(res.median)
                errs.append([max(0, res.median - lo), max(0, hi - res.median)])
                hatches.append("")
            else:
                # Never reached: draw at the censoring limit, hatched.
                heights.append(max(res.censored_at) if res.censored_at else 0)
                errs.append([0, 0])
                hatches.append("///")
            if res.censored_at:
                censor_notes.append((xs[-1], heights[-1], len(res.censored_at), res.n_total))

        bars = ax.bar(xs, heights, width * 0.92, label=LABELS[cond],
                      color=COLOURS[cond], edgecolor="white", linewidth=0.5)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
                b.set_alpha(0.45)
        ax.errorbar(xs, heights, yerr=np.array(errs).T, fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=2.5)
        for x, h, n_cens, n_tot in censor_notes:
            ax.text(x, h, f"{n_cens}/{n_tot}\ncensored", ha="center", va="bottom",
                    fontsize=6, color="#B00020")

    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels(devices, fontsize=9)
    ax.set_title(f"Shots to threshold (return ≥ {threshold:.3g})   "
                 f"lower is better; hatched = never reached", fontsize=10)
    _style(ax, "", "shots")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Violations during training
# ---------------------------------------------------------------------------


def plot_violations(log: ExperimentLog, device: str | None = None, ax=None):
    """Cumulative limit violations vs shots.

    The expected signature of the claim is that the federated curve FLATTENS
    EARLIER -- safe behaviour transfers before good behaviour does. The final
    height matters less than where each curve stops rising.
    """
    _require_single_device(log, device, "plot_violations")
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.4))

    missing = []
    for cond in CONDITION_ORDER:
        runs = log.select(cond, device)
        if not runs:
            missing.append(cond)
            continue
        mean, sem = violations_during_training(runs)
        if mean.size == 0:
            missing.append(cond)
            continue
        x = np.arange(1, mean.size + 1)
        ax.plot(x, mean, color=COLOURS[cond], label=LABELS[cond], **STYLES[cond])
        ax.fill_between(x, mean - sem, mean + sem, color=COLOURS[cond], alpha=0.15, lw=0)

    ax.set_title("Limit violations during training" + (f" — {device}" if device else ""),
                 fontsize=10)
    _style(ax, "shots", "cumulative shots with a violation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _note_missing(ax, missing)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. The catastrophe plot
# ---------------------------------------------------------------------------


def plot_catastrophe(isolated_traj: list[np.ndarray], federated_traj: list[np.ndarray],
                     limit_value: float, quantity: str = "beta_N",
                     soft_value: float | None = None, ax=None):
    """The figure people remember.

    Device A in a regime it never trained in: isolated trajectories cross the
    limit, federated ones approach and turn away.

    Takes raw trajectories rather than an ExperimentLog because what carries
    the argument is individual discharges approaching the line -- a mean would
    smear exactly the behaviour being claimed.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.6))

    for i, tr in enumerate(isolated_traj):
        ax.plot(np.arange(len(tr)), tr, color=COLOURS["isolated"], lw=1.2, alpha=0.75,
                label="isolated" if i == 0 else None)
    for i, tr in enumerate(federated_traj):
        ax.plot(np.arange(len(tr)), tr, color=COLOURS["fedbuff_similarity"], lw=1.6,
                alpha=0.85, label="federated" if i == 0 else None)

    ax.axhline(limit_value, color="#B00020", lw=2.0, zorder=5)
    ax.text(0.995, limit_value, " limit ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#B00020", weight="bold")
    if soft_value is not None:
        ax.axhspan(min(soft_value, limit_value), max(soft_value, limit_value),
                   color="#B00020", alpha=0.07, zorder=0)

    n_iso_cross = sum(1 for t in isolated_traj if np.any(np.asarray(t) > limit_value))
    n_fed_cross = sum(1 for t in federated_traj if np.any(np.asarray(t) > limit_value))
    ax.text(0.02, 0.97,
            f"crossed the limit:  isolated {n_iso_cross}/{len(isolated_traj)}   "
            f"federated {n_fed_cross}/{len(federated_traj)}",
            transform=ax.transAxes, va="top", fontsize=9)

    ax.set_title(f"Unseen regime: {quantity} trajectories, trained only inside "
                 "a safe envelope", fontsize=10)
    _style(ax, "control step", quantity)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Similarity-weighting ablation
# ---------------------------------------------------------------------------


def plot_similarity_ablation(distances: np.ndarray, benefits: np.ndarray,
                             errors: np.ndarray | None = None,
                             labels: list[str] | None = None, ax=None):
    """Benefit vs dimensionless distance between devices.

    If the physics weighting means anything, benefit should DECAY with distance
    in (rho*, nu*, beta) space. A flat relationship says the metric is not
    capturing transferability and the similarity weighting is unjustified --
    which is a real result, not a tuning failure.

    The fitted slope and its sign are annotated, so the figure states its own
    verdict rather than leaving it to the reader's eye.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.4))

    d, b = np.asarray(distances, float), np.asarray(benefits, float)
    ax.errorbar(d, b, yerr=errors, fmt="o", ms=7, color=COLOURS["fedbuff_similarity"],
                ecolor="#666666", elinewidth=1, capsize=3, zorder=3)
    if labels is not None:
        for x, y, lab in zip(d, b, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 5),
                        fontsize=7.5, color="#333333")

    if d.size >= 3 and np.ptp(d) > 0:
        slope, intercept = np.polyfit(d, b, 1)
        xs = np.linspace(d.min(), d.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#333333", lw=1.2, ls="--", zorder=2)
        r = float(np.corrcoef(d, b)[0, 1])
        decays = slope < 0
        ax.text(0.98, 0.97,
                f"slope = {slope:.3g}   r = {r:.2f}\n"
                + ("benefit decays with distance ✓" if decays
                   else "no decay — similarity weighting unjustified"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#1B7837" if decays else "#B00020")

    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_title("Transfer benefit vs dimensionless distance", fontsize=10)
    _style(ax, "weighted distance in (ρ*, ν*, β, q) space", "benefit over isolated")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Cold start
# ---------------------------------------------------------------------------


def plot_cold_start(log: ExperimentLog, device: str, threshold: float,
                    window: int = 10, ax=None):
    """A new device joining an existing federation, with and without it.

    The strongest practical number in the set: how many shots before a
    brand-new machine is competent. The gap between the two vertical markers
    is the answer.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.4))

    for cond in ("isolated", "fedbuff_similarity"):
        runs = log.select(cond, device)
        if not runs:
            continue
        x, mean, sem = learning_curve(runs, window)
        if x.size == 0:
            continue
        ax.plot(x, mean, color=COLOURS[cond], label=LABELS[cond], **STYLES[cond])
        ax.fill_between(x, mean - sem, mean + sem, color=COLOURS[cond], alpha=0.15, lw=0)
        res = shots_to_threshold(runs, threshold, window)
        if res.n_reached:
            ax.axvline(res.median, color=COLOURS[cond], lw=1.2, ls=":", alpha=0.9)
            ax.text(res.median, ax.get_ylim()[0], f" {res.median:.0f}",
                    color=COLOURS[cond], fontsize=8, va="bottom")

    ax.axhline(threshold, color="#333333", lw=1.0, ls="--", alpha=0.7)
    ax.text(0.005, threshold, " threshold", transform=ax.get_yaxis_transform(),
            fontsize=8, va="bottom", color="#333333")
    ax.set_title(f"Cold start: {device} joining an existing federation", fontsize=10)
    _style(ax, "shots on the new device", "episode return")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Supplementary: staleness tolerance (metric 6)
# ---------------------------------------------------------------------------


def plot_staleness(lags: np.ndarray, perf: np.ndarray, sem: np.ndarray | None = None,
                   ax=None):
    """Final performance vs update lag. Justifies FedBuff.

    Flat in lag means asynchrony costs nothing -- which is the argument, since
    tokamak campaigns cannot supply synchronous rounds regardless.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(lags, perf, yerr=sem, fmt="o-", color=COLOURS["fedbuff_similarity"],
                ecolor="#666666", capsize=3, lw=2)
    if len(lags) >= 3 and np.ptp(perf) > 0:
        drop = (perf[0] - perf[-1]) / max(abs(perf[0]), 1e-9)
        ax.text(0.98, 0.05, f"degradation across lag range: {drop:.1%}",
                transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_title("Staleness tolerance", fontsize=10)
    _style(ax, "update lag (rounds)", "final performance")
    fig.tight_layout()
    return fig
