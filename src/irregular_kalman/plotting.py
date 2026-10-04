"""Static, reproducible scientific plots with explicit uncertainty semantics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import MODELS, Z95


def make_plots(report: dict, demo: dict, output: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        _performance(report, output)
        _trajectory(demo, output)


def _performance(report, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), layout="constrained")
    panels = [
        ("level_rmse_missing", "Level RMSE on missing rows (lower is better)", None),
        ("coverage95_missing", "95% level-band coverage on missing rows", 0.95),
        ("innovation_nis_mean", "Mean squared standardized innovation", 1.0),
        ("predictive_nll_per_observation", "Predictive negative log density (lower is better)", None),
    ]
    x = np.arange(len(MODELS))
    for ax, (metric, title, target) in zip(axes.flat, panels):
        for offset, scenario, label, color in (
            (-0.17, "random_missing", "Random missingness", "#27658a"),
            (0.17, "block_outage", "+ 40-row outage", "#d07b37"),
        ):
            selected = {row["model"]: row["metrics"][metric] for row in report["summary"] if row["scenario"] == scenario}
            means = [selected[model]["mean"] for model in MODELS]
            errors = [2 * selected[model]["mcse"] for model in MODELS]
            ax.bar(x + offset, means, 0.32, yerr=errors, capsize=3, color=color, label=label)
        if target is not None:
            ax.axhline(target, color="#343c43", ls="--", lw=1)
        ax.set_xticks(x, ["Oracle clock", "Fitted clock", "Fitted unit clock"])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"{report['repetitions']} independent test paths; bars ±2 Monte Carlo SE\nConditional on one fixed training path; all three methods are online", fontsize=13)
    fig.savefig(output / "performance.png", dpi=160)
    plt.close(fig)


def _trajectory(demo, output):
    times = np.array(demo["times"])
    truth = np.array(demo["true_level"])
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=True, layout="constrained")
    start, end = demo["block_indices"]
    for level_key, sd_key, label, color, style in (
        ("filtered_level", "filtered_sd", "Filter: online", "#27658a", "-"),
        ("smoothed_level_offline", "smoothed_sd_offline", "RTS smoother: offline, uses future data", "#bd6831", "--"),
    ):
        error = np.array(demo[level_key]) - truth
        sd = np.array(demo[sd_key])
        axes[0].plot(times, error, style, color=color, label=label, lw=1.2)
        axes[0].fill_between(times, error - Z95 * sd, error + Z95 * sd, color=color, alpha=0.13)
        axes[1].plot(times, 2 * Z95 * sd, style, color=color, label=label)
    for ax in axes:
        ax.axvspan(times[start], times[end], color="#555b63", alpha=0.1)
        ax.grid(alpha=0.2)
    axes[0].axhline(0, color="#252c31", lw=1)
    axes[0].set(ylabel="Estimate minus latent truth", title="Same fitted model, different information sets")
    axes[0].legend(fontsize=9, loc="best")
    axes[1].set(xlabel="Actual elapsed time", ylabel="95% level-band width", title="Shaded region: 40 consecutive missing observations")
    fig.suptitle("A fixed independent demonstration path — not a selected best case", fontsize=13)
    fig.savefig(output / "outage_and_smoothing.png", dpi=160)
    plt.close(fig)
