"""Fit on one independent trajectory; audit frozen models on fresh trajectories."""

import argparse
from dataclasses import asdict
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .core import kalman_filter, rts_smooth
from .fit import fit_noise
from .synthetic import PRIOR_COV, PRIOR_MEAN, TRUE_Q, TRUE_R, simulate

Z95 = 1.959963984540054
MODELS = ("oracle_clock", "fitted_clock", "fitted_unit_clock")


def score(result, truth: np.ndarray, observations: np.ndarray) -> dict:
    missing = np.isnan(observations)
    observed = ~missing
    if missing.sum() == 0 or observed.sum() < 3:
        raise ValueError("scoring needs missing rows and at least three observed rows")
    error = result.filtered_mean[:, 0] - truth[:, 0]
    sd = np.sqrt(result.filtered_cov[:, 0, 0])
    covered = np.abs(error) <= Z95 * sd
    standardized = result.innovations[observed] / np.sqrt(result.innovation_variance[observed])
    return {
        "level_rmse_all": float(np.sqrt(np.mean(error**2))),
        "level_rmse_missing": float(np.sqrt(np.mean(error[missing] ** 2))),
        "coverage95_all": float(covered.mean()),
        "coverage95_missing": float(covered[missing].mean()),
        "mean95_width_missing": float((2 * Z95 * sd[missing]).mean()),
        "innovation_mean": float(standardized.mean()),
        "innovation_nis_mean": float(np.mean(standardized**2)),
        "innovation_acf1": float(np.corrcoef(standardized[:-1], standardized[1:])[0, 1]),
        "predictive_nll_per_observation": float(-result.log_likelihood / observed.sum()),
    }


def mean_and_mcse(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("summaries need at least two finite independent trajectory values")
    return {"mean": float(values.mean()), "mcse": float(values.std(ddof=1) / np.sqrt(values.size))}


def demonstration(seed: int, fitted_q: float, fitted_r: float) -> dict:
    data = simulate([seed, 2])
    observations = data.observations((90, 130))
    filtered = kalman_filter(data.times, observations, fitted_q, fitted_r, PRIOR_MEAN, PRIOR_COV)
    smoothed = rts_smooth(filtered)
    changed = observations.copy()
    cutoff = 150
    changed[cutoff:] += 10  # Only future finite observations change; NaNs stay missing.
    revised = kalman_filter(data.times, changed, fitted_q, fitted_r, PRIOR_MEAN, PRIOR_COV)
    revised_smooth = rts_smooth(revised)
    causality = {
        "future_edit_starts_at_index": cutoff,
        "future_observation_shift": 10,
        "filter_prefix_max_change": float(np.max(np.abs(filtered.filtered_mean[:cutoff] - revised.filtered_mean[:cutoff]))),
        "smoother_prefix_max_change": float(np.max(np.abs(smoothed.mean[:cutoff] - revised_smooth.mean[:cutoff]))),
    }
    if causality["filter_prefix_max_change"] != 0 or causality["smoother_prefix_max_change"] <= 0:
        raise ArithmeticError("filter/smoother causality demonstration failed")
    return {
        "times": data.times.tolist(),
        "true_level": data.states[:, 0].tolist(),
        "observations": [None if np.isnan(y) else float(y) for y in observations],
        "filtered_level": filtered.filtered_mean[:, 0].tolist(),
        "filtered_sd": np.sqrt(filtered.filtered_cov[:, 0, 0]).tolist(),
        "smoothed_level_offline": smoothed.mean[:, 0].tolist(),
        "smoothed_sd_offline": np.sqrt(smoothed.covariance[:, 0, 0]).tolist(),
        "block_indices": [90, 130],
        "causality": causality,
    }


def run_experiment(repetitions: int = 80, seed: int = 20261004, training_length: int = 400):
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    training = simulate([seed, 0], n=training_length)
    training_y = training.observations()
    real_fit = fit_noise(training.times, training_y, PRIOR_MEAN, PRIOR_COV)
    unit_fit = fit_noise(np.arange(training_length), training_y, PRIOR_MEAN, PRIOR_COV)
    parameters = {
        "oracle_clock": (TRUE_Q, TRUE_R),
        "fitted_clock": (real_fit.q, real_fit.r),
        "fitted_unit_clock": (unit_fit.q, unit_fit.r),
    }
    rows = []
    for rep in range(repetitions):
        data = simulate([seed, 1, rep])
        for scenario, block in (("random_missing", None), ("block_outage", (90, 130))):
            observations = data.observations(block)
            for model, (q, r) in parameters.items():
                times = np.arange(len(data.times)) if model == "fitted_unit_clock" else data.times
                result = kalman_filter(times, observations, q, r, PRIOR_MEAN, PRIOR_COV)
                rows.append({"replication": rep, "scenario": scenario, "model": model,
                             "metrics": score(result, data.states, observations)})
    summary = []
    paired = []
    for scenario in ("random_missing", "block_outage"):
        for model in MODELS:
            selected = [row["metrics"] for row in rows if row["scenario"] == scenario and row["model"] == model]
            summary.append({"scenario": scenario, "model": model,
                            "metrics": {key: mean_and_mcse([row[key] for row in selected]) for key in selected[0]}})
        by_pair = {(row["replication"], row["model"]): row["metrics"] for row in rows if row["scenario"] == scenario}
        paired.append({"scenario": scenario, "contrast": "fitted_clock minus fitted_unit_clock",
                       "metrics": {key: mean_and_mcse([
                           by_pair[rep, "fitted_clock"][key] - by_pair[rep, "fitted_unit_clock"][key]
                           for rep in range(repetitions)])
                           for key in ("level_rmse_missing", "coverage95_missing", "predictive_nll_per_observation")}})
    demo = demonstration(seed, real_fit.q, real_fit.r)
    report = {
        "seed": seed,
        "repetitions": repetitions,
        "data": "Seeded synthetic integrated-Wiener trajectories; no external or private data.",
        "design": {"training_length": training_length, "test_length": 240, "random_missing_rate": 0.2,
                   "block_outage_indices": [90, 130], "true_q": TRUE_Q, "true_r": TRUE_R,
                   "prior_mean": PRIOR_MEAN.tolist(), "prior_cov": PRIOR_COV.tolist(),
                   "mcse_unit": "independent test trajectory, conditional on the single fitted training trajectory",
                   "model_comparisons_paired": True, "smoother_in_online_comparison": False},
        "training_fits": {"fitted_clock": asdict(real_fit), "fitted_unit_clock": asdict(unit_fit)},
        "summary": summary,
        "paired_differences": paired,
        "causality_demo": demo["causality"],
    }
    # Serialization is also a nonfinite-value gate, not a performance-ranking gate.
    json.dumps(report, allow_nan=False)
    json.dumps(rows, allow_nan=False)
    return report, rows, demo


def write_artifacts(report, rows, demo, output: Path):
    from .plotting import make_plots

    output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("summary", report), ("replications", rows), ("demonstration", demo)):
        (output / f"{name}.json").write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_plots(report, demo, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report, rows, demo = run_experiment(args.repetitions, args.seed)
    write_artifacts(report, rows, demo, args.output)
    print(json.dumps({"output": str(args.output), "repetitions": args.repetitions,
                      "training_fits": report["training_fits"], "causality_demo": report["causality_demo"]}, indent=2))


if __name__ == "__main__":
    main()
