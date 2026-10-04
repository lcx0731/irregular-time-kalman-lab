import json

import numpy as np
import pytest

from irregular_kalman.experiment import mean_and_mcse, run_experiment, write_artifacts


@pytest.fixture(scope="module")
def small_run():
    return run_experiment(repetitions=2, seed=731, training_length=60)


def test_mcse_uses_independent_trajectory_values():
    result = mean_and_mcse([1, 3, 5])
    assert result["mean"] == 3
    assert result["mcse"] == pytest.approx(2 / np.sqrt(3))


def test_small_workflow_is_complete_and_paired(small_run):
    report, rows, demo = small_run
    assert len(rows) == 2 * 2 * 3
    assert len(report["summary"]) == 6
    assert len(report["paired_differences"]) == 2
    assert report["design"]["smoother_in_online_comparison"] is False
    for contrast in report["paired_differences"]:
        scenario = contrast["scenario"]
        lookup = {(r["replication"], r["model"]): r["metrics"] for r in rows if r["scenario"] == scenario}
        values = [lookup[i, "fitted_clock"]["level_rmse_missing"] - lookup[i, "fitted_unit_clock"]["level_rmse_missing"] for i in range(2)]
        assert contrast["metrics"]["level_rmse_missing"] == mean_and_mcse(values)
    assert demo["causality"]["filter_prefix_max_change"] == 0
    assert demo["causality"]["smoother_prefix_max_change"] > 0


def test_saved_artifacts_are_finite_and_include_real_figures(small_run, tmp_path):
    report, rows, demo = small_run
    write_artifacts(report, rows, demo, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    assert json.loads((tmp_path / "replications.json").read_text()) == rows
    for name in ("performance.png", "outage_and_smoothing.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / name).stat().st_size > 10_000


@pytest.mark.parametrize("repetitions", [0, 1, 2.5, True])
def test_invalid_replication_count(repetitions):
    with pytest.raises(ValueError):
        run_experiment(repetitions=repetitions)
