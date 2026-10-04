from types import SimpleNamespace

import numpy as np
import pytest

from irregular_kalman.core import kalman_filter
from irregular_kalman.fit import fit_noise
from irregular_kalman.synthetic import PRIOR_COV, PRIOR_MEAN, TRUE_Q, TRUE_R, simulate


def test_simulation_repeats_and_missingness_does_not_change_latent_path():
    first = simulate(123, n=80, missing_rate=0.2)
    second = simulate(123, n=80, missing_rate=0.5)
    repeated = simulate(123, n=80, missing_rate=0.2)
    for field in ("times", "states", "measurements"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        np.testing.assert_array_equal(getattr(first, field), getattr(repeated, field))
    assert np.all(~first.missing | second.missing)
    assert np.all(np.diff(first.times) > 0)
    assert np.std(np.diff(first.times)) > 0.5


def test_missing_block_changes_only_the_observation_mask():
    data = simulate(123, n=80)
    ordinary = data.observations()
    block = data.observations((25, 45))
    assert np.isnan(block[25:45]).all()
    np.testing.assert_array_equal(block[:25], ordinary[:25])
    np.testing.assert_array_equal(block[45:], ordinary[45:])
    assert np.isfinite(data.measurements).all()


def test_noise_fit_is_finite_converged_and_improves_starting_likelihood():
    data = simulate(531, n=80)
    fitted = fit_noise(data.times, data.observations(), PRIOR_MEAN, PRIOR_COV)
    assert 1e-5 <= fitted.q <= 10
    assert 1e-4 <= fitted.r <= 100
    assert fitted.successful_starts >= 1
    assert fitted.total_starts == 3
    assert fitted.negative_log_likelihood <= fitted.best_start_negative_log_likelihood + 1e-6
    result = kalman_filter(data.times, data.observations(), fitted.q, fitted.r, PRIOR_MEAN, PRIOR_COV)
    assert fitted.negative_log_likelihood == pytest.approx(-result.log_likelihood)


def test_failed_optimizer_is_not_silently_accepted(monkeypatch):
    import irregular_kalman.fit as fit_module

    monkeypatch.setattr(fit_module, "minimize", lambda *a, **k: SimpleNamespace(success=False, fun=0))
    data = simulate(531, n=20, missing_rate=0)
    with pytest.raises(RuntimeError, match="all three"):
        fit_noise(data.times, data.observations(), PRIOR_MEAN, PRIOR_COV)


def test_fit_rejects_insufficient_observed_training_data():
    with pytest.raises(ValueError, match="eight"):
        fit_noise(np.arange(10), np.full(10, np.nan), PRIOR_MEAN, PRIOR_COV)


def test_joseph_covariances_stay_symmetric_and_positive_through_outage():
    data = simulate(128, n=100)
    result = kalman_filter(data.times, data.observations((25, 65)), TRUE_Q, TRUE_R, PRIOR_MEAN, PRIOR_COV)
    np.testing.assert_allclose(result.filtered_cov, result.filtered_cov.transpose(0, 2, 1), atol=1e-14)
    assert np.linalg.eigvalsh(result.filtered_cov).min() > 0
    assert result.filtered_cov[64, 0, 0] > result.filtered_cov[25, 0, 0]


@pytest.mark.parametrize("n,missing_rate", [(1, 0.2), (True, 0.2), (3.5, 0.2), (10, -0.1), (10, 1.1), (10, np.nan)])
def test_invalid_simulation_settings(n, missing_rate):
    with pytest.raises(ValueError):
        simulate(1, n, missing_rate)
