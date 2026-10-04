import numpy as np
import pytest
from scipy.stats import multivariate_normal

from irregular_kalman.core import dynamics, kalman_filter, rts_smooth

PRIOR_MEAN = np.array([0.3, -0.1])
PRIOR_COV = np.array([[1.0, 0.2], [0.2, 0.4]])


def dense_condition(times, observations, q, r):
    """Independent continuous-time covariance kernel, not a Kalman recursion."""
    n = len(times)
    mean = np.c_[PRIOR_MEAN[0] + times * PRIOR_MEAN[1], np.full(n, PRIOR_MEAN[1])].ravel()
    covariance = np.empty((2 * n, 2 * n))
    a00, a01, a11 = PRIOR_COV[0, 0], PRIOR_COV[0, 1], PRIOR_COV[1, 1]
    for i, a in enumerate(times):
        for j, b in enumerate(times):
            m = min(a, b)
            covariance[2*i:2*i+2, 2*j:2*j+2] = [
                [a00 + (a+b)*a01 + a*b*a11 + q*(a*b*m - (a+b)*m*m/2 + m**3/3),
                 a01 + a*a11 + q*(a*m - m*m/2)],
                [a01 + b*a11 + q*(b*m - m*m/2), a11 + q*m],
            ]
    indices = 2 * np.flatnonzero(np.isfinite(observations))
    observed = observations[np.isfinite(observations)]
    obs_cov = covariance[np.ix_(indices, indices)] + r * np.eye(len(indices))
    cross_cov = covariance[:, indices]
    post_mean = mean + cross_cov @ np.linalg.solve(obs_cov, observed - mean[indices])
    post_cov = covariance - cross_cov @ np.linalg.solve(obs_cov, cross_cov.T)
    likelihood = multivariate_normal.logpdf(observed, mean[indices], obs_cov)
    return post_mean.reshape(n, 2), post_cov, likelihood


def test_integrated_noise_covariance_and_semigroup():
    a, q1 = dynamics(2, 0.3)
    np.testing.assert_allclose(a, [[1, 2], [0, 1]])
    np.testing.assert_allclose(q1, [[0.8, 0.6], [0.6, 0.6]])
    b, q2 = dynamics(0.7, 0.3)
    whole, qwhole = dynamics(2.7, 0.3)
    np.testing.assert_allclose(b @ a, whole, atol=1e-14)
    np.testing.assert_allclose(b @ q1 @ b.T + q2, qwhole, atol=1e-14)


def test_single_observation_has_hand_calculated_update():
    result = kalman_filter([0], [2], 0.2, 1, [0, 0], np.eye(2))
    np.testing.assert_allclose(result.filtered_mean[0], [1, 0])
    np.testing.assert_allclose(result.filtered_cov[0], [[0.5, 0], [0, 1]])
    assert result.innovations[0] == 2
    assert result.innovation_variance[0] == 2
    assert result.log_likelihood == pytest.approx(-0.5 * (np.log(4*np.pi) + 2))


def test_filter_and_likelihood_match_dense_gaussian_conditioning():
    times = np.array([0, 0.4, 2, 4.7])
    observations = np.array([0.2, np.nan, -0.7, 1.1])
    result = kalman_filter(times, observations, 0.08, 0.4, PRIOR_MEAN, PRIOR_COV)
    for k in range(len(times)):
        mean, cov, _ = dense_condition(times[:k+1], observations[:k+1], 0.08, 0.4)
        np.testing.assert_allclose(result.filtered_mean[k], mean[-1], atol=1e-13)
        np.testing.assert_allclose(result.filtered_cov[k], cov[-2:, -2:], atol=1e-13)
    _, _, log_likelihood = dense_condition(times, observations, 0.08, 0.4)
    assert result.log_likelihood == pytest.approx(log_likelihood, abs=1e-12)


def test_rts_matches_dense_full_data_posterior():
    times = np.array([0, 0.4, 2, 4.7])
    observations = np.array([0.2, np.nan, -0.7, 1.1])
    filtered = kalman_filter(times, observations, 0.08, 0.4, PRIOR_MEAN, PRIOR_COV)
    smooth = rts_smooth(filtered)
    mean, cov, _ = dense_condition(times, observations, 0.08, 0.4)
    np.testing.assert_allclose(smooth.mean, mean, atol=1e-13)
    for k in range(len(times)):
        np.testing.assert_allclose(smooth.covariance[k], cov[2*k:2*k+2, 2*k:2*k+2], atol=1e-13)
        assert np.linalg.eigvalsh(filtered.filtered_cov[k] - smooth.covariance[k]).min() >= -1e-12
    np.testing.assert_array_equal(smooth.mean[-1], filtered.filtered_mean[-1])


def test_missing_rows_predict_only_and_contribute_no_likelihood():
    result = kalman_filter([0, 1, 5], [np.nan] * 3, 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    np.testing.assert_array_equal(result.filtered_mean, result.predicted_mean)
    np.testing.assert_array_equal(result.filtered_cov, result.predicted_cov)
    assert np.isnan(result.innovations).all()
    assert result.log_likelihood == 0
    assert result.filtered_cov[-1, 0, 0] > result.filtered_cov[0, 0, 0]


def test_inserting_unobserved_times_does_not_change_posteriors_at_observed_times():
    original = kalman_filter([0, 3, 7], [0, 0.5, 0.4], 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    extra = kalman_filter([0, 1, 2, 3, 4, 6, 7], [0, np.nan, np.nan, 0.5, np.nan, np.nan, 0.4], 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    np.testing.assert_allclose(original.filtered_mean, extra.filtered_mean[[0, 3, 6]], atol=1e-13)
    np.testing.assert_allclose(original.filtered_cov, extra.filtered_cov[[0, 3, 6]], atol=1e-13)


def test_future_edits_leave_filter_prefix_unchanged_but_change_smoother():
    times = np.arange(12, dtype=float)
    y = np.sin(times)
    before = kalman_filter(times, y, 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    y[7:] += 10
    after = kalman_filter(times, y, 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    np.testing.assert_array_equal(before.filtered_mean[:7], after.filtered_mean[:7])
    np.testing.assert_array_equal(before.filtered_cov[:7], after.filtered_cov[:7])
    assert np.max(np.abs(rts_smooth(before).mean[:7] - rts_smooth(after).mean[:7])) > 0.1


@pytest.mark.parametrize("times,observations", [([], []), ([0, 0], [1, 2]), ([0, -1], [1, 2]), ([0, np.nan], [1, 2]), ([0, 1], [1]), ([0], [np.inf])])
def test_invalid_series_is_rejected(times, observations):
    with pytest.raises(ValueError):
        kalman_filter(times, observations, 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)


@pytest.mark.parametrize("q,r", [(-1, 1), (1, 0), (np.nan, 1), (1, np.inf)])
def test_invalid_noise_parameters_are_rejected(q, r):
    with pytest.raises(ValueError):
        kalman_filter([0], [1], q, r, PRIOR_MEAN, PRIOR_COV)


@pytest.mark.parametrize("cov", [np.zeros((2, 2)), [[1, 1], [0, 1]], [[1, 2], [2, 1]], [[1]]])
def test_invalid_prior_is_rejected(cov):
    with pytest.raises(ValueError):
        kalman_filter([0], [1], 0.1, 0.4, PRIOR_MEAN, cov)


def test_changing_time_origin_does_not_change_results():
    a = kalman_filter([0, 2, 3], [0.2, np.nan, 0.7], 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    b = kalman_filter([100, 102, 103], [0.2, np.nan, 0.7], 0.1, 0.4, PRIOR_MEAN, PRIOR_COV)
    np.testing.assert_array_equal(a.filtered_mean, b.filtered_mean)
