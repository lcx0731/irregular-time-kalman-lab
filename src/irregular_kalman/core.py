"""Exact integrated-Wiener transitions, Joseph updates, and offline RTS smoothing."""

from dataclasses import dataclass

import numpy as np


def dynamics(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State=(level,slope), d(level)=slope*dt, d(slope)=sqrt(q)*dW.

    q is a diffusion intensity, not the discrete slope variance. The covariance
    of integrated acceleration scales differently in every matrix entry.
    """
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(q) or q < 0:
        raise ValueError("dt and q must be finite and nonnegative")
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process_cov = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    if not np.all(np.isfinite(process_cov)):
        raise ValueError("process covariance overflow")
    return transition, process_cov


def validate_series(times, observations) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if times.ndim != 1 or times.size == 0 or observations.shape != times.shape:
        raise ValueError("times and observations must be matching nonempty vectors")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    if np.any(np.isinf(observations)):
        raise ValueError("use NaN for missing observations, not infinity")
    return times, observations


def validate_prior(mean, covariance) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("prior must have a 2-vector mean and 2x2 covariance")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("prior must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError("prior covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("prior covariance must be positive definite")
    return mean.copy(), covariance.copy()


@dataclass(frozen=True)
class FilterResult:
    predicted_mean: np.ndarray
    predicted_cov: np.ndarray
    filtered_mean: np.ndarray
    filtered_cov: np.ndarray
    transitions: np.ndarray
    innovations: np.ndarray
    innovation_variance: np.ndarray
    log_likelihood: float


def kalman_filter(times, observations, q: float, r: float, prior_mean, prior_cov) -> FilterResult:
    """Use only observations at or before each time; missing rows predict only.

    The prior is at times[0], immediately before its observation. r is the
    measurement variance. Innovations and likelihood contributions use the
    predicted state BEFORE assimilating the corresponding observation.
    """
    times, observations = validate_series(times, observations)
    mean, covariance = validate_prior(prior_mean, prior_cov)
    if not np.isfinite(q) or q < 0 or not np.isfinite(r) or r <= 0:
        raise ValueError("q must be nonnegative and r strictly positive; both finite")
    n = len(times)
    predicted_mean = np.empty((n, 2))
    filtered_mean = np.empty((n, 2))
    predicted_cov = np.empty((n, 2, 2))
    filtered_cov = np.empty((n, 2, 2))
    transitions = np.empty((n, 2, 2))
    innovations = np.full(n, np.nan)
    variances = np.empty(n)
    log_likelihood = 0.0
    for k in range(n):
        dt = 0.0 if k == 0 else times[k] - times[k - 1]
        transition, process_cov = dynamics(dt, q)
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process_cov
        covariance = (covariance + covariance.T) / 2
        predicted_mean[k], predicted_cov[k], transitions[k] = mean, covariance, transition
        variance = float(covariance[0, 0] + r)
        if not np.isfinite(variance) or variance <= 0:
            raise FloatingPointError("invalid predictive observation variance")
        variances[k] = variance
        if not np.isnan(observations[k]):
            innovation = observations[k] - mean[0]
            gain = covariance[:, 0] / variance
            mean = mean + gain * innovation
            # Joseph form protects symmetry/PSD better than subtracting KHP.
            residual_map = np.eye(2)
            residual_map[:, 0] -= gain
            covariance = residual_map @ covariance @ residual_map.T + r * np.outer(gain, gain)
            covariance = (covariance + covariance.T) / 2
            innovations[k] = innovation
            log_likelihood -= 0.5 * (np.log(2 * np.pi * variance) + innovation**2 / variance)
        filtered_mean[k], filtered_cov[k] = mean, covariance
    if not np.isfinite(log_likelihood) or not np.all(np.isfinite(filtered_mean)):
        raise FloatingPointError("nonfinite filter result")
    return FilterResult(predicted_mean, predicted_cov, filtered_mean, filtered_cov,
                        transitions, innovations, variances, float(log_likelihood))


@dataclass(frozen=True)
class SmoothResult:
    mean: np.ndarray
    covariance: np.ndarray


def rts_smooth(filtered: FilterResult) -> SmoothResult:
    """OFFLINE ONLY: condition past states on all observations, including future ones."""
    mean = filtered.filtered_mean.copy()
    covariance = filtered.filtered_cov.copy()
    for k in range(len(mean) - 2, -1, -1):
        gain = np.linalg.solve(
            filtered.predicted_cov[k + 1],
            filtered.transitions[k + 1] @ filtered.filtered_cov[k],
        ).T
        mean[k] += gain @ (mean[k + 1] - filtered.predicted_mean[k + 1])
        covariance[k] += gain @ (covariance[k + 1] - filtered.predicted_cov[k + 1]) @ gain.T
        covariance[k] = (covariance[k] + covariance[k].T) / 2
    return SmoothResult(mean, covariance)
