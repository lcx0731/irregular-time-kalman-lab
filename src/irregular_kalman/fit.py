"""Train-only noise calibration by Gaussian innovation likelihood."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .core import kalman_filter, validate_prior, validate_series


@dataclass(frozen=True)
class NoiseFit:
    q: float
    r: float
    negative_log_likelihood: float
    best_start_negative_log_likelihood: float
    successful_starts: int
    total_starts: int
    at_bound: bool
    observed_training_points: int


def fit_noise(times, observations, prior_mean, prior_cov) -> NoiseFit:
    """No held-out observations or latent states enter this function.

    Optimize log(q), log(r) with fixed broad bounds and three fixed starts.
    This is a plug-in MLE; downstream bands do not integrate parameter uncertainty.
    At least eight observed values are required as a guard, not an identifiability proof.
    """
    times, observations = validate_series(times, observations)
    validate_prior(prior_mean, prior_cov)
    observed = int(np.isfinite(observations).sum())
    if observed < 8:
        raise ValueError("at least eight observed training values are required")
    bounds = np.log([[1e-5, 10.0], [1e-4, 100.0]])

    def objective(log_parameters):
        q, r = np.exp(log_parameters)
        return -kalman_filter(times, observations, q, r, prior_mean, prior_cov).log_likelihood

    starts = [(0.01, 0.2), (0.1, 1.0), (1.0, 4.0)]
    initial_values = [objective(np.log(start)) for start in starts]
    results = [
        minimize(objective, np.log(start), method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-5})
        for start in starts
    ]
    successful = [result for result in results if result.success and np.isfinite(result.fun)]
    if not successful:
        raise RuntimeError("noise calibration failed from all three starts")
    best = min(successful, key=lambda result: result.fun)
    if best.fun > min(initial_values) + 1e-6:
        raise RuntimeError("converged noise fit did not improve the starting likelihood")
    q, r = np.exp(best.x)
    at_bound = bool(np.any(np.isclose(best.x, bounds[:, 0], atol=1e-4, rtol=0))
                    or np.any(np.isclose(best.x, bounds[:, 1], atol=1e-4, rtol=0)))
    return NoiseFit(float(q), float(r), float(best.fun), float(min(initial_values)),
                    len(successful), len(starts), at_bound, observed)
