"""Deterministic seeded trajectories; no market, user, or downloaded data."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .core import dynamics

TRUE_Q = 0.035
TRUE_R = 0.64
PRIOR_MEAN = np.array([0.0, 0.1])
PRIOR_COV = np.diag([1.0, 0.1])


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    measurements: np.ndarray
    missing: np.ndarray

    def observations(self, block: tuple[int, int] | None = None) -> np.ndarray:
        observed = self.measurements.copy()
        observed[self.missing] = np.nan
        if block is not None:
            start, end = block
            if not 0 <= start < end <= len(observed):
                raise ValueError("invalid missing-block interval")
            observed[start:end] = np.nan
        return observed


def simulate(seed: int | list[int], n: int = 240, missing_rate: float = 0.2) -> Trajectory:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must lie in [0, 1]")
    # Independent child streams stop a change in missingness from changing the path.
    clock_seed, state_seed, noise_seed, missing_seed = np.random.SeedSequence(seed).spawn(4)
    clock_rng = np.random.default_rng(clock_seed)
    state_rng = np.random.default_rng(state_seed)
    noise_rng = np.random.default_rng(noise_seed)
    missing_rng = np.random.default_rng(missing_seed)
    gaps = clock_rng.uniform(0.25, 1.75, n - 1)
    gaps += (clock_rng.random(n - 1) < 0.1) * clock_rng.uniform(3, 7, n - 1)
    times = np.r_[0.0, gaps.cumsum()]
    states = np.empty((n, 2))
    states[0] = state_rng.multivariate_normal(PRIOR_MEAN, PRIOR_COV)
    for k in range(1, n):
        transition, covariance = dynamics(gaps[k - 1], TRUE_Q)
        states[k] = transition @ states[k - 1] + np.linalg.cholesky(covariance) @ state_rng.normal(size=2)
    measurements = states[:, 0] + np.sqrt(TRUE_R) * noise_rng.normal(size=n)
    missing = missing_rng.random(n) < missing_rate
    return Trajectory(times, states, measurements, missing)
