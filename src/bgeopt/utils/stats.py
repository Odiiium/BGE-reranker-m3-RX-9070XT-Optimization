"""Statistics shared by run comparison and latency aggregation: bootstrap, randomization and Welch t-intervals."""

from __future__ import annotations

import numpy as np
from scipy import stats


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def bootstrap_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Means of `values` resampled with precomputed `indices` [n_boot, n]; share indices to keep samples paired."""
    return values[indices].mean(axis=-1)


def sign_flip_p_value(diffs: np.ndarray, signs: np.ndarray) -> float:
    """Two-sided paired randomization test of mean(diffs) == 0 using precomputed random signs [n_perm, n]."""
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    permuted = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed * (1 - 1e-12)) + 1) / (len(permuted) + 1))


def median_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    medians = np.median(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return percentile_ci(medians, level)


def ratio_of_medians_ci(numerator: np.ndarray, denominator: np.ndarray, n_boot: int, rng: np.random.Generator,
                        level: float = 0.95) -> tuple[float, float]:
    """CI of median(numerator) / median(denominator) for two independent samples (e.g. latency speedup)."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    num_medians = np.median(num[rng.integers(0, len(num), size=(n_boot, len(num)))], axis=1)
    den_medians = np.median(den[rng.integers(0, len(den), size=(n_boot, len(den)))], axis=1)
    return percentile_ci(num_medians / den_medians, level)


def session_ratio_ci(sessions_num: np.ndarray, sessions_den: np.ndarray,
                     level: float = 0.95) -> tuple[float, float, float] | None:
    """Welch t-interval for the ratio of geometric means of per-session statistics (e.g. session p50 latencies).

    Sessions (separate processes) are the independent units: latency can shift between processes as a whole, so
    resampling individual calls is overconfident. Returns (ratio, low, high), or None with fewer than 2 sessions
    on either side."""
    a = np.log(np.asarray(sessions_num, dtype=np.float64))
    b = np.log(np.asarray(sessions_den, dtype=np.float64))
    if len(a) < 2 or len(b) < 2:
        return None
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return float(np.exp(diff)), float(np.exp(diff)), float(np.exp(diff))
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = float(stats.t.ppf(0.5 + level / 2, df)) * se
    return float(np.exp(diff)), float(np.exp(diff - half)), float(np.exp(diff + half))
