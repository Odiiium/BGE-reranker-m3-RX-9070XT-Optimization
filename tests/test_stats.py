import numpy as np
import pytest

from bgeopt.utils.stats import bootstrap_means, median_ci, percentile_ci, ratio_of_medians_ci, sign_flip_p_value


def test_bootstrap_ci_covers_true_mean_and_zero_diffs_are_degenerate():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, size=400)
    boot = bootstrap_means(values, rng.integers(0, len(values), size=(2000, len(values))))
    low, high = percentile_ci(boot, 0.95)
    assert low < 0.3 < high and high - low < 0.25
    zeros = np.zeros(50)
    assert percentile_ci(bootstrap_means(zeros, rng.integers(0, 50, size=(100, 50)))) == (0.0, 0.0)


def test_sign_flip_p_value_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 1.0], size=(4000, 300))
    assert sign_flip_p_value(np.zeros(300), signs) == 1.0
    assert sign_flip_p_value(rng.normal(0.5, 1.0, 300), signs) < 0.001
    assert sign_flip_p_value(rng.normal(0.0, 1.0, 300), signs) > 0.01


def test_median_and_ratio_cis():
    rng = np.random.default_rng(2)
    a = rng.normal(100.0, 5.0, 300)
    b = rng.normal(50.0, 2.5, 300)
    low, high = median_ci(a, 1000, rng)
    assert low < 100 < high
    low, high = ratio_of_medians_ci(a, b, 1000, rng)
    assert low < 2.0 < high
    low, high = ratio_of_medians_ci(a, a, 1000, rng)
    assert low <= 1.0 <= high
    assert high - low == pytest.approx(high - low)  # finite
