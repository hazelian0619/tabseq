import numpy as np

from tabseq.baselines.four_metrics import (
    conformalized_quantile_interval,
    interval_midpoint,
    normalize_interval_bounds,
)


def test_normalize_interval_bounds_and_midpoint() -> None:
    lower, upper = normalize_interval_bounds(np.array([2.0, 0.0]), np.array([1.0, 1.0]))
    midpoint = interval_midpoint(lower, upper)

    assert np.allclose(lower, np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(upper, np.array([2.0, 1.0], dtype=np.float32))
    assert np.allclose(midpoint, np.array([1.5, 0.5], dtype=np.float32))


def test_conformalized_quantile_interval_uses_calibration_scores() -> None:
    y_lower, y_upper, correction = conformalized_quantile_interval(
        y_lower_cal=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        y_upper_cal=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        y_cal=np.array([0.2, 2.4, 2.5], dtype=np.float32),
        y_lower_val=np.array([10.0, 20.0], dtype=np.float32),
        y_upper_val=np.array([11.0, 21.0], dtype=np.float32),
        confidence=0.9,
    )

    assert abs(correction - 0.4) < 1e-6
    assert np.allclose(y_lower, np.array([9.6, 19.6], dtype=np.float32))
    assert np.allclose(y_upper, np.array([11.4, 21.4], dtype=np.float32))


def test_conformalized_quantile_interval_uses_finite_sample_conformal_quantile() -> None:
    scores = np.arange(11, dtype=np.float32)
    y_lower, y_upper, correction = conformalized_quantile_interval(
        y_lower_cal=-np.ones_like(scores),
        y_upper_cal=np.zeros_like(scores),
        y_cal=scores,
        y_lower_val=np.array([10.0], dtype=np.float32),
        y_upper_val=np.array([20.0], dtype=np.float32),
        confidence=0.9,
    )

    assert abs(correction - 10.0) < 1e-6
    assert np.allclose(y_lower, np.array([0.0], dtype=np.float32))
    assert np.allclose(y_upper, np.array([30.0], dtype=np.float32))
