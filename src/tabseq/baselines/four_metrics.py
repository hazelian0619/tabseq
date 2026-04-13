from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.utils.config import write_json


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(*, out_root: str, dataset: str, model: str, run_id: Optional[str]) -> tuple[str, str]:
    rid = str(run_id) if run_id else default_run_id()
    path = os.path.join(str(out_root), str(dataset), str(model), f"run_{rid}")
    os.makedirs(path, exist_ok=True)
    return path, rid


def infer_depth_from_targets(
    y_train: np.ndarray,
    *,
    min_depth: int = 4,
    max_depth: int = 9,
) -> tuple[int, Dict[str, float]]:
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y_train.size == 0:
        raise ValueError("y_train must not be empty")

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = max(y_max - y_min, 1e-12)
    q25, q75 = np.percentile(y_train, [25.0, 75.0])
    y_iqr = max(float(q75 - q25), 1e-12)

    target_bin_width = max(y_iqr / 6.0, y_range / 128.0, 1e-12)
    target_bins = int(np.clip(np.ceil(y_range / target_bin_width), 2**int(min_depth), 2 ** (int(max_depth) - 2)))
    raw_depth = int(np.clip(np.round(np.log2(target_bins)), int(min_depth), int(max_depth) - 1))
    depth = int(np.clip(raw_depth + 1, int(min_depth), int(max_depth)))
    return depth, {
        "y_range": y_range,
        "y_iqr": y_iqr,
        "target_bins": float(target_bins),
        "raw_depth": float(raw_depth),
    }


def split_train_calibration(
    *,
    n_samples: int,
    seed: int,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(calibration_fraction) < 1.0):
        raise ValueError("calibration_fraction must be in (0, 1)")
    indices = np.arange(int(n_samples))
    idx_fit, idx_cal = train_test_split(
        indices,
        test_size=float(calibration_fraction),
        random_state=int(seed),
    )
    return np.asarray(idx_fit), np.asarray(idx_cal)


def concat_num_cat(
    x_num: np.ndarray,
    x_cat: Optional[np.ndarray],
) -> np.ndarray:
    x_num = np.asarray(x_num, dtype=np.float32)
    if x_cat is None or x_cat.shape[1] == 0:
        return x_num
    x_cat = np.asarray(x_cat, dtype=np.float32)
    return np.concatenate([x_num, x_cat], axis=1)


def build_catboost_features(split: Any) -> tuple[Any, Any, list[int]]:
    if split.X_cat_train is None or split.X_cat_train.shape[1] == 0:
        return split.X_train, split.X_val, []

    import pandas as pd

    n_num = int(split.X_train.shape[1])
    n_cat = int(split.X_cat_train.shape[1])
    num_cols = [f"num_{i}" for i in range(n_num)]
    cat_cols = [f"cat_{i}" for i in range(n_cat)]

    x_train = pd.concat(
        [
            pd.DataFrame(split.X_train, columns=num_cols),
            pd.DataFrame(split.X_cat_train.astype("int64"), columns=cat_cols),
        ],
        axis=1,
    )
    x_val = pd.concat(
        [
            pd.DataFrame(split.X_val, columns=num_cols),
            pd.DataFrame(split.X_cat_val.astype("int64"), columns=cat_cols),
        ],
        axis=1,
    )
    cat_features = list(range(n_num, n_num + n_cat))
    return x_train, x_val, cat_features


def normalize_interval_bounds(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_lower = np.asarray(y_lower, dtype=np.float32).reshape(-1)
    y_upper = np.asarray(y_upper, dtype=np.float32).reshape(-1)
    lower = np.minimum(y_lower, y_upper)
    upper = np.maximum(y_lower, y_upper)
    return lower.astype(np.float32), upper.astype(np.float32)


def interval_midpoint(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> np.ndarray:
    lower, upper = normalize_interval_bounds(y_lower, y_upper)
    return (0.5 * (lower + upper)).astype(np.float32)


def conformal_residual_interval(
    *,
    y_fit_pred_cal: np.ndarray,
    y_cal: np.ndarray,
    y_pred_val: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    residuals = np.abs(np.asarray(y_fit_pred_cal, dtype=np.float64) - np.asarray(y_cal, dtype=np.float64))
    q = float(np.quantile(residuals, float(confidence)))
    y_pred_val = np.asarray(y_pred_val, dtype=np.float32).reshape(-1)
    y_lower = y_pred_val - q
    y_upper = y_pred_val + q
    return y_lower.astype(np.float32), y_upper.astype(np.float32), q


def _quantile_higher(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("values must not be empty")
    q = float(np.clip(float(q), 0.0, 1.0))
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(values, q, interpolation="higher"))


def conformalized_quantile_interval(
    *,
    y_lower_cal: np.ndarray,
    y_upper_cal: np.ndarray,
    y_cal: np.ndarray,
    y_lower_val: np.ndarray,
    y_upper_val: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower_cal, upper_cal = normalize_interval_bounds(y_lower_cal, y_upper_cal)
    lower_val, upper_val = normalize_interval_bounds(y_lower_val, y_upper_val)
    y_cal = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    scores = np.maximum(lower_cal.astype(np.float64) - y_cal, y_cal - upper_cal.astype(np.float64))
    correction = _quantile_higher(scores, float(confidence))
    y_lower = lower_val.astype(np.float64) - correction
    y_upper = upper_val.astype(np.float64) + correction
    y_lower, y_upper = normalize_interval_bounds(y_lower, y_upper)
    return y_lower, y_upper, correction


def _values_to_leaf_idx(values: np.ndarray, encoder: TraceLabelEncoder) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clipped = np.clip(values, float(encoder.v_min), float(encoder.v_max))
    denom = max(float(encoder.v_max) - float(encoder.v_min), 1e-12)
    scaled = (clipped - float(encoder.v_min)) / denom
    leaf_idx = np.floor(scaled * int(encoder.n_bins)).astype(np.int64)
    return np.clip(leaf_idx, 0, int(encoder.n_bins) - 1)


def compute_four_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    encoder: TraceLabelEncoder,
    confidence: float,
    tolerance_bins: int = 1,
    clip_interval_to_train_range: bool = True,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    y_lower = np.asarray(y_lower, dtype=np.float32).reshape(-1)
    y_upper = np.asarray(y_upper, dtype=np.float32).reshape(-1)

    if clip_interval_to_train_range:
        lo = float(encoder.v_min)
        hi = float(encoder.v_max)
        y_lower = np.clip(y_lower, lo, hi)
        y_upper = np.clip(y_upper, lo, hi)

    lower = np.minimum(y_lower, y_upper)
    upper = np.maximum(y_lower, y_upper)

    true_leaf_idx = _values_to_leaf_idx(y_true, encoder)
    pred_leaf_idx = _values_to_leaf_idx(y_pred, encoder)

    bin_acc = float(np.mean(pred_leaf_idx == true_leaf_idx))
    tol_bin_acc = float(np.mean(np.abs(pred_leaf_idx - true_leaf_idx) <= int(tolerance_bins)))
    covered = (y_true >= lower) & (y_true <= upper)
    avg_coverage = float(np.mean(covered))
    avg_length = float(np.mean(upper - lower))

    return {
        "bin_acc": bin_acc,
        f"tol_bin_acc@{int(tolerance_bins)}": tol_bin_acc,
        "avg_coverage": avg_coverage,
        "avg_length": avg_length,
    }


def save_run_artifacts(
    *,
    run_dir: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> None:
    write_json(os.path.join(run_dir, "config.json"), config)
    write_json(os.path.join(run_dir, "metrics_val.json"), metrics)
    np.savez(
        os.path.join(run_dir, "predictions_val.npz"),
        y_true=np.asarray(y_true, dtype=np.float32),
        y_pred=np.asarray(y_pred, dtype=np.float32),
        y_lower=np.asarray(y_lower, dtype=np.float32),
        y_upper=np.asarray(y_upper, dtype=np.float32),
    )
