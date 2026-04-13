#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.baselines.four_metrics import (
    build_catboost_features,
    compute_four_metrics,
    conformalized_quantile_interval,
    infer_depth_from_targets,
    interval_midpoint,
    make_run_dir,
    normalize_interval_bounds,
    save_run_artifacts,
    split_train_calibration,
)
from tabseq.data.datasets import load_dataset_split
from tabseq.labels.trace_encoder import TraceLabelEncoder


def _select_rows(x: Any, indices: Any) -> Any:
    if hasattr(x, "iloc"):
        return x.iloc[indices].reset_index(drop=True)
    return x[indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--depth", type=int, default=None, help="leaf bin depth; default is auto from y_train")
    ap.add_argument("--interval-method", choices=["quantile", "cqr"], default="cqr")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--model-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2-leaf-reg", type=float, default=3.0)
    ap.add_argument("--out-root", type=str, default="outputs/baselines_four_metrics")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("catboost is required for scripts/test_catboost_quantile_4metrics.py") from exc

    split = load_dataset_split(args.dataset, random_state=int(args.seed), val_size=float(args.val_size))
    auto_depth, auto_meta = infer_depth_from_targets(split.y_train)
    depth = int(args.depth) if args.depth is not None else int(auto_depth)

    v_min = float(split.y_train.min())
    v_max = float(split.y_train.max())
    encoder = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)

    x_train, x_val, cat_features = build_catboost_features(split)
    alpha = 1.0 - float(args.confidence)
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    if str(args.interval_method) == "cqr":
        idx_fit, idx_cal = split_train_calibration(
            n_samples=len(split.y_train),
            seed=int(args.seed),
            calibration_fraction=float(args.calibration_fraction),
        )
        x_fit = _select_rows(x_train, idx_fit)
        x_cal = _select_rows(x_train, idx_cal)
        y_fit = split.y_train[idx_fit]
        y_cal = split.y_train[idx_cal]
    else:
        x_fit = x_train
        x_cal = None
        y_fit = split.y_train
        y_cal = None

    model = CatBoostRegressor(
        iterations=int(args.iterations),
        depth=int(args.model_depth),
        learning_rate=float(args.learning_rate),
        l2_leaf_reg=float(args.l2_leaf_reg),
        loss_function=f"MultiQuantile:alpha={q_lower},{q_upper}",
        random_seed=int(args.seed),
        verbose=False,
    )
    model.fit(x_fit, y_fit, cat_features=cat_features or None)

    pred_val = np.asarray(model.predict(x_val))
    if pred_val.ndim != 2 or pred_val.shape[1] != 2:
        raise ValueError(f"expected MultiQuantile predictions with shape (N, 2), got {pred_val.shape}")
    pred_lower_val, pred_upper_val = normalize_interval_bounds(pred_val[:, 0], pred_val[:, 1])

    correction = 0.0
    if str(args.interval_method) == "cqr":
        assert x_cal is not None and y_cal is not None
        pred_cal = np.asarray(model.predict(x_cal))
        if pred_cal.ndim != 2 or pred_cal.shape[1] != 2:
            raise ValueError(f"expected MultiQuantile predictions with shape (N, 2), got {pred_cal.shape}")
        pred_lower_cal, pred_upper_cal = normalize_interval_bounds(pred_cal[:, 0], pred_cal[:, 1])
        y_lower, y_upper, correction = conformalized_quantile_interval(
            y_lower_cal=pred_lower_cal,
            y_upper_cal=pred_upper_cal,
            y_cal=y_cal,
            y_lower_val=pred_lower_val,
            y_upper_val=pred_upper_val,
            confidence=float(args.confidence),
        )
        interval_method = "conformalized_quantile"
    else:
        y_lower, y_upper = pred_lower_val, pred_upper_val
        interval_method = "quantile"

    y_pred = interval_midpoint(y_lower, y_upper)
    metrics = compute_four_metrics(
        y_true=split.y_val,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        encoder=encoder,
        confidence=float(args.confidence),
        tolerance_bins=1,
        clip_interval_to_train_range=True,
    )

    run_dir, resolved_run_id = make_run_dir(
        out_root=str(args.out_root),
        dataset=str(args.dataset),
        model="catboost_quantile",
        run_id=args.run_id,
    )
    config: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": "catboost_quantile",
        "run_id": resolved_run_id,
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "calibration_fraction": float(args.calibration_fraction),
        "confidence": float(args.confidence),
        "depth": int(depth),
        "auto_depth": int(auto_depth),
        "auto_depth_meta": auto_meta,
        "v_min": v_min,
        "v_max": v_max,
        "iterations": int(args.iterations),
        "model_depth": int(args.model_depth),
        "learning_rate": float(args.learning_rate),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "interval_method": interval_method,
        "quantiles": [float(q_lower), float(q_upper)],
        "quantile_correction": float(correction),
        "point_prediction": "interval_midpoint",
    }
    payload = {
        "dataset": str(args.dataset),
        "split": "val",
        "model": "catboost_quantile",
        "confidence": float(args.confidence),
        "depth": int(depth),
        "interval_method": interval_method,
        **metrics,
    }

    save_run_artifacts(
        run_dir=run_dir,
        config=config,
        metrics=payload,
        y_true=split.y_val,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
    )
    model.save_model(os.path.join(run_dir, "model.cbm"))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
