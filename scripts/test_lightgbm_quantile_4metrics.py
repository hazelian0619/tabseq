#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.baselines.four_metrics import (
    compute_four_metrics,
    concat_num_cat,
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


def _build_model(*, alpha: float, args: argparse.Namespace):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=int(args.n_estimators),
        num_leaves=int(args.num_leaves),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="quantile",
        alpha=float(alpha),
        random_state=int(args.seed),
        verbosity=-1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--depth", type=int, default=None, help="leaf bin depth; default is auto from y_train")
    ap.add_argument("--interval-method", choices=["quantile", "cqr"], default="cqr")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--out-root", type=str, default="outputs/baselines_four_metrics")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm is required for scripts/test_lightgbm_quantile_4metrics.py") from exc

    split = load_dataset_split(args.dataset, random_state=int(args.seed), val_size=float(args.val_size))
    auto_depth, auto_meta = infer_depth_from_targets(split.y_train)
    depth = int(args.depth) if args.depth is not None else int(auto_depth)

    v_min = float(split.y_train.min())
    v_max = float(split.y_train.max())
    encoder = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)

    x_train = concat_num_cat(split.X_train, split.X_cat_train)
    x_val = concat_num_cat(split.X_val, split.X_cat_val)
    alpha = 1.0 - float(args.confidence)
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    if str(args.interval_method) == "cqr":
        idx_fit, idx_cal = split_train_calibration(
            n_samples=len(split.y_train),
            seed=int(args.seed),
            calibration_fraction=float(args.calibration_fraction),
        )
        x_fit = x_train[idx_fit]
        x_cal = x_train[idx_cal]
        y_fit = split.y_train[idx_fit]
        y_cal = split.y_train[idx_cal]
    else:
        x_fit = x_train
        x_cal = None
        y_fit = split.y_train
        y_cal = None

    lower_model = _build_model(alpha=q_lower, args=args)
    upper_model = _build_model(alpha=q_upper, args=args)
    lower_model.fit(x_fit, y_fit)
    upper_model.fit(x_fit, y_fit)

    pred_lower_val = lower_model.predict(x_val)
    pred_upper_val = upper_model.predict(x_val)
    pred_lower_val, pred_upper_val = normalize_interval_bounds(pred_lower_val, pred_upper_val)

    correction = 0.0
    if str(args.interval_method) == "cqr":
        assert x_cal is not None and y_cal is not None
        pred_lower_cal = lower_model.predict(x_cal)
        pred_upper_cal = upper_model.predict(x_cal)
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
        model="lightgbm_quantile",
        run_id=args.run_id,
    )
    config: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": "lightgbm_quantile",
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
        "n_estimators": int(args.n_estimators),
        "num_leaves": int(args.num_leaves),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "interval_method": interval_method,
        "quantiles": [float(q_lower), float(q_upper)],
        "quantile_correction": float(correction),
        "point_prediction": "interval_midpoint",
    }
    payload = {
        "dataset": str(args.dataset),
        "split": "val",
        "model": "lightgbm_quantile",
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
    lower_model.booster_.save_model(os.path.join(run_dir, "model_lower.txt"))
    upper_model.booster_.save_model(os.path.join(run_dir, "model_upper.txt"))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
