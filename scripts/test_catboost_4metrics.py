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
    build_catboost_features,
    compute_four_metrics,
    conformal_residual_interval,
    infer_depth_from_targets,
    make_run_dir,
    save_run_artifacts,
    split_train_calibration,
)
from tabseq.data.datasets import load_dataset_split
from tabseq.labels.trace_encoder import TraceLabelEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--depth", type=int, default=None, help="leaf bin depth; default is auto from y_train")
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
        raise RuntimeError("catboost is required for scripts/test_catboost_4metrics.py") from exc

    split = load_dataset_split(args.dataset, random_state=int(args.seed), val_size=float(args.val_size))
    auto_depth, auto_meta = infer_depth_from_targets(split.y_train)
    depth = int(args.depth) if args.depth is not None else int(auto_depth)

    v_min = float(split.y_train.min())
    v_max = float(split.y_train.max())
    encoder = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)

    X_train, X_val, cat_features = _build_catboost_features(split)
    idx_fit, idx_cal = split_train_calibration(
        n_samples=len(split.y_train),
        seed=int(args.seed),
        calibration_fraction=float(args.calibration_fraction),
    )

    if hasattr(X_train, "iloc"):
        X_fit = X_train.iloc[idx_fit].reset_index(drop=True)
        X_cal = X_train.iloc[idx_cal].reset_index(drop=True)
    else:
        X_fit = X_train[idx_fit]
        X_cal = X_train[idx_cal]
    y_fit = split.y_train[idx_fit]
    y_cal = split.y_train[idx_cal]

    model = CatBoostRegressor(
        iterations=int(args.iterations),
        depth=int(args.model_depth),
        learning_rate=float(args.learning_rate),
        l2_leaf_reg=float(args.l2_leaf_reg),
        loss_function="RMSE",
        random_seed=int(args.seed),
        verbose=False,
    )
    model.fit(X_fit, y_fit, cat_features=cat_features or None)

    pred_cal = model.predict(X_cal)
    pred_val = model.predict(X_val)
    y_lower, y_upper, residual_q = conformal_residual_interval(
        y_fit_pred_cal=pred_cal,
        y_cal=y_cal,
        y_pred_val=pred_val,
        confidence=float(args.confidence),
    )
    metrics = compute_four_metrics(
        y_true=split.y_val,
        y_pred=pred_val,
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
        model="catboost",
        run_id=args.run_id,
    )
    config: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": "catboost",
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
        "interval_method": "split_conformal_residual",
        "residual_quantile": float(residual_q),
    }
    payload = {
        "dataset": str(args.dataset),
        "split": "val",
        "model": "catboost",
        "confidence": float(args.confidence),
        "depth": int(depth),
        "interval_method": "split_conformal_residual",
        **metrics,
    }

    save_run_artifacts(
        run_dir=run_dir,
        config=config,
        metrics=payload,
        y_true=split.y_val,
        y_pred=pred_val,
        y_lower=y_lower,
        y_upper=y_upper,
    )
    model.save_model(os.path.join(run_dir, "model.cbm"))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
