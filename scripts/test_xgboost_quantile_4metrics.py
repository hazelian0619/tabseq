#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.baselines.four_metrics import (
    compute_four_metrics,
    concat_num_cat,
    conformalized_quantile_interval,
    format_duration,
    infer_depth_from_targets,
    interval_midpoint,
    log_progress,
    make_run_dir,
    normalize_interval_bounds,
    save_run_artifacts,
    split_train_calibration,
)
from tabseq.data.datasets import load_dataset_split
from tabseq.labels.trace_encoder import TraceLabelEncoder


def _build_model(*, alpha: float, args: argparse.Namespace):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        objective="reg:quantileerror",
        quantile_alpha=float(alpha),
        random_state=int(args.seed),
        tree_method="hist",
        verbosity=0,
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
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--log-every", type=int, default=50, help="training progress print period")
    ap.add_argument("--out-root", type=str, default="outputs/baselines_four_metrics")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    try:
        import xgboost  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost is required for scripts/test_xgboost_quantile_4metrics.py") from exc

    load_start = perf_counter()
    log_progress(
        f"loading dataset={args.dataset} seed={int(args.seed)} val_size={float(args.val_size):.2f}"
    )
    split = load_dataset_split(args.dataset, random_state=int(args.seed), val_size=float(args.val_size))
    n_cat = 0 if split.X_cat_train is None else int(split.X_cat_train.shape[1])
    log_progress(
        "dataset loaded "
        f"train_rows={len(split.y_train)} val_rows={len(split.y_val)} "
        f"num_features={int(split.X_train.shape[1])} cat_features={n_cat} "
        f"elapsed={format_duration(perf_counter() - load_start)}"
    )
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
    log_progress(
        "prepared split "
        f"interval_method={str(args.interval_method)} fit_rows={len(y_fit)} "
        f"cal_rows={0 if y_cal is None else len(y_cal)} val_rows={len(split.y_val)} depth={depth}"
    )

    lower_model = _build_model(alpha=q_lower, args=args)
    upper_model = _build_model(alpha=q_upper, args=args)
    log_every = max(1, int(args.log_every))
    fit_start = perf_counter()
    log_progress(
        "start fit lower quantile model "
        f"alpha={q_lower:.3f} n_estimators={int(args.n_estimators)} max_depth={int(args.max_depth)} "
        f"learning_rate={float(args.learning_rate)} log_every={log_every}"
    )
    lower_model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit)], verbose=log_every)
    log_progress(f"lower quantile model finished elapsed={format_duration(perf_counter() - fit_start)}")
    fit_start = perf_counter()
    log_progress(
        "start fit upper quantile model "
        f"alpha={q_upper:.3f} n_estimators={int(args.n_estimators)} max_depth={int(args.max_depth)} "
        f"learning_rate={float(args.learning_rate)} log_every={log_every}"
    )
    upper_model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit)], verbose=log_every)
    log_progress(f"upper quantile model finished elapsed={format_duration(perf_counter() - fit_start)}")

    predict_start = perf_counter()
    log_progress("start predict validation quantiles")
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
    log_progress(f"prediction and interval construction finished elapsed={format_duration(perf_counter() - predict_start)}")

    y_pred = interval_midpoint(y_lower, y_upper)
    metrics = compute_four_metrics(
        y_true=split.y_val,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        encoder=encoder,
        confidence=float(args.confidence),
        tolerance_bins=1,
        clip_interval_to_train_range=(interval_method != "conformalized_quantile"),
    )

    run_dir, resolved_run_id = make_run_dir(
        out_root=str(args.out_root),
        dataset=str(args.dataset),
        model="xgboost_quantile",
        run_id=args.run_id,
    )
    config: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": "xgboost_quantile",
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
        "max_depth": int(args.max_depth),
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
        "model": "xgboost_quantile",
        "confidence": float(args.confidence),
        "depth": int(depth),
        "interval_method": interval_method,
        **metrics,
    }

    save_start = perf_counter()
    log_progress("saving artifacts")
    save_run_artifacts(
        run_dir=run_dir,
        config=config,
        metrics=payload,
        y_true=split.y_val,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
    )
    lower_model.save_model(os.path.join(run_dir, "model_lower.json"))
    upper_model.save_model(os.path.join(run_dir, "model_upper.json"))
    log_progress(f"artifacts saved elapsed={format_duration(perf_counter() - save_start)} run_dir={run_dir}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
