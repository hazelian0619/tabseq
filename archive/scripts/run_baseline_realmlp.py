import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from tabseq.data.datasets import load_california_housing_split
from tabseq.metrics.regression import compute_point_interval_metrics
from tabseq.utils.git import get_git_hash
from tabseq.utils.seed import set_seed

try:
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pytabkit is required for RealMLP baseline. Install with: python3 -m pip install pytabkit"
    ) from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tabseq-config", default=None, help="path to TabSeq config.json (for v_min/v_max)")
    ap.add_argument("--v-min", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=None)
    ap.add_argument(
        "--clip-range",
        action="store_true",
        help="clip y_true/y_lower/y_upper into [v_min, v_max] before metrics",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    alpha = 1.0 - float(args.confidence)
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    split = load_california_housing_split(random_state=args.seed)
    X_train = split.X_train
    y_train = split.y_train
    X_val = split.X_val
    y_val = split.y_val

    v_min = None
    v_max = None
    if args.tabseq_config:
        with open(args.tabseq_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        v_min = float(cfg["v_min"])
        v_max = float(cfg["v_max"])
    if args.v_min is not None:
        v_min = float(args.v_min)
    if args.v_max is not None:
        v_max = float(args.v_max)

    model = RealMLP_TD_Regressor(
        train_metric_name=f"multi_pinball({q_lower},{q_upper})",
        random_state=args.seed,
        device=args.device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    preds = np.asarray(model.predict(X_val))
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError(f"expected RealMLP predictions with shape (N, 2), got {preds.shape}")
    y_lower = np.minimum(preds[:, 0], preds[:, 1])
    y_upper = np.maximum(preds[:, 0], preds[:, 1])

    y_true_eval = np.asarray(y_val)
    y_lower_eval = np.asarray(y_lower)
    y_upper_eval = np.asarray(y_upper)
    bin_edges_02 = None
    bin_edges_04 = None
    if v_min is not None and v_max is not None:
        if args.clip_range:
            y_true_eval = np.clip(y_true_eval, v_min, v_max)
            y_lower_eval = np.clip(y_lower_eval, v_min, v_max)
            y_upper_eval = np.clip(y_upper_eval, v_min, v_max)
        bin_edges_02 = np.arange(v_min, v_max + 0.2, 0.2)
        bin_edges_04 = np.arange(v_min, v_max + 0.4, 0.4)

    metrics = compute_point_interval_metrics(
        y_true=torch.from_numpy(np.asarray(y_true_eval)),
        y_lower=torch.from_numpy(np.asarray(y_lower_eval)),
        y_upper=torch.from_numpy(np.asarray(y_upper_eval)),
        confidence=float(args.confidence),
        return_extras=True,
        bin_edges_02=bin_edges_02,
        bin_edges_04=bin_edges_04,
    )
    metrics["model"] = "realmlp"
    if v_min is not None and v_max is not None:
        y_raw = np.asarray(y_val)
        oob = (y_raw < v_min) | (y_raw > v_max)
        metrics["oob_rate"] = float(np.mean(oob))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "baselines", f"baseline_realmlp_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "model": "realmlp",
        "seed": args.seed,
        "confidence": float(args.confidence),
        "quantiles": [q_lower, q_upper],
        "device": args.device,
        "data_split": "california_housing",
        "v_min": v_min,
        "v_max": v_max,
        "clip_range": bool(args.clip_range),
    }

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")
    with open(os.path.join(out_dir, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
