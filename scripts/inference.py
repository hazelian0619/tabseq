#!/usr/bin/env python3
"""
Fixed-interval inference (greedy by default).

Goal: measure hit-rate for fixed-width intervals on validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabseq.data.datasets import load_dataset_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.utils.config import choose, load_config, resolve_section, write_json
from tabseq.utils.seed import set_seed


def _resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint.pt not found in dir: {path}")
        return ckpt_path
    return path


def _infer_model_class(state_dict: dict):
    if any(key.startswith("decoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel
    if any(key.startswith("tabular_encoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel
    if any(key.startswith("encoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel


def _parse_widths(widths_str: str) -> List[float]:
    parts = [p.strip() for p in widths_str.split(",") if p.strip()]
    widths = [float(p) for p in parts]
    if not widths:
        raise ValueError("widths must be a non-empty list, e.g. 0.2,0.4")
    return widths


def _greedy_step_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    depth: int,
    n_bins: int,
    temperature: float,
    sos_token: int = 2,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B = x_num.shape[0]
    device = x_num.device
    dec_input = torch.zeros((B, depth), dtype=torch.long, device=device)
    dec_input[:, 0] = sos_token
    step_probs_out = torch.empty((B, depth, n_bins), dtype=torch.float32, device=device)
    start = [0 for _ in range(B)]
    end = [n_bins for _ in range(B)]

    for t in range(depth):
        logits = model({"x_num": x_num, "x_cat": x_cat, "dec_input": dec_input})
        probs_t = torch.sigmoid(logits[:, t, :] / temperature)
        step_probs_out[:, t, :] = probs_t
        if t < depth - 1:
            bits = torch.empty((B,), dtype=torch.long, device=device)
            for b in range(B):
                s = start[b]
                e = end[b]
                mid = (s + e) // 2
                left = probs_t[b, s:mid].mean().item()
                right = probs_t[b, mid:e].mean().item()
                bit = 1 if right > left else 0
                bits[b] = bit
                if bit == 0:
                    end[b] = mid
                else:
                    start[b] = mid
            dec_input[:, t + 1] = bits

    return step_probs_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="outputs/<dataset>/run_xxx/ or outputs/<dataset>/run_xxx/checkpoint.pt")
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dataset", type=str, default=None, help="override dataset name")
    ap.add_argument("--random-state", type=int, default=None, help="default uses checkpoint seed")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mode", choices=["greedy", "teacher_forcing"], default=None)
    ap.add_argument(
        "--widths",
        type=str,
        default=None,
        help="Fixed interval widths in y-space, e.g. 0.2,0.4",
    )
    args = ap.parse_args()

    defaults = {
        "batch_size": 256,
        "temperature": 1.0,
        "mode": "greedy",
        "widths": "0.2,0.4",
    }
    config = load_config(args.config)
    infer_cfg = resolve_section(config, "inference")
    args.batch_size = choose(args.batch_size, infer_cfg.get("batch_size"), defaults["batch_size"])
    args.temperature = choose(args.temperature, infer_cfg.get("temperature"), defaults["temperature"])
    args.mode = choose(args.mode, infer_cfg.get("mode"), defaults["mode"])
    args.widths = choose(args.widths, infer_cfg.get("widths"), defaults["widths"])
    args.random_state = choose(args.random_state, infer_cfg.get("random_state"), None)
    args.dataset = choose(args.dataset, infer_cfg.get("dataset"), None)
    if args.mode not in {"greedy", "teacher_forcing"}:
        raise ValueError(f"invalid mode: {args.mode}")

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    depth = int(cfg["depth"])
    n_bins = int(cfg["n_bins"])
    n_num_features = int(cfg["n_num_features"])
    v_min = float(cfg["v_min"])
    v_max = float(cfg["v_max"])

    random_state = int(args.random_state) if args.random_state is not None else int(cfg.get("seed", 0))
    set_seed(random_state)

    dataset = args.dataset or cfg.get("dataset", "california_housing")
    if args.dataset is not None and cfg.get("dataset") and args.dataset != cfg.get("dataset"):
        raise ValueError(f"dataset mismatch: args.dataset={args.dataset} vs ckpt={cfg.get('dataset')}")

    split = load_dataset_split(dataset, random_state=random_state)
    X_val, y_val = split.X_val, split.y_val
    X_cat_val = split.X_cat_val
    cat_cardinalities = cfg.get("cat_cardinalities") or []
    n_cat_features = int(X_cat_val.shape[1]) if X_cat_val is not None else 0
    if n_cat_features == 0 and cat_cardinalities:
        raise ValueError("checkpoint expects categorical features but inference data has none")
    if n_cat_features > 0:
        if not cat_cardinalities:
            raise ValueError("inference data has categorical features but checkpoint has no cat_cardinalities")
        if len(cat_cardinalities) != n_cat_features:
            raise ValueError(
                f"len(cat_cardinalities)={len(cat_cardinalities)} must match X_cat_val.shape[1]={n_cat_features}"
            )

    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)
    ds = TabSeqDataset(
        X_num=X_val,
        X_cat=X_cat_val if X_cat_val is not None else np.zeros((len(y_val), 0), dtype=np.int64),
        y=y_val,
        encoder=enc,
        is_train=False,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model_cls = _infer_model_class(ckpt["model_state_dict"])
    if model_cls is TransformerTabSeqModel:
        model = model_cls(
            n_num_features=n_num_features,
            depth=depth,
            n_bins=n_bins,
            cat_cardinalities=cat_cardinalities,
        )
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metric_calc = ExtendedHolographicMetric(enc)
    step_probs_all = []
    y_true_all = []

    with torch.no_grad():
        for batch in dl:
            if args.mode == "teacher_forcing":
                logits = model(batch)
                step_probs = torch.sigmoid(logits / float(args.temperature))
            else:
                step_probs = _greedy_step_probs(
                    model=model,
                    x_num=batch["x_num"],
                    x_cat=batch.get("x_cat"),
                    depth=depth,
                    n_bins=n_bins,
                    temperature=float(args.temperature),
                    sos_token=2,
                )
            step_probs_all.append(step_probs.cpu())
            y_target = batch["y_clipped"] if "y_clipped" in batch else batch["y_raw"]
            y_true_all.append(y_target.cpu())

    step_probs = torch.cat(step_probs_all, dim=0)
    y_true = torch.cat(y_true_all, dim=0)

    leaf_probs = metric_calc.leaf_probs_from_step_probs(step_probs)
    bin_values = torch.tensor(
        [enc.decode_bin_index(i) for i in range(leaf_probs.shape[1])],
        dtype=torch.float32,
    )
    y_pred = torch.sum(leaf_probs * bin_values, dim=1)

    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    widths = _parse_widths(args.widths)
    results = {}
    for w in widths:
        half = float(w) / 2.0
        lower = y_pred - half
        upper = y_pred + half
        covered = (y_true >= lower) & (y_true <= upper)
        hit_rate = float(torch.mean(covered.float()).item())
        results[str(w)] = {"hit_rate": hit_rate, "n": int(covered.numel()), "width": float(w)}

    out = {
        "mode": args.mode,
        "temperature": float(args.temperature),
        "widths": widths,
        "point_MAE": float(mae),
        "point_RMSE": float(rmse),
        "fixed_interval_hit_rate": results,
    }

    out_dir = os.path.dirname(ckpt_path)
    write_json(
        os.path.join(out_dir, "fixed_interval_config.json"),
        {
            "config_path": args.config,
            "batch_size": int(args.batch_size),
            "random_state": int(random_state),
            "temperature": float(args.temperature),
            "mode": str(args.mode),
            "widths": widths,
        },
    )
    out_json = os.path.join(out_dir, "fixed_interval_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(out_dir, "fixed_interval_eval.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["width", "hit_rate", "n", "point_MAE", "point_RMSE"])
        writer.writeheader()
        for w in widths:
            row = results[str(w)]
            writer.writerow(
                {
                    "width": row["width"],
                    "hit_rate": row["hit_rate"],
                    "n": row["n"],
                    "point_MAE": float(mae),
                    "point_RMSE": float(rmse),
                }
            )

    print("saved:", out_json)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
