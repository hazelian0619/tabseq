#!/usr/bin/env python3
"""
区间宽度扫描实验：
遍历不同 n_bins（由各自 checkpoint 决定），计算覆盖率、平均宽度、Score。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    raise ValueError("unknown model type in checkpoint")


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
        mask = torch.zeros_like(probs_t)
        for b in range(B):
            mask[b, start[b]:end[b]] = 1.0
        probs_t = probs_t * mask
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


def _build_val_loader(
    *,
    dataset: str,
    random_state: int,
    batch_size: int,
    v_min: float,
    v_max: float,
    depth: int,
    expected_cat_features: int,
) -> tuple[DataLoader, TraceLabelEncoder]:
    split = load_dataset_split(dataset, random_state=random_state)
    X_val, y_val = split.X_val, split.y_val
    X_cat_val = split.X_cat_val
    n_cat_features = int(X_cat_val.shape[1]) if X_cat_val is not None else 0
    if n_cat_features != expected_cat_features:
        raise ValueError(
            f"X_cat_val.shape[1]={n_cat_features} must match expected_cat_features={expected_cat_features}"
        )
    encoder = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)
    ds = TabSeqDataset(
        X_num=X_val,
        X_cat=X_cat_val if X_cat_val is not None else np.zeros((len(y_val), 0), dtype=np.int64),
        y=y_val,
        encoder=encoder,
        is_train=False,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl, encoder


def _evaluate_ckpt(
    ckpt_path: str,
    *,
    device: torch.device,
    mode: str,
    confidence: float,
    temperature: float,
    batch_size: int,
    random_state: int | None,
    dataset: Optional[str],
    lambda_width: float,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    depth = int(cfg["depth"])
    n_bins = int(cfg["n_bins"])
    n_num_features = int(cfg["n_num_features"])
    v_min = float(cfg["v_min"])
    v_max = float(cfg["v_max"])
    cat_cardinalities = cfg.get("cat_cardinalities") or []
    n_cat_features = len(cat_cardinalities)

    seed = int(random_state) if random_state is not None else int(cfg.get("seed", 0))
    set_seed(seed)

    ckpt_dataset = cfg.get("dataset", "california_housing")
    if dataset is not None and dataset != ckpt_dataset:
        raise ValueError(f"dataset mismatch: args.dataset={dataset} vs ckpt={ckpt_dataset}")
    use_dataset = dataset or ckpt_dataset

    dl, encoder = _build_val_loader(
        dataset=use_dataset,
        random_state=seed,
        batch_size=batch_size,
        v_min=v_min,
        v_max=v_max,
        depth=depth,
        expected_cat_features=n_cat_features,
    )

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
    model.to(device)
    model.eval()

    metric_calc = ExtendedHolographicMetric(encoder)
    step_probs_all: List[torch.Tensor] = []
    y_true_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            if mode == "teacher_forcing":
                logits = model(batch)
                step_probs = torch.sigmoid(logits / float(temperature))
            elif mode == "greedy":
                step_probs = _greedy_step_probs(
                    model=model,
                    x_num=batch["x_num"],
                    x_cat=batch.get("x_cat"),
                    depth=depth,
                    n_bins=n_bins,
                    temperature=float(temperature),
                    sos_token=2,
                )
            else:
                raise ValueError(f"unknown mode: {mode}")

            step_probs_all.append(step_probs.cpu())
            y_target = batch["y_clipped"] if "y_clipped" in batch else batch["y_raw"]
            y_true_all.append(y_target.cpu())

    metrics = metric_calc.compute_metrics(
        model_probs=torch.cat(step_probs_all, dim=0),
        y_true=torch.cat(y_true_all, dim=0),
        confidence=float(confidence),
        return_extras=False,
    )
    score = float(metrics["PICP"]) - float(lambda_width) * float(metrics["MPIW"])

    return {
        "ckpt": ckpt_path,
        "depth": depth,
        "n_bins": n_bins,
        "bin_width": float(encoder.bin_width),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "PICP": metrics["PICP"],
        "MPIW": metrics["MPIW"],
        "coverage": metrics["PICP"],
        "avg_width": metrics["MPIW"],
        "score": score,
        "confidence": float(confidence),
        "mode": mode,
        "temperature": float(temperature),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True, help="checkpoint 路径或 run 目录，可传多个")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["teacher_forcing", "greedy"],
    )
    parser.add_argument("--lambda-width", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    defaults = {
        "device": "cuda",
        "batch_size": 256,
        "confidence": 0.90,
        "temperature": 1.0,
        "mode": "teacher_forcing",
        "lambda_width": 0.5,
        "out_dir": ".",
    }
    config = load_config(args.config)
    scan_cfg = resolve_section(config, "width_scan")
    args.device = choose(args.device, scan_cfg.get("device"), defaults["device"])
    args.batch_size = choose(args.batch_size, scan_cfg.get("batch_size"), defaults["batch_size"])
    args.confidence = choose(args.confidence, scan_cfg.get("confidence"), defaults["confidence"])
    args.temperature = choose(args.temperature, scan_cfg.get("temperature"), defaults["temperature"])
    args.mode = choose(args.mode, scan_cfg.get("mode"), defaults["mode"])
    args.lambda_width = choose(args.lambda_width, scan_cfg.get("lambda_width"), defaults["lambda_width"])
    args.out_dir = choose(args.out_dir, scan_cfg.get("out_dir"), defaults["out_dir"])
    args.random_state = choose(args.random_state, scan_cfg.get("random_state"), None)
    args.dataset = choose(args.dataset, scan_cfg.get("dataset"), None)
    if args.mode not in {"teacher_forcing", "greedy"}:
        raise ValueError(f"invalid mode: {args.mode}")

    device = torch.device(args.device)
    ckpt_paths = [_resolve_ckpt_path(path) for path in args.ckpt]

    results = []
    for ckpt_path in ckpt_paths:
        metrics = _evaluate_ckpt(
            ckpt_path,
            device=device,
            mode=args.mode,
            confidence=args.confidence,
            temperature=args.temperature,
            batch_size=args.batch_size,
            random_state=args.random_state,
            dataset=args.dataset,
            lambda_width=args.lambda_width,
        )
        results.append(metrics)
        print(
            f"n_bins={metrics['n_bins']}: PICP={metrics['PICP']:.3f}, "
            f"MPIW={metrics['MPIW']:.3f}, Score={metrics['score']:.3f}"
        )

    df = pd.DataFrame(results).sort_values(by="n_bins").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    write_json(
        os.path.join(args.out_dir, "width_scan_config.json"),
        {
            "config_path": args.config,
            "batch_size": int(args.batch_size),
            "random_state": args.random_state,
            "dataset": args.dataset,
            "confidence": float(args.confidence),
            "temperature": float(args.temperature),
            "mode": str(args.mode),
            "lambda_width": float(args.lambda_width),
            "device": str(args.device),
            "ckpts": ckpt_paths,
        },
    )
    csv_path = os.path.join(args.out_dir, f"width_scan_results_{args.mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df["n_bins"], df["PICP"], "o-", label="PICP")
    plt.plot(df["n_bins"], df["MPIW"], "s-", label="MPIW")
    plt.plot(df["n_bins"], df["score"], "^-", label="Score")
    plt.xlabel("n_bins (桶数)")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Interval Width vs Performance")
    fig_path = os.path.join(args.out_dir, f"width_performance_curve_{args.mode}.png")
    plt.savefig(fig_path)
    plt.show()


if __name__ == "__main__":
    main()
