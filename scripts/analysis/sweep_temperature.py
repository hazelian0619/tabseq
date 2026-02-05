import argparse
import csv
import json
import os
from datetime import datetime
from math import ceil
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabseq.data.datasets import load_dataset_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric
try:
    from tabseq.models.minimal_model import MinimalTabSeqModel
except ImportError:  # 兼容旧环境：若无 minimal_model，就只用主模型
    MinimalTabSeqModel = None
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
    return MinimalTabSeqModel or TransformerTabSeqModel


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


def _temperature_grid(t_min: float, t_max: float, t_step: float) -> list[float]:
    if t_step <= 0:
        raise ValueError("t_step must be > 0")
    n_steps = int(ceil((t_max - t_min) / t_step))
    temps = []
    for i in range(n_steps + 1):
        t = t_min + i * t_step
        if t > t_max + 1e-9:
            break
        temps.append(round(t, 3))
    return temps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir or checkpoint.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--t-step", type=float, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--random-state", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    args = ap.parse_args()

    defaults = {
        "t_min": 0.5,
        "t_max": 10.0,
        "t_step": 0.1,
        "confidence": 0.90,
        "random_state": 0,
        "batch_size": 256,
    }
    config = load_config(args.config)
    sweep_cfg = resolve_section(config, "temperature_sweep")
    args.t_min = choose(args.t_min, sweep_cfg.get("t_min"), defaults["t_min"])
    args.t_max = choose(args.t_max, sweep_cfg.get("t_max"), defaults["t_max"])
    args.t_step = choose(args.t_step, sweep_cfg.get("t_step"), defaults["t_step"])
    args.confidence = choose(args.confidence, sweep_cfg.get("confidence"), defaults["confidence"])
    args.random_state = choose(args.random_state, sweep_cfg.get("random_state"), defaults["random_state"])
    args.batch_size = choose(args.batch_size, sweep_cfg.get("batch_size"), defaults["batch_size"])
    args.out_dir = choose(args.out_dir, sweep_cfg.get("out_dir"), None)
    args.dataset = choose(args.dataset, sweep_cfg.get("dataset"), None)

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    depth = int(cfg["depth"])
    n_bins = int(cfg["n_bins"])
    n_num_features = int(cfg["n_num_features"])
    v_min = float(cfg["v_min"])
    v_max = float(cfg["v_max"])

    set_seed(args.random_state)
    dataset = args.dataset or cfg.get("dataset", "california_housing")
    if args.dataset is not None and cfg.get("dataset") and args.dataset != cfg.get("dataset"):
        raise ValueError(f"dataset mismatch: args.dataset={args.dataset} vs ckpt={cfg.get('dataset')}")
    split = load_dataset_split(dataset, random_state=args.random_state)
    X_val, y_val = split.X_val, split.y_val
    X_cat_val = split.X_cat_val
    cat_cardinalities = cfg.get("cat_cardinalities") or []
    n_cat_features = int(X_cat_val.shape[1]) if X_cat_val is not None else 0
    if n_cat_features == 0 and cat_cardinalities:
        raise ValueError("checkpoint expects categorical features but sweep data has none")
    if n_cat_features > 0:
        if not cat_cardinalities:
            raise ValueError("sweep data has categorical features but checkpoint has no cat_cardinalities")
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("outputs", "sweeps", "temperature", f"temperature_sweep_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    write_json(
        os.path.join(out_dir, "temperature_sweep_config.json"),
        {
            "config_path": args.config,
            "dataset": dataset,
            "t_min": float(args.t_min),
            "t_max": float(args.t_max),
            "t_step": float(args.t_step),
            "confidence": float(args.confidence),
            "random_state": int(args.random_state),
            "batch_size": int(args.batch_size),
            "out_dir": out_dir,
        },
    )

    temps = _temperature_grid(args.t_min, args.t_max, args.t_step)
    rows = []

    for t in temps:
        step_probs_all = []
        y_true_all = []
        with torch.no_grad():
            for batch in dl:
                step_probs = _greedy_step_probs(
                    model=model,
                    x_num=batch["x_num"],
                    x_cat=batch.get("x_cat"),
                    depth=depth,
                    n_bins=n_bins,
                    temperature=t,
                    sos_token=2,
                )
                step_probs_all.append(step_probs.cpu())
                y_true_all.append(batch["y_raw"].cpu())

        metrics = metric_calc.compute_metrics(
            model_probs=torch.cat(step_probs_all, dim=0),
            y_true=torch.cat(y_true_all, dim=0),
            confidence=float(args.confidence),
            return_extras=True,
        )
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = "greedy"
        metrics["temperature"] = float(t)

        out_path = os.path.join(out_dir, f"metrics_val_greedy_T{t:.1f}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        rows.append(
            {
                "temperature": t,
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "PICP": metrics.get("PICP"),
                "MPIW": metrics.get("MPIW"),
                "bin_acc_0.2": metrics.get("bin_acc_0.2"),
                "bin_acc_0.4": metrics.get("bin_acc_0.4"),
            }
        )

    rows.sort(key=lambda r: r["temperature"])
    summary_path = os.path.join(out_dir, "temperature_sweep_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = [r["temperature"] for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(T, [r["PICP"] for r in rows], color="#1f77b4")
        axes[0].axhline(float(args.confidence), color="#d62728", linestyle="--", linewidth=1)
        axes[0].set_ylabel("PICP")
        axes[0].set_title("Temperature Sweep (greedy)")

        axes[1].plot(T, [r["MPIW"] for r in rows], color="#2ca02c")
        axes[1].set_ylabel("MPIW")

        axes[2].plot(T, [r["MAE"] for r in rows], color="#ff7f0e", label="MAE")
        axes[2].plot(T, [r["RMSE"] for r in rows], color="#9467bd", label="RMSE")
        axes[2].set_ylabel("Error")
        axes[2].set_xlabel("Temperature")
        axes[2].legend(loc="best", fontsize=8)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, "temperature_sweep_plot.png")
        fig.savefig(plot_path, dpi=150)
    except Exception:
        pass

    print("saved:", summary_path)


if __name__ == "__main__":
    main()
