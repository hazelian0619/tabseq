#!/usr/bin/env python3
"""
Prefix-search evaluation for leaf probabilities.

Modes:
  - full: enumerate all leaf prefixes (exact leaf_probs)
  - beam: beam search over prefixes using range-average probabilities (approx)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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


def _build_leaf_dec_inputs(depth: int, n_bins: int, sos_token: int, device: torch.device) -> torch.Tensor:
    dec_inputs = torch.empty((n_bins, depth), dtype=torch.long, device=device)
    for leaf_idx in range(n_bins):
        bits = [int(b) for b in format(leaf_idx, f"0{depth}b")]
        dec = [sos_token] + bits[:-1]
        dec_inputs[leaf_idx] = torch.tensor(dec, dtype=torch.long, device=device)
    return dec_inputs


def _leaf_probs_full(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    dec_inputs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B = x_num.shape[0]
    n_bins = dec_inputs.shape[0]
    leaf_probs = torch.empty((B, n_bins), dtype=torch.float32, device=x_num.device)

    for leaf_idx in range(n_bins):
        dec_input = dec_inputs[leaf_idx].unsqueeze(0).expand(B, -1)
        logits = model({"x_num": x_num, "x_cat": x_cat, "dec_input": dec_input})
        probs = torch.sigmoid(logits / temperature)
        leaf_prob = torch.prod(probs[:, :, leaf_idx], dim=1)
        leaf_probs[:, leaf_idx] = leaf_prob

    leaf_probs = leaf_probs / (torch.sum(leaf_probs, dim=1, keepdim=True) + 1e-9)
    return leaf_probs


def _bits_to_index(bits: List[int]) -> int:
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return idx


def _beam_leaf_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    depth: int,
    n_bins: int,
    temperature: float,
    beam_size: int,
    sos_token: int = 2,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if beam_size <= 0:
        raise ValueError("beam_size must be > 0")

    B = x_num.shape[0]
    device = x_num.device
    eps = 1e-9

    states: List[List[Dict]] = [
        [{"bits": [], "start": 0, "end": n_bins, "logprob": 0.0}] for _ in range(B)
    ]

    for t in range(depth):
        prefix_bits: List[List[int]] = []
        prefix_start: List[int] = []
        prefix_end: List[int] = []
        prefix_logprob: List[float] = []
        prefix_sample: List[int] = []

        for b in range(B):
            for st in states[b]:
                prefix_bits.append(st["bits"])
                prefix_start.append(st["start"])
                prefix_end.append(st["end"])
                prefix_logprob.append(st["logprob"])
                prefix_sample.append(b)

        P = len(prefix_bits)
        dec_input = torch.zeros((P, depth), dtype=torch.long, device=device)
        dec_input[:, 0] = sos_token
        if t > 0:
            bits_tensor = torch.tensor(prefix_bits, dtype=torch.long, device=device)
            dec_input[:, 1 : t + 1] = bits_tensor

        sample_idx = torch.tensor(prefix_sample, dtype=torch.long, device=device)
        x_rep = x_num[sample_idx]
        x_cat_rep = x_cat[sample_idx] if x_cat is not None else None
        logits = model({"x_num": x_rep, "x_cat": x_cat_rep, "dec_input": dec_input})
        probs = torch.sigmoid(logits[:, t, :] / temperature)

        new_states: List[List[Dict]] = [[] for _ in range(B)]
        for i in range(P):
            s = prefix_start[i]
            e = prefix_end[i]
            mid = (s + e) // 2
            left = probs[i, s:mid].mean().item()
            right = probs[i, mid:e].mean().item()
            lp = prefix_logprob[i]
            b = prefix_sample[i]

            new_states[b].append(
                {
                    "bits": prefix_bits[i] + [0],
                    "start": s,
                    "end": mid,
                    "logprob": lp + math.log(max(left, eps)),
                }
            )
            new_states[b].append(
                {
                    "bits": prefix_bits[i] + [1],
                    "start": mid,
                    "end": e,
                    "logprob": lp + math.log(max(right, eps)),
                }
            )

        states = []
        for b in range(B):
            sorted_states = sorted(new_states[b], key=lambda s: s["logprob"], reverse=True)
            states.append(sorted_states[:beam_size])

    leaf_probs = torch.zeros((B, n_bins), dtype=torch.float32, device=device)
    for b in range(B):
        for st in states[b]:
            leaf_idx = _bits_to_index(st["bits"])
            leaf_probs[b, leaf_idx] = math.exp(st["logprob"])

    leaf_probs = leaf_probs / (torch.sum(leaf_probs, dim=1, keepdim=True) + 1e-9)
    return leaf_probs


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fmt_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")

def _safe_path(out_dir: str, filename: str) -> str:
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(filename)
    idx = 2
    while True:
        candidate = os.path.join(out_dir, f"{base}_v{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

def _maybe_subsample(
    leaf_probs: np.ndarray,
    y_raw: np.ndarray,
    y_clipped: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(y_raw):
        return leaf_probs, y_raw, y_clipped
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_raw), size=max_samples, replace=False)
    return leaf_probs[idx], y_raw[idx], y_clipped[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="outputs/<dataset>/run_xxx/ or outputs/<dataset>/run_xxx/checkpoint.pt")
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=None, help="default uses checkpoint seed")
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mode", choices=["full", "beam", "both"], default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="0 means use full validation set")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--export-leaf-probs", action="store_true", help="export leaf_probs/y_true/bin_edges to npz")
    ap.add_argument("--export-samples", type=int, default=0, help="subsample count for export (0 means all)")
    args = ap.parse_args()

    defaults = {
        "batch_size": 128,
        "confidence": 0.90,
        "temperature": 1.0,
        "mode": "both",
        "beam_size": 8,
        "max_samples": 0,
        "device": None,
        "export_leaf_probs": False,
        "export_samples": 0,
    }
    config = load_config(args.config)
    prefix_cfg = resolve_section(config, "prefix_search")
    args.batch_size = choose(args.batch_size, prefix_cfg.get("batch_size"), defaults["batch_size"])
    args.confidence = choose(args.confidence, prefix_cfg.get("confidence"), defaults["confidence"])
    args.temperature = choose(args.temperature, prefix_cfg.get("temperature"), defaults["temperature"])
    args.mode = choose(args.mode, prefix_cfg.get("mode"), defaults["mode"])
    args.beam_size = choose(args.beam_size, prefix_cfg.get("beam_size"), defaults["beam_size"])
    args.max_samples = choose(args.max_samples, prefix_cfg.get("max_samples"), defaults["max_samples"])
    args.device = choose(args.device, prefix_cfg.get("device"), defaults["device"])
    args.export_leaf_probs = choose(
        args.export_leaf_probs, prefix_cfg.get("export_leaf_probs"), defaults["export_leaf_probs"]
    )
    args.export_samples = choose(
        args.export_samples, prefix_cfg.get("export_samples"), defaults["export_samples"]
    )
    args.random_state = choose(args.random_state, prefix_cfg.get("random_state"), None)
    args.dataset = choose(args.dataset, prefix_cfg.get("dataset"), None)
    if args.mode not in {"full", "beam", "both"}:
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
        raise ValueError("checkpoint expects categorical features but eval data has none")
    if n_cat_features > 0:
        if not cat_cardinalities:
            raise ValueError("eval data has categorical features but checkpoint has no cat_cardinalities")
        if len(cat_cardinalities) != n_cat_features:
            raise ValueError(
                f"len(cat_cardinalities)={len(cat_cardinalities)} must match X_cat_val.shape[1]={n_cat_features}"
            )
    if args.max_samples and args.max_samples > 0:
        X_val = X_val[: args.max_samples]
        y_val = y_val[: args.max_samples]
        if X_cat_val is not None:
            X_cat_val = X_cat_val[: args.max_samples]

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
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    metric_calc = ExtendedHolographicMetric(enc)
    dec_inputs = None
    if args.mode in ("full", "both"):
        dec_inputs = _build_leaf_dec_inputs(depth=depth, n_bins=n_bins, sos_token=2, device=device)

    def run_full() -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        leaf_probs_all = []
        y_raw_all = []
        y_clipped_all = []
        with torch.no_grad():
            for batch in dl:
                x_num = batch["x_num"].to(device)
                x_cat = batch.get("x_cat")
                if x_cat is not None:
                    x_cat = x_cat.to(device)
                leaf_probs = _leaf_probs_full(
                    model=model,
                    x_num=x_num,
                    x_cat=x_cat,
                    dec_inputs=dec_inputs,
                    temperature=float(args.temperature),
                )
                leaf_probs_all.append(leaf_probs.cpu())
                y_raw_all.append(batch["y_raw"].cpu())
                y_clipped_all.append(batch["y_clipped"].cpu())
        leaf_probs_np = torch.cat(leaf_probs_all, dim=0).numpy()
        y_raw_np = torch.cat(y_raw_all, dim=0).numpy()
        y_clipped_np = torch.cat(y_clipped_all, dim=0).numpy()
        y_metric = y_clipped_np
        metrics = metric_calc.compute_metrics_from_leaf_probs(
            leaf_probs=torch.from_numpy(leaf_probs_np),
            y_true=torch.from_numpy(y_metric),
            confidence=float(args.confidence),
            return_extras=True,
        )
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = "prefix_full"
        metrics["temperature"] = float(args.temperature)
        return metrics, leaf_probs_np, y_raw_np, y_clipped_np

    def run_beam() -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        leaf_probs_all = []
        y_raw_all = []
        y_clipped_all = []
        with torch.no_grad():
            for batch in dl:
                x_num = batch["x_num"].to(device)
                x_cat = batch.get("x_cat")
                if x_cat is not None:
                    x_cat = x_cat.to(device)
                leaf_probs = _beam_leaf_probs(
                    model=model,
                    x_num=x_num,
                    x_cat=x_cat,
                    depth=depth,
                    n_bins=n_bins,
                    temperature=float(args.temperature),
                    beam_size=int(args.beam_size),
                    sos_token=2,
                )
                leaf_probs_all.append(leaf_probs.cpu())
                y_raw_all.append(batch["y_raw"].cpu())
                y_clipped_all.append(batch["y_clipped"].cpu())
        leaf_probs_np = torch.cat(leaf_probs_all, dim=0).numpy()
        y_raw_np = torch.cat(y_raw_all, dim=0).numpy()
        y_clipped_np = torch.cat(y_clipped_all, dim=0).numpy()
        y_metric = y_clipped_np
        metrics = metric_calc.compute_metrics_from_leaf_probs(
            leaf_probs=torch.from_numpy(leaf_probs_np),
            y_true=torch.from_numpy(y_metric),
            confidence=float(args.confidence),
            return_extras=True,
        )
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = f"prefix_beam_k{int(args.beam_size)}"
        metrics["temperature"] = float(args.temperature)
        return metrics, leaf_probs_np, y_raw_np, y_clipped_np

    def export_leaf_probs(
        out_path: str,
        leaf_probs: np.ndarray,
        y_raw: np.ndarray,
        y_clipped: np.ndarray,
        mode: str,
    ) -> None:
        leaf_probs, y_raw, y_clipped = _maybe_subsample(
            leaf_probs, y_raw, y_clipped, int(args.export_samples), seed=int(random_state)
        )
        bin_edges = np.linspace(v_min, v_max, n_bins + 1, dtype=np.float32)
        bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1])
        meta = {
            "ckpt": ckpt_path,
            "split": "val",
            "mode": mode,
            "beam_size": int(args.beam_size),
            "temperature": float(args.temperature),
            "depth": int(depth),
            "n_bins": int(n_bins),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "random_state": int(random_state),
        }
        np.savez_compressed(
            out_path,
            leaf_probs=leaf_probs.astype(np.float32),
            y_raw=y_raw.astype(np.float32),
            y_clipped=y_clipped.astype(np.float32),
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            meta=json.dumps(meta, ensure_ascii=False),
        )

    out_dir = os.path.dirname(ckpt_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common_tag = f"T{_fmt_float(float(args.temperature))}_C{_fmt_float(float(args.confidence))}"
    write_json(
        _safe_path(out_dir, f"prefix_search_config_{common_tag}_{timestamp}.json"),
        {
            "config_path": args.config,
            "dataset": dataset,
            "batch_size": int(args.batch_size),
            "random_state": int(random_state),
            "confidence": float(args.confidence),
            "temperature": float(args.temperature),
            "mode": str(args.mode),
            "beam_size": int(args.beam_size),
            "max_samples": int(args.max_samples),
            "device": str(device),
            "export_leaf_probs": bool(args.export_leaf_probs),
            "export_samples": int(args.export_samples),
        },
    )

    if args.mode in ("full", "both"):
        m_full, leaf_full, y_raw_full, y_clip_full = run_full()
        full_tag = f"prefix_full_{common_tag}_N{int(args.max_samples) or 'all'}_{timestamp}"
        out_path = _safe_path(out_dir, f"metrics_val_{full_tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(m_full, f, ensure_ascii=False, indent=2)
        print("saved:", out_path)
        if args.export_leaf_probs:
            export_samples = "all" if int(args.export_samples) <= 0 else str(int(args.export_samples))
            export_path = _safe_path(
                out_dir, f"leaf_probs_val_{full_tag}_E{export_samples}.npz"
            )
            export_leaf_probs(export_path, leaf_full, y_raw_full, y_clip_full, mode="prefix_full")
            print("saved:", export_path)

    if args.mode in ("beam", "both"):
        m_beam, leaf_beam, y_raw_beam, y_clip_beam = run_beam()
        beam_tag = (
            f"prefix_beam_K{int(args.beam_size)}_{common_tag}_N{int(args.max_samples) or 'all'}_{timestamp}"
        )
        out_path = _safe_path(out_dir, f"metrics_val_{beam_tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(m_beam, f, ensure_ascii=False, indent=2)
        print("saved:", out_path)
        if args.export_leaf_probs:
            export_samples = "all" if int(args.export_samples) <= 0 else str(int(args.export_samples))
            export_path = _safe_path(
                out_dir, f"leaf_probs_val_{beam_tag}_E{export_samples}.npz"
            )
            export_leaf_probs(export_path, leaf_beam, y_raw_beam, y_clip_beam, mode=m_beam["mode"])
            print("saved:", export_path)

    if args.mode == "both":
        both = {"prefix_full": m_full, "prefix_beam": m_beam}
        both_tag = f"prefix_both_{common_tag}_K{int(args.beam_size)}_{timestamp}"
        out_path = _safe_path(out_dir, f"metrics_val_{both_tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(both, f, ensure_ascii=False, indent=2)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
