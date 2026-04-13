#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.data.datasets import load_dataset_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.inference import calibrate_confidence_from_leaf_probs, collect_beam_outputs, collect_greedy_outputs
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
    raise ValueError("unable to infer model class from checkpoint state_dict keys")


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        requested = str(requested)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)
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


def _parse_optional_float_grid(raw):
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    else:
        values = [float(raw)]
    return tuple(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run 目录或 checkpoint.pt 路径")
    ap.add_argument("--config", type=str, default=None, help="测试配置文件（可选）")
    ap.add_argument("--dataset", type=str, default=None, help="覆盖 checkpoint 里的 dataset")
    ap.add_argument("--random-state", type=int, default=None, help="覆盖 checkpoint 里的数据划分种子")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--confidence-grid", type=str, default=None, help="comma-separated inference confidences for auto calibration on val")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--interval-method", choices=["cdf", "shortest_mass", "relaxed_mass", "peak_merge"], default=None)
    ap.add_argument("--peak-merge-alpha", type=float, default=None)
    ap.add_argument("--mask-outside", type=float, default=None)
    ap.add_argument("--mode", choices=["greedy", "beam"], default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--leaf-prior-weight", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    defaults = {
        "batch_size": 256,
        "confidence": 0.90,
        "temperature": 1.0,
        "interval_method": "relaxed_mass",
        "peak_merge_alpha": 0.33,
        "mask_outside": 0.0,
        "mode": "beam",
        "beam_size": 8,
        "leaf_prior_weight": 0.0,
        "device": None,
    }

    config = load_config(args.config)
    test_cfg = resolve_section(config, "test")
    eval_cfg = resolve_section(config, "eval")
    requested_confidence = args.confidence

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    target_confidence = float(
        choose(
            test_cfg.get("target_confidence"),
            eval_cfg.get("target_confidence"),
            cfg.get("target_confidence"),
            cfg.get("confidence"),
            defaults["confidence"],
        )
    )
    args.batch_size = choose(
        args.batch_size, test_cfg.get("batch_size"), eval_cfg.get("batch_size"), cfg.get("batch_size"), defaults["batch_size"]
    )
    args.confidence = choose(
        args.confidence,
        test_cfg.get("confidence"),
        cfg.get("calibrated_confidence"),
        eval_cfg.get("confidence"),
        cfg.get("confidence"),
        defaults["confidence"],
    )
    confidence_grid = _parse_optional_float_grid(
        choose(args.confidence_grid, test_cfg.get("confidence_grid"), eval_cfg.get("confidence_grid"), cfg.get("confidence_grid"))
    )
    args.temperature = choose(
        args.temperature, test_cfg.get("temperature"), eval_cfg.get("temperature"), cfg.get("temperature"), defaults["temperature"]
    )
    args.interval_method = choose(
        args.interval_method,
        test_cfg.get("interval_method"),
        eval_cfg.get("interval_method"),
        cfg.get("interval_method"),
        defaults["interval_method"],
    )
    args.peak_merge_alpha = choose(
        args.peak_merge_alpha,
        test_cfg.get("peak_merge_alpha"),
        eval_cfg.get("peak_merge_alpha"),
        cfg.get("peak_merge_alpha"),
        defaults["peak_merge_alpha"],
    )
    args.mask_outside = choose(
        args.mask_outside,
        test_cfg.get("mask_outside"),
        eval_cfg.get("mask_outside"),
        cfg.get("mask_outside"),
        defaults["mask_outside"],
    )
    args.mode = choose(args.mode, test_cfg.get("mode"), cfg.get("mode"), defaults["mode"])
    args.beam_size = choose(
        args.beam_size,
        test_cfg.get("beam_size"),
        eval_cfg.get("beam_size"),
        cfg.get("beam_size"),
        defaults["beam_size"],
    )
    args.leaf_prior_weight = choose(
        args.leaf_prior_weight,
        test_cfg.get("leaf_prior_weight"),
        eval_cfg.get("leaf_prior_weight"),
        cfg.get("leaf_prior_weight"),
        defaults["leaf_prior_weight"],
    )
    args.device = choose(args.device, test_cfg.get("device"), cfg.get("device"), defaults["device"])
    args.random_state = choose(args.random_state, test_cfg.get("random_state"), eval_cfg.get("random_state"), cfg.get("seed"), None)
    args.dataset = choose(args.dataset, test_cfg.get("dataset"), eval_cfg.get("dataset"), cfg.get("dataset"), None)
    if args.mode not in {"greedy", "beam"}:
        raise ValueError(f"invalid mode for scripts/test.py: {args.mode}")

    device = _resolve_device(args.device)

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

    split = load_dataset_split(
        dataset,
        random_state=random_state,
        val_size=float(cfg.get("val_size", 0.2)),
    )
    X_cat_val = split.X_cat_val
    cat_cardinalities = cfg.get("cat_cardinalities") or []
    n_cat_features = int(X_cat_val.shape[1]) if X_cat_val is not None else 0
    if n_cat_features == 0 and cat_cardinalities:
        raise ValueError("checkpoint expects categorical features but test data has none")
    if n_cat_features > 0:
        if not cat_cardinalities:
            raise ValueError("test data has categorical features but checkpoint has no cat_cardinalities")
        if len(cat_cardinalities) != n_cat_features:
            raise ValueError(
                f"len(cat_cardinalities)={len(cat_cardinalities)} must match X_cat_val.shape[1]={n_cat_features}"
            )

    encoder = TraceLabelEncoder(
        v_min=v_min,
        v_max=v_max,
        depth=depth,
        bin_edges=np.asarray(cfg.get("bin_edges"), dtype=np.float32) if cfg.get("bin_edges") is not None else None,
        binning_strategy=str(cfg.get("binning_strategy", "uniform")),
    )
    dataset_val = TabSeqDataset(
        X_num=split.X_val,
        X_cat=X_cat_val if X_cat_val is not None else np.zeros((len(split.y_val), 0), dtype=np.int64),
        y=split.y_val,
        encoder=encoder,
        is_train=False,
    )
    dataloader = DataLoader(dataset_val, batch_size=int(args.batch_size), shuffle=False)

    model_cls = _infer_model_class(ckpt["model_state_dict"])
    if model_cls is TransformerTabSeqModel:
        model = model_cls(
            n_num_features=n_num_features,
            depth=depth,
            n_bins=n_bins,
            cat_cardinalities=cat_cardinalities,
            encoder_type=str(cfg.get("encoder_type", "vanilla")),
            d_model=int(cfg.get("d_model", 64)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    metric_calc = ExtendedHolographicMetric(encoder)
    if args.mode == "beam":
        outputs = collect_beam_outputs(
            model=model,
            dl=dataloader,
            device=device,
            depth=depth,
            n_bins=n_bins,
            temperature=float(args.temperature),
            beam_size=int(args.beam_size),
            leaf_prior_weight=float(args.leaf_prior_weight),
            mask_outside=float(args.mask_outside),
            sos_token=2,
        )
        leaf_probs = outputs["leaf_probs"]
    else:
        outputs = collect_greedy_outputs(
            model=model,
            dl=dataloader,
            device=device,
            depth=depth,
            n_bins=n_bins,
            temperature=float(args.temperature),
            mask_outside=float(args.mask_outside),
            sos_token=2,
        )
        leaf_probs = metric_calc.leaf_probs_from_step_probs(outputs["model_probs"])

    if requested_confidence is None and confidence_grid:
        calibrated_metrics = calibrate_confidence_from_leaf_probs(
            metric_calc=metric_calc,
            leaf_probs=leaf_probs,
            y_true=outputs["y_raw"],
            y_leaf_idx=outputs["y_leaf_idx"],
            target_confidence=float(target_confidence),
            confidence_grid=confidence_grid,
            interval_method=str(args.interval_method),
            peak_merge_alpha=float(args.peak_merge_alpha),
            tolerance_bins=1,
        )
        args.confidence = float(calibrated_metrics["calibrated_confidence"])
        metrics = dict(calibrated_metrics)
    else:
        metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
            leaf_probs=leaf_probs,
            y_true=outputs["y_raw"],
            y_leaf_idx=outputs["y_leaf_idx"],
            confidence=float(args.confidence),
            interval_method=str(args.interval_method),
            peak_merge_alpha=float(args.peak_merge_alpha),
            tolerance_bins=1,
        )
    y_raw = outputs["y_raw"]

    payload = {
        "dataset": dataset,
        "split": "val",
        "mode": str(args.mode),
        "confidence": float(args.confidence),
        "target_confidence": float(target_confidence),
        "interval_method": str(args.interval_method),
        "n_samples": int(y_raw.shape[0]),
        "binning_strategy": str(cfg.get("binning_strategy", "uniform")),
        **metrics,
    }
    if args.mode == "beam":
        payload["beam_size"] = int(args.beam_size)
        payload["leaf_prior_weight"] = float(args.leaf_prior_weight)
    if str(args.interval_method) == "peak_merge":
        payload["peak_merge_alpha"] = float(args.peak_merge_alpha)

    out_dir = os.path.dirname(ckpt_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.interval_method) == "peak_merge":
        interval_tag = f"pm_a{_fmt_float(float(args.peak_merge_alpha))}"
    elif str(args.interval_method) == "relaxed_mass":
        interval_tag = "relaxed_mass"
    elif str(args.interval_method) == "shortest_mass":
        interval_tag = "shortest_mass"
    else:
        interval_tag = "cdf"
    if args.mode == "beam":
        mode_tag = f"beam_K{int(args.beam_size)}"
    else:
        mode_tag = "greedy"
    filename = f"metrics_test_{mode_tag}_T{_fmt_float(float(args.temperature))}_C{_fmt_float(float(args.confidence))}_{interval_tag}_{timestamp}.json"
    out_path = _safe_path(out_dir, filename)
    write_json(out_path, payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
