#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.training import TrainConfig, train_tabseq_model
from tabseq.utils.config import choose, load_config, resolve_section


def _parse_optional_float_grid(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    else:
        values = [float(raw)]
    return tuple(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-size", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--binning-strategy", choices=["uniform", "quantile"], default=None)
    ap.add_argument("--encoder-type", type=str, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--confidence-grid", type=str, default=None, help="comma-separated inference confidences for val calibration")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--interval-method", choices=["cdf", "shortest_mass", "relaxed_mass", "peak_merge"], default=None)
    ap.add_argument("--peak-merge-alpha", type=float, default=None)
    ap.add_argument("--mask-outside", type=float, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--leaf-prior-weight", type=float, default=None)
    ap.add_argument("--selection-warmup-epochs", type=int, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--bin-step-02", type=float, default=None)
    ap.add_argument("--bin-step-04", type=float, default=None)
    ap.add_argument("--width-bins", type=str, default=None, help="comma-separated width bins")
    args = ap.parse_args()

    defaults = TrainConfig()
    config = load_config(args.config)
    train_cfg = resolve_section(config, "train")
    eval_cfg = resolve_section(config, "eval")

    width_bins = choose(args.width_bins, train_cfg.get("width_bins"), list(defaults.width_bins))
    if isinstance(width_bins, str):
        width_bins = [float(x.strip()) for x in width_bins.split(",") if x.strip()]
    elif isinstance(width_bins, tuple):
        width_bins = list(width_bins)
    confidence_grid = _parse_optional_float_grid(
        choose(args.confidence_grid, train_cfg.get("confidence_grid"), eval_cfg.get("confidence_grid"), defaults.confidence_grid)
    )

    cfg = TrainConfig(
        dataset=str(choose(args.dataset, train_cfg.get("dataset"), defaults.dataset)),
        seed=int(choose(args.seed, train_cfg.get("seed"), defaults.seed)),
        val_size=float(choose(args.val_size, train_cfg.get("val_size"), defaults.val_size)),
        batch_size=int(choose(args.batch_size, train_cfg.get("batch_size"), defaults.batch_size)),
        epochs=int(choose(args.epochs, train_cfg.get("epochs"), defaults.epochs)),
        lr=float(choose(args.lr, train_cfg.get("lr"), defaults.lr)),
        weight_decay=float(choose(args.weight_decay, train_cfg.get("weight_decay"), defaults.weight_decay)),
        depth=int(choose(args.depth, train_cfg.get("depth"), defaults.depth)),
        binning_strategy=str(choose(args.binning_strategy, train_cfg.get("binning_strategy"), defaults.binning_strategy)),
        encoder_type=str(choose(args.encoder_type, train_cfg.get("encoder_type"), defaults.encoder_type)),
        d_model=int(choose(args.d_model, train_cfg.get("d_model"), defaults.d_model)),
        n_heads=int(choose(args.n_heads, train_cfg.get("n_heads"), defaults.n_heads)),
        n_layers=int(choose(args.n_layers, train_cfg.get("n_layers"), defaults.n_layers)),
        dropout=float(choose(args.dropout, train_cfg.get("dropout"), defaults.dropout)),
        confidence=float(choose(args.confidence, train_cfg.get("confidence"), defaults.confidence)),
        confidence_grid=tuple(confidence_grid or ()),
        temperature=float(choose(args.temperature, train_cfg.get("temperature"), eval_cfg.get("temperature"), defaults.temperature)),
        interval_method=str(
            choose(args.interval_method, train_cfg.get("interval_method"), eval_cfg.get("interval_method"), defaults.interval_method)
        ),
        peak_merge_alpha=float(
            choose(
                args.peak_merge_alpha,
                train_cfg.get("peak_merge_alpha"),
                eval_cfg.get("peak_merge_alpha"),
                defaults.peak_merge_alpha,
            )
        ),
        mask_outside=float(
            choose(args.mask_outside, train_cfg.get("mask_outside"), eval_cfg.get("mask_outside"), defaults.mask_outside)
        ),
        beam_size=int(choose(args.beam_size, train_cfg.get("beam_size"), eval_cfg.get("beam_size"), defaults.beam_size)),
        leaf_prior_weight=float(
            choose(
                args.leaf_prior_weight,
                train_cfg.get("leaf_prior_weight"),
                eval_cfg.get("leaf_prior_weight"),
                defaults.leaf_prior_weight,
            )
        ),
        selection_warmup_epochs=int(
            choose(args.selection_warmup_epochs, train_cfg.get("selection_warmup_epochs"), defaults.selection_warmup_epochs)
        ),
        out_root=str(choose(args.out_root, train_cfg.get("out_root"), defaults.out_root)),
        run_id=choose(args.run_id, train_cfg.get("run_id"), defaults.run_id),
        device=choose(args.device, train_cfg.get("device"), defaults.device),
        num_workers=int(choose(args.num_workers, train_cfg.get("num_workers"), defaults.num_workers)),
        bin_step_02=float(choose(args.bin_step_02, train_cfg.get("bin_step_02"), defaults.bin_step_02)),
        bin_step_04=float(choose(args.bin_step_04, train_cfg.get("bin_step_04"), defaults.bin_step_04)),
        width_bins=tuple(float(x) for x in width_bins),
    )

    run_dir = train_tabseq_model(cfg)
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
