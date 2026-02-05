#!/usr/bin/env python3
"""
Plot aggregated leaf_probs shape + sample-count curves (no top-k/top-p).

Example:
  PYTHONPATH=src python3 scripts/plot_leaf_probs_summary.py \
    --inputs outputs/california_housing/run_20260202_112549/leaf_probs_val_greedy_*.npz \
            outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_*.npz \
    --labels greedy beam16 \
    --show-y-hist \
    --out outputs/analysis/leaf_probs_summary_greedy_vs_beam16.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required for plotting.") from exc


def _load_npz(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    data = np.load(path, allow_pickle=True)
    meta_raw = data["meta"] if "meta" in data.files else None
    meta: Dict[str, object] = {}
    if meta_raw is not None:
        try:
            meta = json.loads(meta_raw.item())
        except Exception:
            meta = {"raw": str(meta_raw)}
    arrays = {k: data[k] for k in data.files if k != "meta"}
    return arrays, meta


def _default_label(path: str, meta: Dict[str, object]) -> str:
    if meta and "mode" in meta:
        return str(meta["mode"])
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _select_y(arrays: Dict[str, np.ndarray], y_source: str) -> np.ndarray:
    if y_source == "raw":
        return arrays.get("y_raw", arrays.get("y_clipped"))
    return arrays.get("y_clipped", arrays.get("y_raw"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files exported by eval/eval_prefix_search")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all samples in file")
    ap.add_argument("--band-low", type=float, default=0.1, help="lower percentile for band")
    ap.add_argument("--band-high", type=float, default=0.9, help="upper percentile for band")
    ap.add_argument("--show-median", action="store_true", help="plot median line")
    ap.add_argument("--show-y-hist", action="store_true", help="overlay y_true histogram on count panel")
    ap.add_argument("--count-mode", choices=["soft", "peak", "both"], default="soft")
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    loaded = []
    for path in args.inputs:
        arrays, meta = _load_npz(path)
        loaded.append((path, arrays, meta))

    n_plots = len(loaded)
    fig, axes = plt.subplots(
        nrows=2 * n_plots,
        ncols=1,
        figsize=(11, max(4.0, 3.0 * n_plots)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1] * n_plots},
    )
    if n_plots == 1:
        axes = [axes[0], axes[1]]

    for i, (path, arrays, meta) in enumerate(loaded):
        leaf_probs = arrays["leaf_probs"]
        if args.max_samples and args.max_samples > 0:
            leaf_probs = leaf_probs[: args.max_samples]

        y_true = _select_y(arrays, args.y_source)
        if y_true is None:
            raise ValueError(f"missing y in {path}")
        if args.max_samples and args.max_samples > 0:
            y_true = y_true[: args.max_samples]

        if "bin_edges" in arrays:
            bin_edges = arrays["bin_edges"]
        elif "bin_centers" in arrays:
            bin_centers = arrays["bin_centers"]
            bin_edges = np.concatenate(
                [bin_centers[:1], 0.5 * (bin_centers[:-1] + bin_centers[1:]), bin_centers[-1:]]
            )
        else:
            raise ValueError(f"missing bin_edges/bin_centers in {path}")

        bin_centers = arrays.get("bin_centers")
        if bin_centers is None:
            bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1])

        mean_probs = leaf_probs.mean(axis=0)
        low_q, high_q = np.percentile(leaf_probs, [args.band_low * 100, args.band_high * 100], axis=0)
        median = np.median(leaf_probs, axis=0)

        label = args.labels[i] if args.labels else _default_label(path, meta)
        ax_shape = axes[2 * i]
        ax_shape.fill_between(bin_centers, low_q, high_q, color="#1f77b4", alpha=0.2, label="p10-p90")
        ax_shape.plot(bin_centers, mean_probs, color="#1f77b4", linewidth=1.8, label="mean")
        if args.show_median:
            ax_shape.plot(bin_centers, median, color="#2ca02c", linewidth=1.2, linestyle="--", label="median")
        ax_shape.set_title(f"{label} | mean shape + band")
        ax_shape.set_ylabel("prob")
        ax_shape.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax_shape.legend(loc="upper right", fontsize=8, ncol=2)

        ax_count = axes[2 * i + 1]
        if args.count_mode in ("soft", "both"):
            soft_mass = leaf_probs.sum(axis=0)
            ax_count.plot(bin_centers, soft_mass, color="#ff7f0e", linewidth=1.6, label="soft_mass")
        if args.count_mode in ("peak", "both"):
            peak_idx = np.argmax(leaf_probs, axis=1)
            peak_counts = np.bincount(peak_idx, minlength=leaf_probs.shape[1])
            ax_count.plot(bin_centers, peak_counts, color="#9467bd", linewidth=1.2, label="peak_count")
        if args.show_y_hist:
            y_hist, _ = np.histogram(y_true, bins=bin_edges)
            ax_count.bar(bin_centers, y_hist, width=bin_edges[1:] - bin_edges[:-1], alpha=0.25, label="y_true_hist")
        ax_count.set_ylabel("count")
        ax_count.set_xlabel("y (bin center)")
        ax_count.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax_count.legend(loc="upper right", fontsize=8, ncol=2)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
