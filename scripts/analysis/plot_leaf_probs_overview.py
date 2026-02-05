#!/usr/bin/env python3
"""
Plot leaf_probs heatmap + sample-count curves (top-k/top-p/peak-merge) in one figure.

Example:
  PYTHONPATH=src python3 scripts/plot_leaf_probs_overview.py \
    --inputs outputs/california_housing/run_20260202_112549/leaf_probs_val_prefix_beam_K16_T1_C0p9_Nall_*_E200.npz \
    --labels beam16 \
    --top-k 3 5 \
    --top-p 0.8 0.9 \
    --peak-merge-alpha 0.33 \
    --out outputs/analysis/leaf_probs_overview_beam16.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

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


def _top_k_counts(leaf_probs: np.ndarray, k: int) -> np.ndarray:
    n, m = leaf_probs.shape
    k = min(int(k), m)
    counts = np.zeros(m, dtype=np.int64)
    if k <= 0:
        return counts
    for i in range(n):
        idx = np.argpartition(-leaf_probs[i], k - 1)[:k]
        counts[idx] += 1
    return counts


def _top_p_counts(leaf_probs: np.ndarray, p: float) -> np.ndarray:
    n, m = leaf_probs.shape
    p = float(p)
    counts = np.zeros(m, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-leaf_probs[i])
        csum = np.cumsum(leaf_probs[i][order])
        cutoff = int(np.searchsorted(csum, p, side="left"))
        cutoff = min(cutoff, m - 1)
        idx = order[: cutoff + 1]
        counts[idx] += 1
    return counts


def _peak_merge_counts(leaf_probs: np.ndarray, alpha: float) -> np.ndarray:
    n, m = leaf_probs.shape
    alpha = float(alpha)
    counts = np.zeros(m, dtype=np.int64)
    for i in range(n):
        probs = leaf_probs[i]
        peak_idx = int(np.argmax(probs))
        thresh = probs[peak_idx] * alpha
        left = peak_idx
        right = peak_idx
        while left - 1 >= 0 and probs[left - 1] >= thresh:
            left -= 1
        while right + 1 < m and probs[right + 1] >= thresh:
            right += 1
        counts[left : right + 1] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files exported by eval/eval_prefix_search")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument("--sort-by", choices=["y_true", "y_pred", "none"], default="y_true")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all samples in file")
    ap.add_argument("--top-k", nargs="*", type=int, default=[], help="list of k for top-k regions")
    ap.add_argument("--top-p", nargs="*", type=float, default=[], help="list of p for top-p regions")
    ap.add_argument("--peak-merge-alpha", nargs="*", type=float, default=[], help="list of alpha for peak-merge regions")
    ap.add_argument("--no-soft-mass", action="store_true", help="disable soft mass curve")
    ap.add_argument("--show-y-hist", action="store_true", help="overlay y_true histogram")
    ap.add_argument("--vmax", type=float, default=None, help="fix heatmap vmax (shared across inputs)")
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    loaded = []
    for path in args.inputs:
        arrays, meta = _load_npz(path)
        loaded.append((path, arrays, meta))

    global_vmax = args.vmax
    if global_vmax is None:
        global_vmax = max(float(np.max(arrays["leaf_probs"])) for _, arrays, _ in loaded)

    n_plots = len(loaded)
    fig, axes = plt.subplots(
        nrows=2 * n_plots,
        ncols=1,
        figsize=(11, max(4.0, 3.0 * n_plots)),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1] * n_plots},
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

        y_pred = (leaf_probs * bin_centers[None, :]).sum(axis=1)

        if args.sort_by == "y_true":
            order = np.argsort(y_true)
        elif args.sort_by == "y_pred":
            order = np.argsort(y_pred)
        else:
            order = np.arange(len(y_true))

        leaf_probs_sorted = leaf_probs[order]

        ax_heat = axes[2 * i]
        extent = [float(bin_edges[0]), float(bin_edges[-1]), 0, leaf_probs_sorted.shape[0]]
        im = ax_heat.imshow(
            leaf_probs_sorted,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=global_vmax,
        )
        label = args.labels[i] if args.labels else _default_label(path, meta)
        ax_heat.set_title(f"{label} | heatmap (sorted by {args.sort_by})")
        ax_heat.set_ylabel("sample idx")
        fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)

        ax_count = axes[2 * i + 1]
        if not args.no_soft_mass:
            soft_mass = leaf_probs.sum(axis=0)
            ax_count.plot(bin_centers, soft_mass, label="soft_mass (expected count)", linewidth=1.5)

        colors = plt.cm.tab10.colors
        color_idx = 0

        for k in args.top_k:
            counts = _top_k_counts(leaf_probs, k)
            ax_count.plot(
                bin_centers,
                counts,
                label=f"top_k={k}",
                linewidth=1.2,
                color=colors[color_idx % len(colors)],
            )
            color_idx += 1

        for p in args.top_p:
            counts = _top_p_counts(leaf_probs, p)
            ax_count.plot(
                bin_centers,
                counts,
                label=f"top_p={p:g}",
                linewidth=1.2,
                color=colors[color_idx % len(colors)],
            )
            color_idx += 1

        for alpha in args.peak_merge_alpha:
            counts = _peak_merge_counts(leaf_probs, alpha)
            ax_count.plot(
                bin_centers,
                counts,
                label=f"peak_merge_a={alpha:g}",
                linewidth=1.2,
                color=colors[color_idx % len(colors)],
                linestyle="--",
            )
            color_idx += 1

        if args.show_y_hist:
            y_hist, _ = np.histogram(y_true, bins=bin_edges)
            ax_count.plot(bin_centers, y_hist, label="y_true_hist", linewidth=1.0, linestyle=":")

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
