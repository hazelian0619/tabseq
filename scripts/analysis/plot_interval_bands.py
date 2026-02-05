#!/usr/bin/env python3
"""
Plot interval bands per sample (sorted), to visualize shape + sample counts.

Example:
  python3 scripts/plot_interval_bands.py \
    --inputs leaf_probs_val_prefix_beam_*.npz \
    --labels beam16 \
    --method cdf --confidence 0.9 \
    --sort-by y_true \
    --out outputs/analysis/interval_bands_beam16.png
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

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


def _interval_cdf(leaf_probs: np.ndarray, bin_edges: np.ndarray, confidence: float) -> Tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(leaf_probs, axis=1)
    alpha = 1.0 - float(confidence)
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)
    lower_idx = (cdf >= lower_q).argmax(axis=1)
    upper_idx = (cdf >= upper_q).argmax(axis=1)
    L = bin_edges[lower_idx]
    U = bin_edges[upper_idx + 1]
    return L, U


def _interval_peak_merge(
    leaf_probs: np.ndarray, bin_edges: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    n = leaf_probs.shape[0]
    L = np.empty(n, dtype=np.float32)
    U = np.empty(n, dtype=np.float32)
    for i in range(n):
        probs = leaf_probs[i]
        peak_idx = int(np.argmax(probs))
        peak_val = float(probs[peak_idx])
        thresh = peak_val * float(alpha)
        left = peak_idx
        right = peak_idx
        while left - 1 >= 0 and probs[left - 1] >= thresh:
            left -= 1
        while right + 1 < probs.shape[0] and probs[right + 1] >= thresh:
            right += 1
        L[i] = bin_edges[left]
        U[i] = bin_edges[right + 1]
    return L, U


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files exported by eval/eval_prefix_search")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--method", choices=["cdf", "peak_merge"], default="cdf")
    ap.add_argument("--confidence", type=float, default=0.9, help="CDF confidence (if method=cdf)")
    ap.add_argument("--alpha", type=float, default=0.33, help="peak-merge threshold ratio (if method=peak_merge)")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument("--sort-by", choices=["y_true", "y_pred", "none"], default="y_true")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all samples in file")
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    loaded = []
    for path in args.inputs:
        arrays, meta = _load_npz(path)
        loaded.append((path, arrays, meta))

    n_plots = len(loaded)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, max(2.5, 2.2 * n_plots)), sharex=True)
    if n_plots == 1:
        axes = [axes]

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
        else:
            raise ValueError(f"missing bin_edges in {path}")
        bin_centers = arrays.get("bin_centers")
        if bin_centers is None:
            bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1])

        y_pred = (leaf_probs * bin_centers[None, :]).sum(axis=1)

        if args.method == "cdf":
            L, U = _interval_cdf(leaf_probs, bin_edges, args.confidence)
        else:
            L, U = _interval_peak_merge(leaf_probs, bin_edges, args.alpha)

        if args.sort_by == "y_true":
            order = np.argsort(y_true)
        elif args.sort_by == "y_pred":
            order = np.argsort(y_pred)
        else:
            order = np.arange(len(y_true))

        L = L[order]
        U = U[order]
        y_true_sorted = y_true[order]

        ax = axes[i]
        y_idx = np.arange(len(y_true_sorted))
        ax.hlines(y_idx, L, U, color="#1f77b4", alpha=0.6, linewidth=1.0)
        ax.scatter(y_true_sorted, y_idx, s=6, color="#d62728", alpha=0.7)

        covered = (y_true_sorted >= L) & (y_true_sorted <= U)
        picp = float(np.mean(covered)) if len(covered) else 0.0
        mpiw = float(np.mean(U - L)) if len(covered) else 0.0

        label = args.labels[i] if args.labels else _default_label(path, meta)
        if args.method == "cdf":
            title = f"{label} | method=cdf | PICP={picp:.3f} | MPIW={mpiw:.3f}"
        else:
            title = f"{label} | method=peak_merge(alpha={args.alpha:.2f}) | PICP={picp:.3f} | MPIW={mpiw:.3f}"
        ax.set_title(title)
        ax.set_ylabel("sample idx")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("y (value)")
    fig.tight_layout()
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
