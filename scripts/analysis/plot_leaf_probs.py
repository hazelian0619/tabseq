#!/usr/bin/env python3
"""
Plot leaf_probs distributions for multiple modes (greedy/beam/full) side-by-side.

Typical usage:
  python3 scripts/plot_leaf_probs.py \
    --inputs leaf_probs_val_greedy_*.npz leaf_probs_val_prefix_beam_*.npz \
    --labels greedy beam16 \
    --sample-idx 0 \
    --out outputs/analysis/leaf_probs_compare.png
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files exported by eval/eval_prefix_search")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--sample-idx", type=int, default=0, help="sample index to plot")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    loaded = []
    for path in args.inputs:
        arrays, meta = _load_npz(path)
        loaded.append((path, arrays, meta))

    sample_idx = int(args.sample_idx)
    for path, arrays, _ in loaded:
        if "leaf_probs" not in arrays:
            raise ValueError(f"missing leaf_probs in {path}")
        if sample_idx < 0 or sample_idx >= arrays["leaf_probs"].shape[0]:
            raise ValueError(f"sample-idx {sample_idx} out of range for {path}")

    ref_y = None
    for _, arrays, _ in loaded:
        y_vals = _select_y(arrays, args.y_source)
        if y_vals is None:
            continue
        y_here = float(y_vals[sample_idx])
        if ref_y is None:
            ref_y = y_here
        elif abs(ref_y - y_here) > 1e-4:
            print(
                f"warning: y_{args.y_source} differs across files at idx={sample_idx} "
                f"(ref={ref_y:.6f}, got={y_here:.6f})"
            )

    n = len(loaded)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (path, arrays, meta) in enumerate(loaded):
        leaf_probs = arrays["leaf_probs"][sample_idx]
        if "bin_centers" in arrays:
            x = arrays["bin_centers"]
        elif "bin_edges" in arrays:
            edges = arrays["bin_edges"]
            x = edges[:-1] + 0.5 * (edges[1:] - edges[:-1])
        else:
            x = np.arange(len(leaf_probs), dtype=np.float32)
        label = args.labels[i] if args.labels else _default_label(path, meta)
        ax = axes[i]
        ax.plot(x, leaf_probs, color="#1f77b4", linewidth=1.5)
        y_vals = _select_y(arrays, args.y_source)
        if y_vals is not None:
            ax.axvline(float(y_vals[sample_idx]), color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_ylabel("prob")
        ax.set_title(label)

    axes[-1].set_xlabel("y (bin center)")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
