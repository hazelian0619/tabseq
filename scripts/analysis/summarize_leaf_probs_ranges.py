#!/usr/bin/env python3
"""
Summarize leaf_probs into y-range bins with n-style counts.

Outputs a markdown block per input:
  - y_true_count: number of samples whose y_true in range
  - soft_mass: sum of probabilities in range (expected count)
  - peak_count: number of samples whose argmax bin in range
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


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


def _parse_ranges(ranges: List[float]) -> List[Tuple[float, float]]:
    if len(ranges) < 2:
        raise ValueError("ranges must contain at least two numbers")
    out = []
    for i in range(len(ranges) - 1):
        out.append((ranges[i], ranges[i + 1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="npz files exported by eval/eval_prefix_search")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument(
        "--ranges",
        nargs="+",
        type=float,
        default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0],
        help="y-range boundaries (e.g., 0.0 0.4 0.8 ...)",
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all samples in file")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    ranges = _parse_ranges(args.ranges)

    for idx, path in enumerate(args.inputs):
        arrays, meta = _load_npz(path)
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

        soft_mass = leaf_probs.sum(axis=0)
        peak_idx = np.argmax(leaf_probs, axis=1)
        peak_centers = bin_centers[peak_idx]

        label = args.labels[idx] if args.labels else _default_label(path, meta)
        print(f"**口径: {label}**")
        print("| y_range | y_true_n | soft_mass (expected) | peak_n |")
        print("| --- | --- | --- | --- |")

        for lo, hi in ranges:
            y_mask = (y_true >= lo) & (y_true < hi)
            bin_mask = (bin_centers >= lo) & (bin_centers < hi)
            peak_mask = (peak_centers >= lo) & (peak_centers < hi)

            y_count = int(np.sum(y_mask))
            soft_sum = float(np.sum(soft_mass[bin_mask]))
            peak_count = int(np.sum(peak_mask))

            print(f"| [{lo:.1f}, {hi:.1f}) | {y_count} | {soft_sum:.1f} | {peak_count} |")
        print("")


if __name__ == "__main__":
    main()
