#!/usr/bin/env python3
"""
Summarize width_stratified_PICP blocks from metrics JSON files.

Outputs markdown blocks in the same style as the baseline (Width [...): value (n=...)).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def _default_label(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _width_key_sort_key(key: str) -> Tuple[float, float]:
    nums = re.findall(r"[-+]?[0-9]*\\.?[0-9]+", key)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return (0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="metrics_val*.json files")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    for idx, path in enumerate(args.inputs):
        data = json.load(open(path))
        label = args.labels[idx] if args.labels else _default_label(path)
        widths = data.get("width_stratified_PICP") or {}

        print(f"**口径: {label}**")
        if not widths:
            print("- (no width_stratified_PICP)")
            print()
            continue

        for k in sorted(widths.keys(), key=_width_key_sort_key):
            print(f"- {k}: {widths[k]}")
        print()


if __name__ == "__main__":
    main()
