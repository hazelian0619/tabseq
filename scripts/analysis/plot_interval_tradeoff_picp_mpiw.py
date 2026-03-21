#!/usr/bin/env python3
"""
Build report-ready PICP-MPIW tradeoff chart from metrics_val*.json.

This script is intended as the supplementary main chart:
- x: MPIW
- y: PICP
- label each point by mode/label
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting") from exc


@dataclass(frozen=True)
class TradeoffPoint:
    label: str
    mode: str
    path: str
    picp: float
    mpiw: float
    confidence: float


def _default_label(path: str, payload: Dict) -> str:
    mode = payload.get("mode")
    if mode:
        return str(mode)
    return os.path.splitext(os.path.basename(path))[0]


def _family(label: str, mode: str) -> str:
    text = (label + " " + mode).lower()
    if "greedy" in text:
        return "greedy"
    if "beam" in text:
        return "beam"
    if "full" in text:
        return "full"
    if "teacher" in text:
        return "teacher"
    return "other"


def _load_point(path: str, label: Optional[str]) -> TradeoffPoint:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Skip nested wrapper metrics files (e.g. metrics_val_prefix.json with subkeys)
    if "PICP" not in payload or "MPIW" not in payload:
        raise ValueError(f"PICP/MPIW missing in {path}; use concrete metrics_val_<mode>*.json")

    final_label = label or _default_label(path, payload)
    mode = str(payload.get("mode", "unknown"))
    return TradeoffPoint(
        label=final_label,
        mode=mode,
        path=path,
        picp=float(payload["PICP"]),
        mpiw=float(payload["MPIW"]),
        confidence=float(payload.get("confidence", np.nan)),
    )


def _write_csv(path: str, points: Sequence[TradeoffPoint]) -> None:
    rows = [
        {
            "label": p.label,
            "mode": p.mode,
            "metrics_path": p.path,
            "PICP": p.picp,
            "MPIW": p.mpiw,
            "confidence": p.confidence,
        }
        for p in points
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(points: Sequence[TradeoffPoint], target_picp: float, title: str, out_png: str) -> None:
    colors = {
        "greedy": "#1f77b4",
        "beam": "#ff7f0e",
        "full": "#2ca02c",
        "teacher": "#9467bd",
        "other": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for p in points:
        fam = _family(p.label, p.mode)
        ax.scatter(p.mpiw, p.picp, s=68, color=colors[fam], edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(p.label, (p.mpiw, p.picp), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.axhline(float(target_picp), color="#d62728", linestyle="--", linewidth=1.2, label=f"target={target_picp:.2f}")

    x_vals = [p.mpiw for p in points]
    x_min = min(x_vals)
    x_max = max(x_vals)
    pad = max((x_max - x_min) * 0.08, 0.02)
    ax.set_xlim(max(0.0, x_min - pad), x_max + pad)
    ax.set_ylim(0.0, 1.02)

    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9)

    # Add a small family legend using dummy points.
    for fam in ["greedy", "beam", "full", "teacher"]:
        ax.scatter([], [], s=60, color=colors[fam], edgecolors="black", linewidths=0.5, label=fam)
    handles, labels = ax.get_legend_handles_labels()
    # Keep unique labels in order.
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", fontsize=8)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="metrics_val*.json files")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels, same length as inputs")
    ap.add_argument("--target-picp", type=float, default=0.9)
    ap.add_argument("--title", default="PICP-MPIW tradeoff (full validation)")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    points: List[TradeoffPoint] = []
    for i, path in enumerate(args.inputs):
        label = args.labels[i] if args.labels else None
        points.append(_load_point(path, label=label))

    points.sort(key=lambda p: p.mpiw)

    out_csv = args.out_csv or os.path.splitext(args.out_png)[0] + ".csv"
    _write_csv(out_csv, points)
    _plot(points, target_picp=float(args.target_picp), title=str(args.title), out_png=args.out_png)

    print("saved:", args.out_png)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
