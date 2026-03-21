#!/usr/bin/env python3
"""
Build report-ready width-stratified coverage chart and table from metrics_val*.json.

This script is intended for P0 main figure:
- x: width bins
- y: coverage in each bin
- annotation: sample count n
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting") from exc


WIDTH_KEY_RE = re.compile(r"Width\s*\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\)")
VALUE_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+)(?:\s*\(n=(\d+)\))?")


@dataclass(frozen=True)
class WidthEntry:
    width_key: str
    lo: float
    hi: float
    coverage: float
    n: int


@dataclass(frozen=True)
class SeriesData:
    label: str
    path: str
    entries: List[WidthEntry]


def _default_label(path: str, payload: Dict) -> str:
    mode = payload.get("mode")
    if mode:
        return str(mode)
    return os.path.splitext(os.path.basename(path))[0]


def _parse_width_key(key: str) -> Tuple[float, float]:
    m = WIDTH_KEY_RE.search(key)
    if not m:
        raise ValueError(f"cannot parse width key: {key}")
    return float(m.group(1)), float(m.group(2))


def _parse_value(value: object) -> Tuple[float, int]:
    if isinstance(value, (int, float)):
        return float(value), 0
    text = str(value)
    m = VALUE_RE.search(text)
    if not m:
        raise ValueError(f"cannot parse width value: {value}")
    cov = float(m.group(1))
    n = int(m.group(2)) if m.group(2) is not None else 0
    return cov, n


def _load_series(path: str, label: Optional[str]) -> SeriesData:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    widths = payload.get("width_stratified_PICP") or {}
    if not isinstance(widths, dict) or not widths:
        raise ValueError(f"width_stratified_PICP missing or empty in {path}")

    entries: List[WidthEntry] = []
    for key, raw in widths.items():
        lo, hi = _parse_width_key(key)
        cov, n = _parse_value(raw)
        entries.append(WidthEntry(width_key=key, lo=lo, hi=hi, coverage=cov, n=n))

    entries.sort(key=lambda it: (it.lo, it.hi))
    final_label = label or _default_label(path, payload)
    return SeriesData(label=final_label, path=path, entries=entries)


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.1f}, {hi:.1f})"


def _collect_bins(series_list: Sequence[SeriesData]) -> List[Tuple[float, float]]:
    all_bins = set()
    for s in series_list:
        for e in s.entries:
            all_bins.add((e.lo, e.hi))
    return sorted(all_bins)


def _to_csv_rows(series_list: Sequence[SeriesData], bins: Sequence[Tuple[float, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for s in series_list:
        by_bin = {(e.lo, e.hi): e for e in s.entries}
        for lo, hi in bins:
            e = by_bin.get((lo, hi))
            rows.append(
                {
                    "label": s.label,
                    "metrics_path": s.path,
                    "width_bin": _bin_label(lo, hi),
                    "width_lo": lo,
                    "width_hi": hi,
                    "coverage": float(e.coverage) if e is not None else np.nan,
                    "n": int(e.n) if e is not None else 0,
                }
            )
    return rows


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    import csv

    if not rows:
        raise ValueError("no rows to write")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault(str(r["label"]), []).append(r)

    with open(path, "w", encoding="utf-8") as f:
        for label in sorted(groups.keys()):
            f.write(f"**口径: {label}**\n")
            f.write("| Width Bin | Coverage | N |\n")
            f.write("| --- | --- | --- |\n")
            for r in groups[label]:
                cov = r["coverage"]
                cov_text = "-" if np.isnan(float(cov)) else f"{float(cov):.4f}"
                f.write(f"| {r['width_bin']} | {cov_text} | {int(r['n'])} |\n")
            f.write("\n")


def _plot(series_list: Sequence[SeriesData], bins: Sequence[Tuple[float, float]], target_picp: float, title: str, out_png: str) -> None:
    labels = [_bin_label(lo, hi) for lo, hi in bins]
    x = np.arange(len(labels), dtype=float)

    num_series = len(series_list)
    width = min(0.82 / max(num_series, 1), 0.32)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, s in enumerate(series_list):
        by_bin = {(e.lo, e.hi): e for e in s.entries}
        y = []
        ns = []
        for b in bins:
            e = by_bin.get(b)
            y.append(np.nan if e is None else e.coverage)
            ns.append(0 if e is None else e.n)

        offset = (i - (num_series - 1) / 2.0) * width
        bars = ax.bar(x + offset, y, width=width, label=s.label, alpha=0.9)

        for bar, n, val in zip(bars, ns, y):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(float(val) + 0.015, 0.995),
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.axhline(float(target_picp), color="#d62728", linestyle="--", linewidth=1.3, label=f"target={target_picp:.2f}")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Coverage (PICP)")
    ax.set_xlabel("Interval width bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="metrics_val*.json files")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels, same length as inputs")
    ap.add_argument("--target-picp", type=float, default=0.9)
    ap.add_argument("--title", type=str, default="Width-stratified coverage (full validation)")
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    series_list = []
    for i, path in enumerate(args.inputs):
        label = args.labels[i] if args.labels else None
        series_list.append(_load_series(path, label=label))

    bins = _collect_bins(series_list)
    rows = _to_csv_rows(series_list, bins)

    out_csv = args.out_csv or os.path.splitext(args.out_png)[0] + ".csv"
    _write_csv(out_csv, rows)

    if args.out_md:
        _write_md(args.out_md, rows)

    _plot(series_list, bins=bins, target_picp=float(args.target_picp), title=str(args.title), out_png=args.out_png)

    print("saved:", args.out_png)
    print("saved:", out_csv)
    if args.out_md:
        print("saved:", args.out_md)


if __name__ == "__main__":
    main()
