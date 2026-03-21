#!/usr/bin/env python3
"""
Refined California main-figure variants with cleaner aesthetics and clearer labels.

Fixes compared with previous variants:
- Resolve overlapping labels in clustered greedy points.
- Make n-value differences explicit and perceptually strong.
- Keep pastel, high-lightness palette (blue/teal/green).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required") from exc


WIDTH_KEY_RE = re.compile(r"Width\s*\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\)")
VALUE_RE = re.compile(r"([-+]?[0-9]*\.?[0-9]+)(?:\s*\(n=(\d+)\))?")


@dataclass(frozen=True)
class TradeoffPoint:
    label: str
    mode: str
    picp: float
    mpiw: float


@dataclass(frozen=True)
class WidthEntry:
    label: str
    width_bin: str
    lo: float
    hi: float
    coverage: float
    n: int


COLORS = {
    "bg": "#F7FAFC",
    "grid": "#D8E5EF",
    "target": "#E07A8A",
    "greedy": "#8EC5E8",
    "beam": "#A7DCCF",
    "full": "#B9E3B2",
    "teacher": "#CABBE9",
    "text": "#2E3A46",
    "outline": "#5F7A8C",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": "#C4D4E0",
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "font.size": 11,
        }
    )


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Avoid clipping rotated ticklabels / annotations.
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor=COLORS["bg"])
    plt.close(fig)
    print("saved:", path)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _family(label: str, mode: str) -> str:
    t = (label + " " + mode).lower()
    if "greedy" in t:
        return "greedy"
    if "beam" in t:
        return "beam"
    if "full" in t:
        return "full"
    if "teacher" in t:
        return "teacher"
    return "greedy"


def _parse_width_key(key: str) -> Tuple[float, float]:
    m = WIDTH_KEY_RE.search(key)
    if not m:
        raise ValueError(f"cannot parse width key: {key}")
    return float(m.group(1)), float(m.group(2))


def _parse_width_value(raw: object) -> Tuple[float, int]:
    if isinstance(raw, (int, float)):
        return float(raw), 0
    m = VALUE_RE.search(str(raw))
    if not m:
        raise ValueError(f"cannot parse width value: {raw}")
    cov = float(m.group(1))
    n = int(m.group(2)) if m.group(2) else 0
    return cov, n


def _load_tradeoff(paths: Sequence[str], labels: Sequence[str]) -> List[TradeoffPoint]:
    out: List[TradeoffPoint] = []
    for p, l in zip(paths, labels):
        d = _load_json(p)
        if "PICP" not in d or "MPIW" not in d:
            continue
        out.append(TradeoffPoint(label=l, mode=str(d.get("mode", "unknown")), picp=float(d["PICP"]), mpiw=float(d["MPIW"])))
    return out


def _load_width(paths: Sequence[str], labels: Sequence[str]) -> List[WidthEntry]:
    out: List[WidthEntry] = []
    for p, l in zip(paths, labels):
        d = _load_json(p)
        ws = d.get("width_stratified_PICP") or {}
        for k, v in ws.items():
            lo, hi = _parse_width_key(k)
            cov, n = _parse_width_value(v)
            out.append(WidthEntry(label=l, width_bin=f"[{lo:.1f}, {hi:.1f})", lo=lo, hi=hi, coverage=cov, n=n))
    return out


def _annotate_tradeoff(ax: plt.Axes, p: TradeoffPoint) -> None:
    # Hand-tuned offsets for clustered labels
    offsets = {
        "greedy_m0.1": (5, 6),
        "greedy_m0.3": (6, 1),
        "greedy_m0.5": (6, 7),
        "greedy_m1": (6, 12),
        "beam16": (6, 6),
        "beam32": (6, 6),
        "full": (6, 6),
        "teacher_tf": (6, 6),
    }
    dx, dy = offsets.get(p.label, (6, 6))
    ax.annotate(p.label, (p.mpiw, p.picp), xytext=(dx, dy), textcoords="offset points", fontsize=10)


def plot_tradeoff_refined(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.grid(True, linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.65)

    for p in points:
        fam = _family(p.label, p.mode)
        ax.scatter(p.mpiw, p.picp, s=125, color=COLORS[fam], edgecolor=COLORS["outline"], linewidth=0.85, zorder=3)
        _annotate_tradeoff(ax, p)

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.6)
    ax.text(max(p.mpiw for p in points), target + 0.01, f"target={target:.2f}", ha="right", va="bottom", fontsize=10)

    # Add zoom window guide derived from the greedy cluster (avoid hard-coded bounds).
    greedy = [p for p in points if _family(p.label, p.mode) == "greedy"]
    if greedy:
        zx = [p.mpiw for p in greedy]
        zy = [p.picp for p in greedy]
        x0, x1 = min(zx), max(zx)
        y0, y1 = min(zy), max(zy)
        pad_x = max(0.02, (x1 - x0) * 0.35)
        pad_y = max(0.02, (y1 - y0) * 0.35)
        xlo, xhi = x0 - pad_x, x1 + pad_x
        ylo, yhi = y0 - pad_y, y1 + pad_y
        zoom_rect = plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, edgecolor="#9CB3C4", linewidth=1.1, linestyle=":")
        ax.add_patch(zoom_rect)
        ax.text(xlo, yhi, "greedy cluster", fontsize=9, color="#6B8597", va="bottom", ha="left")

    xs = [p.mpiw for p in points]
    xmin, xmax = min(xs), max(xs)
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | PICP-MPIW tradeoff (refined pastel)")

    legend = [
        Line2D([0], [0], marker="o", color="w", label="greedy", markerfacecolor=COLORS["greedy"], markeredgecolor=COLORS["outline"], markersize=9),
        Line2D([0], [0], marker="o", color="w", label="beam", markerfacecolor=COLORS["beam"], markeredgecolor=COLORS["outline"], markersize=9),
        Line2D([0], [0], marker="o", color="w", label="full", markerfacecolor=COLORS["full"], markeredgecolor=COLORS["outline"], markersize=9),
        Line2D([0], [0], marker="o", color="w", label="teacher", markerfacecolor=COLORS["teacher"], markeredgecolor=COLORS["outline"], markersize=9),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.88)
    _save(fig, out_path)


def _pick_width_groups(entries: Sequence[WidthEntry]) -> List[str]:
    labels = sorted({e.label for e in entries})
    if not labels:
        return []

    def pick(preferred: Sequence[str], contains: str) -> Optional[str]:
        for p in preferred:
            if p in labels:
                return p
        for l in labels:
            if contains in l:
                return l
        return None

    greedy = pick(["greedy_m1", "greedy", "greedy_m0.5", "greedy_m0.3", "greedy_m0.1"], "greedy")
    beam = pick(["beam32", "beam16", "beam"], "beam")
    full = pick(["full", "prefix_full"], "full")
    teacher = pick(["teacher_tf", "teacher"], "teacher")
    out = [x for x in [greedy, beam, full, teacher] if x]
    if len(out) < min(3, len(labels)):
        out = labels[: min(4, len(labels))]
    return out


def plot_width_refined(entries: Sequence[WidthEntry], target: float, out_path: str) -> None:
    _set_style()
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.8, 1.8]})

    bins = sorted({(e.lo, e.hi, e.width_bin) for e in entries}, key=lambda t: (t[0], t[1]))
    labels = [b[2] for b in bins]
    x = np.arange(len(labels), dtype=float)
    groups = _pick_width_groups(entries)
    bar_w = 0.24
    by = {(e.label, e.width_bin): e for e in entries}

    # Top panel: coverage
    for i, g in enumerate(groups):
        fam = _family(g, g)
        ys = []
        ns = []
        for b in labels:
            e = by.get((g, b))
            ys.append(np.nan if e is None else e.coverage)
            ns.append(0 if e is None else e.n)
        offset = (i - 1) * bar_w
        face = COLORS.get(fam, COLORS["greedy"])
        edgecolors = ["#B25A6B" if (n > 0 and n <= 10) else COLORS["outline"] for n in ns]
        lws = [1.2 if (n > 0 and n <= 10) else 0.6 for n in ns]
        bars = ax_top.bar(x + offset, ys, width=bar_w, color=face, edgecolor=edgecolors, linewidth=lws, label=g)
        for bar, n, yv in zip(bars, ns, ys):
            if np.isnan(yv):
                continue
            if n > 0 and (n <= 10 or n >= 500):
                ax_top.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yv + 0.015,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.6),
                    clip_on=False,
                )

    ax_top.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax_top.set_ylim(0, 1.08)
    ax_top.set_ylabel("Coverage (PICP)")
    ax_top.set_title("California | Width-stratified coverage (refined) + support")
    ax_top.grid(True, axis="y", linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.65)
    ax_top.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.88)

    # Bottom panel: n with log scale + ratio cue
    for i, g in enumerate(groups):
        fam = _family(g, g)
        ns = []
        for b in labels:
            e = by.get((g, b))
            ns.append(0 if e is None else e.n)
        offset = (i - 1) * bar_w
        bars = ax_bottom.bar(
            x + offset,
            [n if n > 0 else np.nan for n in ns],
            width=bar_w,
            color=COLORS.get(fam, COLORS["greedy"]),
            alpha=0.52,
            edgecolor=COLORS["outline"],
            linewidth=0.6,
        )
        for bar, n in zip(bars, ns):
            if n <= 0:
                continue
            if n <= 10 or n >= 500:
                ax_bottom.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    n * 1.07,
                    f"{n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.6),
                    clip_on=False,
                )

    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("Support n (log)")
    ax_bottom.set_xlabel("Interval width bin")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, rotation=20, ha="right")
    ax_bottom.grid(True, axis="y", linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.65)

    # explicit ratio annotation requested by user concern
    ax_bottom.text(0.01, 0.93, "example: n=810 is ~135x n=6", transform=ax_bottom.transAxes, fontsize=9, color="#6B8597")

    _save(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tradeoff-inputs", nargs="+", required=True)
    ap.add_argument("--tradeoff-labels", nargs="+", required=True)
    ap.add_argument("--width-inputs", nargs="+", required=True)
    ap.add_argument("--width-labels", nargs="+", required=True)
    ap.add_argument("--target-picp", type=float, default=0.9)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if len(args.tradeoff_inputs) != len(args.tradeoff_labels):
        raise ValueError("tradeoff labels mismatch")
    if len(args.width_inputs) != len(args.width_labels):
        raise ValueError("width labels mismatch")

    os.makedirs(args.out_dir, exist_ok=True)
    tradeoff = _load_tradeoff(args.tradeoff_inputs, args.tradeoff_labels)
    width = _load_width(args.width_inputs, args.width_labels)

    plot_tradeoff_refined(
        tradeoff,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_refined.png"),
    )
    plot_width_refined(
        width,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "width_stratified_refined_nlog.png"),
    )


if __name__ == "__main__":
    main()
