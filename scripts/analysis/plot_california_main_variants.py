#!/usr/bin/env python3
"""
Generate polished California report figure variants.

Outputs:
- Tradeoff variants (PICP vs MPIW)
- Width-stratified variants with explicit n-encoding
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
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
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


# High-lightness palette (mostly light blue), optimized for clean readability.
COLORS = {
    # background + framing
    "bg": "#FBFDFF",
    "grid": "#DDEAF4",
    "text": "#23313C",
    "spine": "#C7D6E3",
    # families (single hue)
    "greedy": "#BFE6F7",
    "beam": "#9AD5F2",
    "full": "#74C3EA",
    "teacher": "#D6F0FB",
    # accents
    "target": "#3C7EA6",
    "outline": "#2E6F95",
}


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
    text = str(raw)
    m = VALUE_RE.search(text)
    if not m:
        raise ValueError(f"cannot parse width value: {raw}")
    cov = float(m.group(1))
    n = int(m.group(2)) if m.group(2) else 0
    return cov, n


def _load_tradeoff(metrics_paths: Sequence[str], labels: Sequence[str]) -> List[TradeoffPoint]:
    points: List[TradeoffPoint] = []
    for path, label in zip(metrics_paths, labels):
        d = _load_json(path)
        if "PICP" not in d or "MPIW" not in d:
            continue
        points.append(
            TradeoffPoint(label=label, mode=str(d.get("mode", "unknown")), picp=float(d["PICP"]), mpiw=float(d["MPIW"]))
        )
    return points


def _load_width(metrics_paths: Sequence[str], labels: Sequence[str]) -> List[WidthEntry]:
    entries: List[WidthEntry] = []
    for path, label in zip(metrics_paths, labels):
        d = _load_json(path)
        ws = d.get("width_stratified_PICP") or {}
        for k, v in ws.items():
            lo, hi = _parse_width_key(k)
            cov, n = _parse_width_value(v)
            entries.append(
                WidthEntry(label=label, width_bin=f"[{lo:.1f}, {hi:.1f})", lo=lo, hi=hi, coverage=cov, n=n)
            )
    return entries


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["spine"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "font.size": 11,
            "axes.titleweight": "semibold",
        }
    )


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # `bbox_inches="tight"` avoids clipping rotated ticklabels / annotations (and insets).
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor=COLORS["bg"])
    plt.close(fig)
    print("saved:", path)


def _prep_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])


def _format_width_bin_label(width_bin: str) -> str:
    # width_bin looks like "[0.0, 0.4)"
    s = width_bin.strip().lstrip("[").rstrip(")")
    lo, hi = [x.strip() for x in s.split(",")]
    return f"{lo}-{hi}"


def _mask_value(label: str) -> Optional[float]:
    m = re.search(r"greedy_m([0-9]*\\.?[0-9]+)", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _short_tradeoff_label(label: str) -> str:
    # Keep labels compact and consistent.
    mv = _mask_value(label)
    if label.startswith("greedy_m") and mv is not None:
        return f"g {mv:g}"
    if label.startswith("beam"):
        k = re.sub(r"[^0-9]", "", label)
        return f"beam {k}" if k else "beam"
    if label == "teacher_tf":
        return "teacher"
    if label == "full":
        return "full"
    return label.replace("_", " ")


def _scatter_tradeoff(ax: plt.Axes, p: TradeoffPoint, *, emphasize: bool = False) -> None:
    fam = _family(p.label, p.mode)
    size = {"greedy": 44, "beam": 78, "full": 88, "teacher": 86}.get(fam, 70)
    alpha = 0.65 if fam == "greedy" else 0.92
    lw = 1.4 if emphasize else 0.9
    ax.scatter(
        p.mpiw,
        p.picp,
        s=size * (1.25 if emphasize else 1.0),
        marker="o",
        color=COLORS.get(fam, COLORS["greedy"]),
        edgecolor=COLORS["outline"],
        linewidth=lw,
        alpha=alpha,
        zorder=4 if emphasize else 3,
    )


def _largest_gap_split(xs: Sequence[float]) -> Optional[float]:
    xs = sorted(xs)
    if len(xs) < 4:
        return None
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    g, i = max(gaps, key=lambda t: t[0])
    if g <= 0:
        return None
    return (xs[i] + xs[i + 1]) / 2.0


def _pareto_front(points: Sequence[TradeoffPoint]) -> List[TradeoffPoint]:
    """Non-dominated set for (maximize PICP, minimize MPIW)."""
    out: List[TradeoffPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q.picp >= p.picp and q.mpiw <= p.mpiw) and (q.picp > p.picp or q.mpiw < p.mpiw):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda t: t.mpiw)


def plot_tradeoff_variant_scatter(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _prep_axes(ax)

    # Minimal labels: annotate only the "best at target" + full + teacher (if present).
    candidates = [p for p in points if p.picp >= target - 1e-9]
    best = min(candidates, key=lambda p: p.mpiw) if candidates else None

    for p in points:
        _scatter_tradeoff(ax, p, emphasize=(best is not None and p.label == best.label))

    def callout(p: TradeoffPoint, text: str) -> None:
        ax.annotate(
            text,
            xy=(p.mpiw, p.picp),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=COLORS["spine"], lw=1.0),
            clip_on=False,
        )

    if best is not None:
        # Avoid redundant callouts when the best point is itself a named baseline.
        if best.label == "teacher_tf":
            callout(best, "teacher (best@target)")
        elif best.label == "full":
            callout(best, "full (best@target)")
        else:
            callout(best, "best@target")
    for key, label in [("full", "full"), ("teacher_tf", "teacher")]:
        for p in points:
            if p.label == key:
                if best is not None and p.label == best.label:
                    break
                callout(p, label)
                break

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax.text(
        0.99,
        target,
        f"target={target:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=10,
        color=COLORS["target"],
    )

    xs = [p.mpiw for p in points]
    xmin, xmax = min(xs), max(xs)
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | PICP-MPIW tradeoff")

    legend = [
        Line2D([0], [0], marker="o", color="w", label="greedy", markerfacecolor=COLORS["greedy"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="beam", markerfacecolor=COLORS["beam"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="full", markerfacecolor=COLORS["full"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="teacher", markerfacecolor=COLORS["teacher"], markeredgecolor=COLORS["outline"], markersize=8),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.9)
    _save(fig, out_path)


def plot_tradeoff_variant_path(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _prep_axes(ax)

    by_label = {p.label: p for p in points}
    greedy_order = [k for k in ["greedy_m0.1", "greedy_m0.3", "greedy_m0.5", "greedy_m1"] if k in by_label]
    beam_order = [k for k in ["beam16", "beam32", "full"] if k in by_label]

    if greedy_order:
        gx = [by_label[k].mpiw for k in greedy_order]
        gy = [by_label[k].picp for k in greedy_order]
        ax.plot(gx, gy, color=COLORS["greedy"], linewidth=2.2, alpha=0.9, label="greedy path")
        ax.scatter(gx, gy, s=44, marker="o", color=COLORS["greedy"], edgecolor=COLORS["outline"], linewidth=0.9, zorder=3, alpha=0.75)
        # Only label endpoints for clarity.
        for k, name in [(greedy_order[0], "m=0.1"), (greedy_order[-1], "m=1.0")]:
            p = by_label[k]
            ax.annotate(
                name,
                xy=(p.mpiw, p.picp),
                xytext=(10, 8),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=COLORS["spine"], lw=1.0),
                clip_on=False,
            )

    if beam_order:
        bx = [by_label[k].mpiw for k in beam_order]
        by = [by_label[k].picp for k in beam_order]
        ax.plot(bx, by, color=COLORS["beam"], linewidth=2.2, alpha=0.95, label="beam->full path")
        for k in beam_order:
            p = by_label[k]
            fam = "full" if k == "full" else "beam"
            ax.scatter(
                p.mpiw,
                p.picp,
                s=88 if k == "full" else 76,
                marker="o",
                color=COLORS[fam],
                edgecolor=COLORS["outline"],
                linewidth=0.9,
                zorder=3,
            )
        # label only final full point
        if "full" in by_label:
            p = by_label["full"]
            ax.annotate(
                "full",
                xy=(p.mpiw, p.picp),
                xytext=(10, -12),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=COLORS["spine"], lw=1.0),
                clip_on=False,
            )

    if "teacher_tf" in by_label:
        p = by_label["teacher_tf"]
        ax.scatter(
            p.mpiw,
            p.picp,
            s=86,
            marker="o",
            color=COLORS["teacher"],
            edgecolor=COLORS["outline"],
            linewidth=0.9,
            label="teacher",
            zorder=3,
        )

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1.02)
    xs = [p.mpiw for p in points]
    ax.set_xlim(max(0, min(xs) - (max(xs) - min(xs)) * 0.08), max(xs) + (max(xs) - min(xs)) * 0.08)
    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | Tradeoff (paths)")
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.9)
    _save(fig, out_path)


def plot_tradeoff_variant_quadrant(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _prep_axes(ax)
    xs = [p.mpiw for p in points]
    x_ref = float(np.median(xs))

    # soft "target satisfied" shading (keep single hue)
    ax.add_patch(Rectangle((0, target), max(xs) * 1.2, 1.05 - target, facecolor="#EAF6FD", alpha=0.65, zorder=0, linewidth=0))
    for p in points:
        fam = _family(p.label, p.mode)
        ax.scatter(
            p.mpiw,
            p.picp,
            s=46 if fam == "greedy" else 86,
            marker="o",
            color=COLORS[fam],
            edgecolor=COLORS["outline"],
            linewidth=0.9,
            alpha=0.7 if fam == "greedy" else 0.95,
            zorder=3,
        )

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax.axvline(x_ref, color=COLORS["spine"], linestyle=":", linewidth=1.2)
    ax.text(0.02, 0.96, "shaded: PICP>=target", transform=ax.transAxes, fontsize=10, color=COLORS["target"], va="top")

    ax.set_ylim(0, 1.02)
    ax.set_xlim(max(0, min(xs) - (max(xs) - min(xs)) * 0.08), max(xs) + (max(xs) - min(xs)) * 0.08)
    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | Tradeoff (target zone)")
    _save(fig, out_path)


def plot_tradeoff_variant_inset(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    _set_style()
    # NOTE: Using an actual inset axis + tight_layout can lead to clipping artifacts on save.
    # This variant uses an explicit right-hand zoom panel (global + local view) to avoid those issues.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [2.15, 1.0]})

    for a in (ax, axz):
        _prep_axes(a)

    greedy_points = [p for p in points if _family(p.label, p.mode) == "greedy"]

    # Global panel.
    for p in points:
        _scatter_tradeoff(ax, p)

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1.02)
    xs = [p.mpiw for p in points]
    ax.set_xlim(max(0, min(xs) - (max(xs) - min(xs)) * 0.08), max(xs) + (max(xs) - min(xs)) * 0.08)
    ax.set_xlabel("MPIW (interval width)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | Tradeoff (global)")

    # Zoom panel bounds derived from greedy cluster (robust to metric changes).
    if greedy_points:
        zx = [p.mpiw for p in greedy_points]
        zy = [p.picp for p in greedy_points]
        x0, x1 = min(zx), max(zx)
        y0, y1 = min(zy), max(zy)
        pad_x = max(0.02, (x1 - x0) * 0.35)
        pad_y = max(0.02, (y1 - y0) * 0.35)
        xlo, xhi = x0 - pad_x, x1 + pad_x
        ylo, yhi = y0 - pad_y, y1 + pad_y

        # Draw zoom rectangle on global panel.
        ax.add_patch(
            Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, edgecolor=COLORS["spine"], linewidth=1.3, linestyle=":")
        )
        ax.text(xlo, yhi, "greedy zoom", fontsize=10, color=COLORS["text"], va="bottom", ha="left")
    else:
        # fallback: show everything
        xlo, xhi, ylo, yhi = ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]

    # Zoom panel: greedy cluster + reference non-greedy nearest point (optional).
    for p in greedy_points:
        mv = _mask_value(p.label)
        axz.scatter(p.mpiw, p.picp, s=140, marker="o", color=COLORS["greedy"], edgecolor=COLORS["outline"], linewidth=1.0, zorder=4, alpha=0.85)
        short = f"m={mv:g}" if mv is not None else p.label
        axz.annotate(
            short,
            (p.mpiw, p.picp),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.75),
            clip_on=False,
        )

    axz.set_xlim(xlo, xhi)
    axz.set_ylim(ylo, yhi)
    axz.set_xlabel("MPIW")
    axz.set_ylabel("PICP")
    axz.set_title("Greedy (zoom)")

    # Legend on global panel.
    legend = [
        Line2D([0], [0], marker="o", color="w", label="greedy", markerfacecolor=COLORS["greedy"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="beam", markerfacecolor=COLORS["beam"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="full", markerfacecolor=COLORS["full"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="teacher", markerfacecolor=COLORS["teacher"], markeredgecolor=COLORS["outline"], markersize=8),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.9)
    _save(fig, out_path)


def plot_tradeoff_variant_twopanel_zoom(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    """Two-panel zoom: small-MPIW cluster (greedy/teacher) + large-MPIW cluster (beam/full)."""
    _set_style()
    xs = [p.mpiw for p in points]
    split = _largest_gap_split(xs) or float(np.median(xs))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1.0]})
    _prep_axes(ax_l)
    _prep_axes(ax_r)

    left = [p for p in points if p.mpiw <= split]
    right = [p for p in points if p.mpiw >= split]
    if not left or not right:
        # Fallback: single-panel labelled scatter.
        plot_tradeoff_variant_logx(points, target=target, out_path=out_path)
        return

    def draw(ax: plt.Axes, pts: List[TradeoffPoint], title: str) -> None:
        for p in pts:
            _scatter_tradeoff(ax, p)
            ax.annotate(
                _short_tradeoff_label(p.label),
                xy=(p.mpiw, p.picp),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.85),
                clip_on=False,
            )
        ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4)
        ax.set_title(title)

    draw(ax_l, left, "Zoom: small MPIW")
    draw(ax_r, right, "Zoom: large MPIW")

    # Bounds per panel
    for ax, pts in [(ax_l, left), (ax_r, right)]:
        px = [p.mpiw for p in pts]
        pad = max(1e-6, (max(px) - min(px)) * 0.12)
        ax.set_xlim(max(0, min(px) - pad), max(px) + pad)
        ax.set_ylim(0.0, 1.02)

    ax_l.set_ylabel("PICP (coverage)")
    ax_l.set_xlabel("MPIW")
    ax_r.set_xlabel("MPIW")
    fig.suptitle("California | PICP-MPIW tradeoff (two-panel zoom)", y=1.02)
    _save(fig, out_path)


def plot_tradeoff_variant_logx(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    """Log-x tradeoff: spreads the low-MPIW greedy cluster without extra panels."""
    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _prep_axes(ax)

    for p in points:
        _scatter_tradeoff(ax, p)
        ax.annotate(
            _short_tradeoff_label(p.label),
            xy=(p.mpiw, p.picp),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.85),
            clip_on=False,
        )

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("MPIW (log scale)")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("California | PICP-MPIW tradeoff (log-x)")
    _save(fig, out_path)


def plot_tradeoff_variant_scatter_table(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    """Scatter + right-side table: no labels on points, but fully readable mapping."""
    _set_style()
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]})
    _prep_axes(ax)
    ax_t.set_facecolor(COLORS["bg"])
    ax_t.axis("off")

    for p in points:
        _scatter_tradeoff(ax, p)

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4)
    xs = [p.mpiw for p in points]
    xmin, xmax = min(xs), max(xs)
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("MPIW")
    ax.set_ylabel("PICP")
    ax.set_title("Tradeoff")

    # Table-like text (compact, clear).
    rows = sorted(points, key=lambda p: p.mpiw)
    header = "method        PICP    MPIW"
    ax_t.text(0.02, 0.95, "California | Values", fontsize=12, fontweight="semibold", color=COLORS["text"], va="top")
    ax_t.text(0.02, 0.88, header, fontsize=11, family="monospace", color=COLORS["text"], va="top")
    y = 0.84
    dy = 0.065
    for p in rows:
        fam = _family(p.label, p.mode)
        name = _short_tradeoff_label(p.label)
        line = f"{name:<12}  {p.picp:0.3f}  {p.mpiw:0.3f}"
        ax_t.text(0.02, y, line, fontsize=11, family="monospace", color=COLORS["text"], va="top")
        # color chip
        ax_t.add_patch(Rectangle((0.84, y - 0.028), 0.10, 0.028, transform=ax_t.transAxes, facecolor=COLORS.get(fam, COLORS["greedy"]), edgecolor=COLORS["outline"], lw=0.6))
        y -= dy
        if y < 0.05:
            break

    _save(fig, out_path)


def plot_tradeoff_variant_pareto(points: Sequence[TradeoffPoint], target: float, out_path: str) -> None:
    """Standard paper-style: scatter + Pareto frontier + minimal labels."""
    _set_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    _prep_axes(ax)

    for p in points:
        _scatter_tradeoff(ax, p)

    front = _pareto_front(points)
    if len(front) >= 2:
        ax.plot(
            [p.mpiw for p in front],
            [p.picp for p in front],
            color=COLORS["outline"],
            linewidth=2.0,
            alpha=0.75,
            zorder=2,
        )
        # Label only the frontier points.
        for p in front:
            ax.annotate(
                _short_tradeoff_label(p.label),
                xy=(p.mpiw, p.picp),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.85),
                clip_on=False,
            )

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4)
    xs = [p.mpiw for p in points]
    xmin, xmax = min(xs), max(xs)
    pad = (xmax - xmin) * 0.08
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("MPIW")
    ax.set_ylabel("PICP")
    ax.set_title("California | Tradeoff (Pareto frontier)")
    _save(fig, out_path)


def _pick_width_groups(entries: Sequence[WidthEntry]) -> List[str]:
    """Keep the width plot readable by selecting a representative subset of methods."""
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
    # If we only found 1-2 groups, just fall back to first few labels.
    if len(out) < min(3, len(labels)):
        out = labels[: min(4, len(labels))]
    return out


def plot_width_variant_dualpanel(entries: Sequence[WidthEntry], target: float, out_path: str) -> None:
    _set_style()
    # Lighter alternative to bar charts: dots (size encodes support n).
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.2]},
    )
    _prep_axes(ax_top)
    _prep_axes(ax_bottom)

    bins = sorted({(e.lo, e.hi, e.width_bin) for e in entries}, key=lambda t: (t[0], t[1]))
    labels = [b[2] for b in bins]
    xlabels = [_format_width_bin_label(b) for b in labels]
    x = np.arange(len(labels), dtype=float)
    groups = _pick_width_groups(entries)
    by = {(e.label, e.width_bin): e for e in entries}

    ns_all = [e.n for e in entries if e.n > 0]
    n_min = min(ns_all) if ns_all else 1
    n_max = max(ns_all) if ns_all else 1

    def size_from_n(n: int) -> float:
        if n <= 0:
            return 0.0
        s_min, s_max = 35.0, 420.0
        t = (math.sqrt(n) - math.sqrt(n_min)) / (math.sqrt(n_max) - math.sqrt(n_min) + 1e-9)
        return s_min + t * (s_max - s_min)

    fam_order = ["greedy", "beam", "full", "teacher"]
    fam_to_jitter = {fam: (i - (len(fam_order) - 1) / 2.0) * 0.18 for i, fam in enumerate(fam_order)}

    # Top: coverage (PICP)
    for g in groups:
        fam = _family(g, g)
        xs, ys, ss = [], [], []
        for i, b in enumerate(labels):
            e = by.get((g, b))
            if e is None or e.n <= 0:
                continue
            xs.append(x[i] + fam_to_jitter.get(fam, 0.0))
            ys.append(e.coverage)
            ss.append(size_from_n(e.n))
        ax_top.scatter(
            xs,
            ys,
            s=ss,
            marker="o",
            color=COLORS.get(fam, COLORS["greedy"]),
            edgecolor=COLORS["outline"],
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )

    ax_top.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4)
    ax_top.set_ylim(0.0, 1.02)
    ax_top.set_ylabel("Coverage (PICP)")
    ax_top.set_title("California | Width-stratified coverage (dot=size n)")

    fam_handles = [
        Line2D([0], [0], marker="o", color="w", label=f, markerfacecolor=COLORS[f], markeredgecolor=COLORS["outline"], markersize=8)
        for f in fam_order
    ]
    lg1 = ax_top.legend(handles=fam_handles, loc="lower left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, ncol=2)
    ax_top.add_artist(lg1)

    size_refs = [max(1, n_min), int((n_min + n_max) / 2), n_max]
    size_handles = [plt.scatter([], [], s=size_from_n(v), color="#E6F3FB", edgecolor=COLORS["outline"], label=f"n={v}") for v in size_refs]
    ax_top.legend(handles=size_handles, loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.9, title="support")

    # Bottom: support n (log)
    for g in groups:
        fam = _family(g, g)
        xs, ys, ss = [], [], []
        for i, b in enumerate(labels):
            e = by.get((g, b))
            if e is None or e.n <= 0:
                continue
            xs.append(x[i] + fam_to_jitter.get(fam, 0.0))
            ys.append(e.n)
            ss.append(size_from_n(e.n))
        ax_bottom.scatter(
            xs,
            ys,
            s=ss,
            marker="o",
            color=COLORS.get(fam, COLORS["greedy"]),
            edgecolor=COLORS["outline"],
            linewidth=0.8,
            alpha=0.55,
            zorder=3,
        )

    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("n (log)")
    ax_bottom.set_xlabel("Interval width bin")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(xlabels, rotation=0, ha="center")

    _save(fig, out_path)


def plot_width_variant_bubble(entries: Sequence[WidthEntry], target: float, out_path: str) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    _prep_axes(ax)

    bins = sorted({(e.lo, e.hi, e.width_bin) for e in entries}, key=lambda t: (t[0], t[1]))
    labels = [b[2] for b in bins]
    xlabels = [_format_width_bin_label(b) for b in labels]
    x_map = {b: i for i, b in enumerate(labels)}

    # Use the same subset selection as the dual-panel plot for consistency/readability.
    keep = set(_pick_width_groups(entries))
    entries = [e for e in entries if e.label in keep]

    ns = [e.n for e in entries if e.n > 0]
    n_min = min(ns) if ns else 1
    n_max = max(ns) if ns else 1

    def size_from_n(n: int) -> float:
        if n <= 0:
            return 0.0
        # sqrt mapping preserves visibility for small n but highlights large support
        s_min, s_max = 45.0, 820.0
        t = (math.sqrt(n) - math.sqrt(n_min)) / (math.sqrt(n_max) - math.sqrt(n_min) + 1e-9)
        return s_min + t * (s_max - s_min)

    # jitter per mode
    jitter = {"greedy": -0.18, "beam": 0.0, "full": 0.18, "teacher": 0.36}

    for e in entries:
        fam = _family(e.label, e.label)
        x = x_map[e.width_bin] + jitter.get(fam, 0.0)
        s = size_from_n(e.n)
        ax.scatter(
            x,
            e.coverage,
            s=s,
            marker="o",
            color=COLORS.get(fam, COLORS["greedy"]),
            edgecolor=COLORS["outline"],
            linewidth=0.9,
            alpha=0.9,
        )

    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Coverage (PICP)")
    ax.set_xlabel("Interval width bin")
    ax.set_xticks(np.arange(len(labels), dtype=float))
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_title("California | Width-stratified coverage (dot=size n)")

    # Legend reflects families, not exact labels (we may select representative variants).
    mode_legend = [
        Line2D([0], [0], marker="o", color="w", label="greedy", markerfacecolor=COLORS["greedy"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="beam", markerfacecolor=COLORS["beam"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="full", markerfacecolor=COLORS["full"], markeredgecolor=COLORS["outline"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="teacher", markerfacecolor=COLORS["teacher"], markeredgecolor=COLORS["outline"], markersize=8),
    ]
    lg1 = ax.legend(handles=mode_legend, loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.85)
    ax.add_artist(lg1)

    size_refs = [n_min, int((n_min + n_max) / 2), n_max]
    size_handles = [
        plt.scatter([], [], s=size_from_n(v), color="#E6F3FB", edgecolor=COLORS["outline"], label=f"n={v}") for v in size_refs
    ]
    ax.legend(handles=size_handles, loc="lower right", frameon=True, facecolor="#FFFFFF", framealpha=0.85, title="support")

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
        raise ValueError("tradeoff-labels length must match tradeoff-inputs")
    if len(args.width_inputs) != len(args.width_labels):
        raise ValueError("width-labels length must match width-inputs")

    os.makedirs(args.out_dir, exist_ok=True)

    trade_points = _load_tradeoff(args.tradeoff_inputs, args.tradeoff_labels)
    width_entries = _load_width(args.width_inputs, args.width_labels)

    # tradeoff variants
    plot_tradeoff_variant_scatter(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v1_scatter_clean.png"),
    )
    plot_tradeoff_variant_path(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v2_trajectory.png"),
    )
    plot_tradeoff_variant_quadrant(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v3_decision_zone.png"),
    )
    plot_tradeoff_variant_inset(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v4_inset_zoom.png"),
    )
    plot_tradeoff_variant_twopanel_zoom(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v5_twopanel_zoom.png"),
    )
    plot_tradeoff_variant_logx(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v6_logx_labels.png"),
    )
    plot_tradeoff_variant_scatter_table(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v7_scatter_table.png"),
    )
    plot_tradeoff_variant_pareto(
        trade_points,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "interval_tradeoff_v8_pareto_frontier.png"),
    )

    # width variants with explicit n
    plot_width_variant_dualpanel(
        width_entries,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "width_stratified_v2_dualpanel_nlog.png"),
    )
    plot_width_variant_bubble(
        width_entries,
        target=args.target_picp,
        out_path=os.path.join(args.out_dir, "width_stratified_v3_bubble_nsize.png"),
    )


if __name__ == "__main__":
    main()
