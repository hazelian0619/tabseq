#!/usr/bin/env python3
"""
Create a "one-page deck" figure for peak-merge alpha sweep.

Inputs:
  - summary CSV (alpha -> PICP/MPIW/diagnostics), produced by sweep_peak_merge_alpha.py
  - width CSV (alpha x width_bin -> coverage + n), produced by sweep_peak_merge_alpha.py

Outputs:
  - a single PNG with 3 panels:
    A) alpha -> PICP & MPIW (tradeoff overview)
    B) heatmap: alpha x width_bin -> coverage (annotated with n)
    C) stacked bars: width-bin support distribution vs alpha (how alpha shifts width)

Why this figure:
  - It answers "alpha changes what?" in one glance:
      overall tradeoff + stratified reliability + where the samples moved.
  - It stays readable under a light/pastel theme.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting.") from exc


@dataclass(frozen=True)
class SummaryRow:
    alpha: float
    picp: float
    mpiw: float
    interval_mass_mean: float | None
    interval_bins_mean: float | None


@dataclass(frozen=True)
class WidthRow:
    alpha: float
    width_lo: float
    width_hi: float
    coverage: float
    n: int


def _read_summary(path: str) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                SummaryRow(
                    alpha=float(r["alpha"]),
                    picp=float(r["PICP"]),
                    mpiw=float(r["MPIW"]),
                    interval_mass_mean=float(r["interval_mass_mean"]) if "interval_mass_mean" in r else None,
                    interval_bins_mean=float(r["interval_bins_mean"]) if "interval_bins_mean" in r else None,
                )
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _read_width(path: str) -> List[WidthRow]:
    rows: List[WidthRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                WidthRow(
                    alpha=float(r["alpha"]),
                    width_lo=float(r["width_lo"]),
                    width_hi=float(r["width_hi"]),
                    coverage=float(r["coverage"]),
                    n=int(float(r["n"])),
                )
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _fmt_alpha(a: float) -> str:
    return f"{a:.2f}".rstrip("0").rstrip(".")


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.1f}, {hi:.1f})"


def _unique_bins(rows: Sequence[WidthRow]) -> List[Tuple[float, float]]:
    return sorted({(r.width_lo, r.width_hi) for r in rows})


def _pastelize_cmap(name: str, pastel: float) -> mcolors.Colormap:
    base = plt.get_cmap(name)
    pastel = float(pastel)
    pastel = 0.0 if pastel < 0.0 else (1.0 if pastel > 1.0 else pastel)
    if pastel <= 0.0:
        return base
    cols = base(np.linspace(0, 1, 256))
    cols[:, :3] = cols[:, :3] * (1.0 - pastel) + 1.0 * pastel
    return mcolors.ListedColormap(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--width-csv", required=True)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--min-n", type=int, default=0, help="mark low-support cells (n < min-n) in the heatmap")
    ap.add_argument("--cmap", type=str, default="PuBuGn", help="heatmap cmap (pastelized)")
    ap.add_argument("--pastel", type=float, default=0.45, help="blend towards white (0..1)")
    ap.add_argument("--title", type=str, default="Peak-merge alpha sweep (beam16): what changes with alpha?")
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    summary = sorted(_read_summary(args.summary_csv), key=lambda r: float(r.alpha), reverse=True)
    width = _read_width(args.width_csv)
    bins = _unique_bins(width)
    alphas = sorted({float(r.alpha) for r in width}, reverse=True)

    # Build matrices aligned to (alpha, bin)
    cov = np.full((len(alphas), len(bins)), np.nan, dtype=float)
    nmat = np.full((len(alphas), len(bins)), np.nan, dtype=float)
    idx_a = {float(a): i for i, a in enumerate(alphas)}
    idx_b = {b: j for j, b in enumerate(bins)}
    for r in width:
        i = idx_a[float(r.alpha)]
        j = idx_b[(r.width_lo, r.width_hi)]
        cov[i, j] = float(r.coverage)
        nmat[i, j] = float(r.n)

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.05, 1.15], hspace=0.22, wspace=0.18)
    ax_trade = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[:, 1])
    ax_dist = fig.add_subplot(gs[1, 0])

    # Panel A: tradeoff line chart (alpha -> PICP and MPIW)
    x = np.arange(len(summary), dtype=float)
    xs = [float(r.alpha) for r in summary]
    picp = np.array([float(r.picp) for r in summary], dtype=float)
    mpiw = np.array([float(r.mpiw) for r in summary], dtype=float)

    # Pastel colors
    c_picp = (0.25, 0.50, 0.72)  # muted blue
    c_mpiw = (0.85, 0.47, 0.20)  # muted orange

    ax_trade.plot(x, picp, marker="o", linewidth=2.0, markersize=5, color=c_picp, label="PICP")
    ax_trade.axhline(float(args.target), color=(0.75, 0.20, 0.20), linestyle="--", linewidth=1.4, alpha=0.85)
    ax_trade.set_ylim(0.0, 1.02)
    ax_trade.set_ylabel("PICP")
    ax_trade.set_xticks(x)
    ax_trade.set_xticklabels([_fmt_alpha(a) for a in xs])
    ax_trade.set_xlabel("alpha")
    ax_trade.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    ax_trade2 = ax_trade.twinx()
    ax_trade2.plot(x, mpiw, marker="s", linewidth=2.0, markersize=4.5, color=c_mpiw, label="MPIW")
    ax_trade2.set_ylabel("MPIW")
    ax_trade2.grid(False)

    # compact legend (merge two axes)
    handles1, labels1 = ax_trade.get_legend_handles_labels()
    handles2, labels2 = ax_trade2.get_legend_handles_labels()
    ax_trade.legend(handles1 + handles2, labels1 + labels2, loc="lower right", frameon=True, framealpha=0.9)
    ax_trade.set_title("A) Overall tradeoff (PICP vs MPIW)")

    # Panel B: heatmap (coverage, annotated with n)
    cmap = _pastelize_cmap(str(args.cmap), float(args.pastel))
    im = ax_heat.imshow(cov, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("Width-stratified coverage (PICP)")

    ax_heat.set_title("B) Width-stratified PICP (annotated with n)")
    ax_heat.set_xlabel("Interval width bin")
    ax_heat.set_ylabel("alpha")
    ax_heat.set_xticks(np.arange(len(bins)))
    ax_heat.set_xticklabels([_bin_label(lo, hi) for lo, hi in bins], rotation=20, ha="right")
    ax_heat.set_yticks(np.arange(len(alphas)))
    ax_heat.set_yticklabels([_fmt_alpha(a) for a in alphas])

    if int(args.min_n) > 0 and np.isfinite(nmat).any():
        from matplotlib.patches import Rectangle

        for i in range(nmat.shape[0]):
            for j in range(nmat.shape[1]):
                nn = nmat[i, j]
                if not np.isfinite(nn):
                    continue
                if int(nn) >= int(args.min_n):
                    continue
                ax_heat.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=(1.0, 1.0, 1.0, 0.30),
                        edgecolor=(0.4, 0.4, 0.4, 0.55),
                        hatch="///",
                        linewidth=0.5,
                    )
                )

    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if not np.isfinite(cov[i, j]):
                continue
            nn = nmat[i, j]
            nn_text = "?" if not np.isfinite(nn) else str(int(nn))
            low_n = int(args.min_n) > 0 and np.isfinite(nn) and int(nn) < int(args.min_n)
            suffix = "*" if low_n else ""
            # Keep annotation compact for readability.
            txt = f"{cov[i, j]:.2f}{suffix}\n{nn_text}"
            txt_color = "black" if float(cov[i, j]) < 0.72 else "white"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=9, color=txt_color)

    # Panel C: stacked distribution of support over width bins per alpha
    ax_dist.set_title("C) Where samples move: support distribution over width bins")
    ax_dist.set_xlabel("alpha")
    ax_dist.set_ylabel("Support fraction")
    ax_dist.set_ylim(0.0, 1.0)
    ax_dist.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")

    # Build counts per alpha/bin and normalize to fractions.
    counts = np.zeros((len(alphas), len(bins)), dtype=float)
    for i, a in enumerate(alphas):
        for j, b in enumerate(bins):
            v = nmat[i, j]
            counts[i, j] = 0.0 if not np.isfinite(v) else float(v)
    totals = np.sum(counts, axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    frac = counts / totals

    # Pastel colors per width bin (stable ordering).
    bin_cmap = _pastelize_cmap("Set3", 0.15)
    bin_cols = bin_cmap(np.linspace(0.0, 1.0, max(len(bins), 2)))[: len(bins)]
    bottom = np.zeros(len(alphas), dtype=float)
    x2 = np.arange(len(alphas), dtype=float)
    for j, (lo, hi) in enumerate(bins):
        ax_dist.bar(
            x2,
            frac[:, j],
            bottom=bottom,
            width=0.8,
            color=bin_cols[j],
            edgecolor=(1, 1, 1, 0.8),
            linewidth=0.6,
            label=_bin_label(lo, hi),
        )
        bottom = bottom + frac[:, j]
    ax_dist.set_xticks(x2)
    ax_dist.set_xticklabels([_fmt_alpha(a) for a in alphas], rotation=0)
    ax_dist.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, title="Width bin")

    fig.suptitle(str(args.title), y=0.98, fontsize=14)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(args.out_png, dpi=180)
    print("saved:", args.out_png)


if __name__ == "__main__":
    main()

