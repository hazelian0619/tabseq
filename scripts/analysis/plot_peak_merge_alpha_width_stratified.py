#!/usr/bin/env python3
"""
Plot width_stratified_PICP curves for a peak-merge alpha sweep.

Input is the long-form CSV produced by:
  scripts/analysis/sweep_peak_merge_alpha.py --out-width-csv ...

We plot:
  - coverage (PICP) by width bin for each alpha
  - support n by width bin for each alpha (second panel)

Tip:
  If many alphas are plotted, the "report" style is usually more readable:
    - top: coverage curves (colored by alpha with a colorbar)
    - bottom: n heatmap (alpha x width bin), annotated
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
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for plotting.") from exc


@dataclass(frozen=True)
class Row:
    alpha: float
    width_lo: float
    width_hi: float
    coverage: float
    n: int


def _read_rows(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
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


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.1f}, {hi:.1f})"


def _unique_bins(rows: Sequence[Row]) -> List[Tuple[float, float]]:
    bins = {(r.width_lo, r.width_hi) for r in rows}
    return sorted(bins)


def _unique_alphas(rows: Sequence[Row]) -> List[float]:
    alphas = sorted({float(r.alpha) for r in rows}, reverse=True)
    return alphas


def _fmt_alpha(a: float) -> str:
    # Keep labels compact: 0.33, 0.5, ...
    return f"{a:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-csv", required=True, help="alpha x width_bin coverage CSV")
    ap.add_argument("--alphas", nargs="*", type=float, default=None, help="optional alpha subset to plot")
    ap.add_argument(
        "--style",
        choices=["lines", "report"],
        default="lines",
        help="lines: two line charts (coverage + n); report: coverage lines + n heatmap (recommended)",
    )
    ap.add_argument(
        "--highlight-alpha",
        type=float,
        default=None,
        help="optional alpha to highlight (thicker line / markers), e.g. 0.33",
    )
    ap.add_argument("--target", type=float, default=0.9, help="target coverage reference line")
    ap.add_argument("--title", type=str, default="Peak-merge alpha sweep: width-stratified coverage")
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    rows = _read_rows(args.width_csv)
    bins = _unique_bins(rows)
    all_alphas = _unique_alphas(rows)
    alphas = list(args.alphas) if args.alphas else all_alphas

    # Build lookup: alpha -> (bin -> (cov, n))
    by_alpha: Dict[float, Dict[Tuple[float, float], Tuple[float, int]]] = {}
    for r in rows:
        if r.alpha not in alphas:
            continue
        by_alpha.setdefault(float(r.alpha), {})[(r.width_lo, r.width_hi)] = (float(r.coverage), int(r.n))

    # Plot
    xlabels = [_bin_label(lo, hi) for lo, hi in bins]
    x = np.arange(len(bins), dtype=float)

    if args.style == "lines":
        fig, (ax_cov, ax_n) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
        )

        # Use a stable color cycle; plot higher alpha first (narrower) so it doesn't hide others as much.
        for a in sorted(by_alpha.keys(), reverse=True):
            cov = []
            ns = []
            for b in bins:
                v = by_alpha[a].get(b)
                cov.append(np.nan if v is None else float(v[0]))
                ns.append(0 if v is None else int(v[1]))
            ax_cov.plot(x, cov, marker="o", linewidth=1.6, markersize=4, label=f"alpha={_fmt_alpha(a)}")
            ax_n.plot(x, ns, marker="o", linewidth=1.2, markersize=3, label=f"alpha={_fmt_alpha(a)}")

        ax_cov.axhline(
            float(args.target),
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label=f"target={args.target:.2f}",
        )
        ax_cov.set_ylim(0.0, 1.02)
        ax_cov.set_ylabel("Coverage (PICP)")
        ax_cov.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
        ax_cov.legend(loc="lower right", fontsize=9, ncol=2)
        ax_cov.set_title(str(args.title))

        ax_n.set_ylabel("Support n")
        ax_n.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
        ax_n.set_xticks(x)
        ax_n.set_xticklabels(xlabels, rotation=20, ha="right")
        ax_n.set_xlabel("Interval width bin")
    else:
        # Report style: coverage curves (colored by alpha) + n heatmap (alpha x width bin)
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.2], hspace=0.15)
        ax_cov = fig.add_subplot(gs[0, 0])
        ax_n = fig.add_subplot(gs[1, 0], sharex=ax_cov)

        alphas_sorted = sorted(by_alpha.keys(), reverse=True)
        a_min, a_max = float(min(alphas_sorted)), float(max(alphas_sorted))
        cmap = plt.get_cmap("viridis")

        def color_for(a: float):
            if a_max == a_min:
                return cmap(0.5)
            t = (float(a) - a_min) / (a_max - a_min)
            return cmap(t)

        for a in alphas_sorted:
            cov = []
            for b in bins:
                v = by_alpha[a].get(b)
                cov.append(np.nan if v is None else float(v[0]))
            lw = 2.6 if (args.highlight_alpha is not None and abs(float(a) - float(args.highlight_alpha)) < 1e-12) else 1.6
            ms = 5 if lw > 2.0 else 4
            ax_cov.plot(
                x,
                cov,
                marker="o",
                linewidth=lw,
                markersize=ms,
                color=color_for(a),
                alpha=0.95,
            )
            # Inline label on the right for readability (avoid huge legend).
            if len(cov) and not np.isnan(cov[-1]):
                ax_cov.text(
                    x[-1] + 0.02,
                    float(cov[-1]),
                    _fmt_alpha(a),
                    fontsize=8,
                    va="center",
                    color=color_for(a),
                )

        ax_cov.axhline(float(args.target), color="#d62728", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_cov.set_ylim(0.0, 1.02)
        ax_cov.set_ylabel("Coverage (PICP)")
        ax_cov.grid(alpha=0.25, linestyle="--", linewidth=0.6, axis="y")
        ax_cov.set_title(str(args.title))

        # Colorbar for alpha
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=a_min, vmax=a_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_cov, fraction=0.03, pad=0.02)
        cbar.set_label("alpha", rotation=90)

        # Build n matrix for heatmap
        n_mat = np.full((len(alphas_sorted), len(bins)), np.nan, dtype=float)
        for ai, a in enumerate(alphas_sorted):
            for bi, b in enumerate(bins):
                v = by_alpha[a].get(b)
                if v is not None:
                    n_mat[ai, bi] = float(v[1])

        im = ax_n.imshow(n_mat, aspect="auto", interpolation="nearest", cmap="Blues")
        cbar_n = fig.colorbar(im, ax=ax_n, fraction=0.03, pad=0.02)
        cbar_n.set_label("Support n", rotation=90)

        ax_n.set_yticks(np.arange(len(alphas_sorted)))
        ax_n.set_yticklabels([_fmt_alpha(a) for a in alphas_sorted])
        ax_n.set_ylabel("alpha")

        ax_n.set_xticks(x)
        ax_n.set_xticklabels(xlabels, rotation=20, ha="right")
        ax_n.set_xlabel("Interval width bin")

        # Annotate n values if the grid is not too large.
        if n_mat.size <= 80:
            vmax = np.nanmax(n_mat) if np.isfinite(n_mat).any() else 1.0
            for ai in range(n_mat.shape[0]):
                for bi in range(n_mat.shape[1]):
                    val = n_mat[ai, bi]
                    if not np.isfinite(val):
                        continue
                    txt_color = "white" if float(val) > 0.55 * float(vmax) else "black"
                    ax_n.text(bi, ai, str(int(val)), ha="center", va="center", fontsize=7, color=txt_color)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=170)
    print("saved:", args.out_png)


if __name__ == "__main__":
    main()
