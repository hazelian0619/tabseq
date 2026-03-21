#!/usr/bin/env python3
"""
Plot alpha sweep results as a compact heatmap (pastel-friendly).

Input:
  - width CSV (long form): alpha,width_lo,width_hi,coverage,n
    produced by scripts/analysis/sweep_peak_merge_alpha.py --out-width-csv

Output:
  - heatmap png where color encodes coverage (or delta-to-target) and each cell can annotate "cov\\n(n=...)"

Why this plot:
  - It directly matches the experiment object: alpha x width_bin -> coverage + support.
  - It is more readable than many overlapping line charts when alpha count grows.
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
class Row:
    alpha: float
    width_lo: float
    width_hi: float
    coverage: float
    n: int


def _read_rows(path: str) -> List[Row]:
    out: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(
                Row(
                    alpha=float(r["alpha"]),
                    width_lo=float(r["width_lo"]),
                    width_hi=float(r["width_hi"]),
                    coverage=float(r["coverage"]),
                    n=int(float(r["n"])),
                )
            )
    if not out:
        raise ValueError(f"no rows in {path}")
    return out


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.1f}, {hi:.1f})"


def _fmt_alpha(a: float) -> str:
    return f"{a:.2f}".rstrip("0").rstrip(".")


def _unique_bins(rows: Sequence[Row]) -> List[Tuple[float, float]]:
    bins = {(r.width_lo, r.width_hi) for r in rows}
    return sorted(bins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-csv", required=True)
    ap.add_argument("--alphas", nargs="*", type=float, default=None, help="optional subset")
    ap.add_argument("--value", choices=["coverage", "delta"], default="coverage", help="cell color value")
    ap.add_argument("--target", type=float, default=0.9, help="target coverage (used for value=delta)")
    ap.add_argument("--cmap", type=str, default="PuBuGn", help="matplotlib colormap name")
    ap.add_argument(
        "--pastel",
        type=float,
        default=0.35,
        help="blend colormap towards white (0=no change, 1=all white). recommended 0.3~0.6",
    )
    ap.add_argument(
        "--n-as-alpha",
        action="store_true",
        help="modulate cell opacity by normalized n (helps downweight low-support cells visually)",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=0,
        help="if >0, mark cells with n < min-n as low-support (hatched overlay + optional label cue)",
    )
    ap.add_argument("--title", type=str, default="Peak-merge alpha sweep: width-stratified PICP (with n)")
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    rows = _read_rows(args.width_csv)
    bins = _unique_bins(rows)
    alphas_all = sorted({r.alpha for r in rows}, reverse=True)
    alphas = list(args.alphas) if args.alphas else alphas_all
    alphas = [a for a in alphas if a in set(alphas_all)]
    if not alphas:
        raise ValueError("no alphas selected")

    # Matrices: A x B
    cov = np.full((len(alphas), len(bins)), np.nan, dtype=float)
    nmat = np.full((len(alphas), len(bins)), np.nan, dtype=float)
    idx_a = {float(a): i for i, a in enumerate(alphas)}
    idx_b = {b: j for j, b in enumerate(bins)}

    for r in rows:
        if r.alpha not in idx_a:
            continue
        i = idx_a[float(r.alpha)]
        j = idx_b[(r.width_lo, r.width_hi)]
        cov[i, j] = float(r.coverage)
        nmat[i, j] = float(r.n)

    plt.style.use("seaborn-v0_8-whitegrid")

    # value to display
    if args.value == "coverage":
        val = cov
        vmin, vmax = 0.0, 1.0
        cbar_label = "Coverage (PICP)"
    else:
        val = cov - float(args.target)
        finite = val[np.isfinite(val)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
        max_abs = max(max_abs, 0.05)
        vmin, vmax = -max_abs, max_abs
        cbar_label = f"Δ to target ({args.target:.2f})"

    # pastelize cmap
    base = plt.get_cmap(str(args.cmap))
    pastel = float(args.pastel)
    pastel = 0.0 if pastel < 0.0 else (1.0 if pastel > 1.0 else pastel)
    if pastel > 0:
        cols = base(np.linspace(0, 1, 256))
        cols[:, :3] = cols[:, :3] * (1.0 - pastel) + 1.0 * pastel
        cmap = mcolors.ListedColormap(cols)
    else:
        cmap = base

    fig, ax = plt.subplots(figsize=(2.1 + 2.0 * len(bins), 2.0 + 0.55 * len(alphas)))
    im = ax.imshow(val, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_title(str(args.title))
    ax.set_xlabel("Interval width bin")
    ax.set_ylabel("alpha")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([_bin_label(lo, hi) for lo, hi in bins], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([_fmt_alpha(a) for a in alphas])

    # Optional: n -> alpha (opacity)
    if args.n_as_alpha and np.isfinite(nmat).any():
        nn = nmat.copy()
        nn[~np.isfinite(nn)] = 0.0
        nmax = float(np.max(nn)) if np.max(nn) > 0 else 1.0
        # Draw a white overlay with varying alpha to "fade" low-n cells.
        overlay = np.ones((val.shape[0], val.shape[1], 4), dtype=float)
        overlay[:, :, :3] = 1.0
        # Higher n -> more opaque data (less white overlay).
        # alpha of overlay: low-n -> high overlay alpha.
        overlay[:, :, 3] = 0.65 * (1.0 - (nn / nmax))
        ax.imshow(overlay, aspect="auto", interpolation="nearest")

    # Optional: mark low-support cells with hatch.
    if int(args.min_n) > 0 and np.isfinite(nmat).any():
        from matplotlib.patches import Rectangle

        for i in range(nmat.shape[0]):
            for j in range(nmat.shape[1]):
                nn = nmat[i, j]
                if not np.isfinite(nn):
                    continue
                if int(nn) >= int(args.min_n):
                    continue
                # Rectangle coordinates: imshow uses cell centers at integer ticks.
                # So cell spans [j-0.5, j+0.5] x [i-0.5, i+0.5].
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=(1.0, 1.0, 1.0, 0.28),
                        edgecolor=(0.4, 0.4, 0.4, 0.55),
                        hatch="///",
                        linewidth=0.5,
                    )
                )

    # Annotate cells: coverage + n
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if not np.isfinite(cov[i, j]):
                continue
            nn = nmat[i, j]
            nn_text = "?" if not np.isfinite(nn) else str(int(nn))
            low_n = int(args.min_n) > 0 and np.isfinite(nn) and int(nn) < int(args.min_n)
            if args.value == "coverage":
                suffix = "*" if low_n else ""
                text = f"{cov[i, j]:.2f}{suffix}\n(n={nn_text})"
                txt_color = "black" if cov[i, j] < 0.75 else "white"
            else:
                suffix = "*" if low_n else ""
                text = f"{cov[i, j]:.2f}{suffix}\nΔ{(cov[i, j]-float(args.target)):+.2f}\n(n={nn_text})"
                txt_color = "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=txt_color)

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180)
    print("saved:", args.out_png)


if __name__ == "__main__":
    main()
