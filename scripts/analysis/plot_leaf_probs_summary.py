#!/usr/bin/env python3
"""
Plot aggregated leaf_probs shape + sample-count curves (no top-k/top-p).

Example:
  PYTHONPATH=src python3 scripts/analysis/plot_leaf_probs_summary.py \
    --inputs outputs/diamonds/run_20260412_105504_diamonds_ep60_d6_t1p2/leaf_probs_val_greedy_*.npz \
            outputs/diamonds/run_20260412_105504_diamonds_ep60_d6_t1p2/leaf_probs_val_prefix_beam_K16_*.npz \
    --labels greedy beam16 \
    --show-y-hist \
    --out outputs/analysis/leaf_probs_summary_greedy_vs_beam16.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required for plotting.") from exc


def _apply_theme(theme: str) -> None:
    """
    Keep default behavior unless a theme is explicitly requested.
    """
    if theme == "classic":
        return
    if theme != "minimal_blue":
        raise ValueError(f"unknown theme: {theme}")

    # Minimal, low-saturation, monochrome look. We request Times New Roman,
    # but fall back to any installed serif font if it's unavailable.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


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
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all samples in file")
    ap.add_argument("--band-low", type=float, default=0.1, help="lower percentile for band")
    ap.add_argument("--band-high", type=float, default=0.9, help="upper percentile for band")
    ap.add_argument("--show-median", action="store_true", help="plot median line")
    ap.add_argument("--show-y-hist", action="store_true", help="overlay y_true histogram on count panel")
    ap.add_argument("--count-mode", choices=["soft", "peak", "both"], default="soft")
    ap.add_argument(
        "--count-only",
        action="store_true",
        help="plot only the count panel (soft_mass/peak_count/y_true_hist) per input for easier diagnosis",
    )
    ap.add_argument(
        "--sharey-count",
        action="store_true",
        help="share y-axis across count panels (effective when --count-only)",
    )
    ap.add_argument(
        "--x-label-style",
        choices=["simple", "with_meta"],
        default="with_meta",
        help="x-axis label verbosity; use simple for slides",
    )
    ap.add_argument(
        "--theme",
        choices=["classic", "minimal_blue"],
        default="classic",
        help="plot style theme (use minimal_blue for slide-friendly monochrome)",
    )
    ap.add_argument(
        "--no-suptitle",
        action="store_true",
        help="suppress figure-level title (useful for papers/slides that add titles externally)",
    )
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    ap.add_argument("--title", default=None, help="figure title")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    _apply_theme(str(args.theme))

    loaded = []
    for path in args.inputs:
        arrays, meta = _load_npz(path)
        loaded.append((path, arrays, meta))

    n_plots = len(loaded)
    if args.count_only:
        fig, axes = plt.subplots(
            nrows=n_plots,
            ncols=1,
            figsize=(11, max(2.6, 1.6 * n_plots)),
            sharex=True,
            sharey=bool(args.sharey_count),
        )
        if n_plots == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(
            nrows=2 * n_plots,
            ncols=1,
            figsize=(11, max(4.0, 3.0 * n_plots)),
            sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1] * n_plots},
        )
        if n_plots == 1:
            axes = [axes[0], axes[1]]

    for i, (path, arrays, meta) in enumerate(loaded):
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

        label = args.labels[i] if args.labels else _default_label(path, meta)
        # Shared, unambiguous x-axis semantics (centers).
        bin_width = float(np.mean(bin_edges[1:] - bin_edges[:-1]))
        x_label = "y (bin center)"
        if args.x_label_style == "with_meta":
            x_label = f"y (bin center; y_source={args.y_source}; n_bins={leaf_probs.shape[1]}; bin_width={bin_width:.4f})"

        if not args.count_only:
            mean_probs = leaf_probs.mean(axis=0)
            low_q, high_q = np.percentile(leaf_probs, [args.band_low * 100, args.band_high * 100], axis=0)
            median = np.median(leaf_probs, axis=0)

            ax_shape = axes[2 * i]
            ax_shape.fill_between(bin_centers, low_q, high_q, color="#1f77b4", alpha=0.2, label="p10-p90")
            ax_shape.plot(bin_centers, mean_probs, color="#1f77b4", linewidth=1.8, label="mean")
            if args.show_median:
                ax_shape.plot(bin_centers, median, color="#2ca02c", linewidth=1.2, linestyle="--", label="median")
            ax_shape.set_title(f"{label} | mean shape + band")
            ax_shape.set_ylabel("prob")
            ax_shape.grid(alpha=0.2, linestyle="--", linewidth=0.5)
            ax_shape.legend(loc="upper right", fontsize=8, ncol=2)

            ax_count = axes[2 * i + 1]
        else:
            ax_count = axes[i]
            ax_count.set_title(f"{label}", fontsize=12, pad=4)

        # Monochrome palette for minimal theme (keeps classic defaults otherwise).
        c_soft = "#4f6d8a" if args.theme == "minimal_blue" else "#ff7f0e"
        c_peak = "#6e90b8" if args.theme == "minimal_blue" else "#9467bd"
        c_hist = "#d6e3f3" if args.theme == "minimal_blue" else None

        # Avoid repeating legends in count-only mode: label only on the first subplot,
        # then use a single figure-level legend.
        lab_soft = "soft_mass" if (not args.count_only or i == 0) else "_nolegend_"
        lab_peak = "peak_count" if (not args.count_only or i == 0) else "_nolegend_"
        lab_hist = "y_true_hist" if (not args.count_only or i == 0) else "_nolegend_"

        if args.count_mode in ("soft", "both"):
            soft_mass = leaf_probs.sum(axis=0)
            ax_count.plot(bin_centers, soft_mass, color=c_soft, linewidth=1.6, label=lab_soft)
        if args.count_mode in ("peak", "both"):
            peak_idx = np.argmax(leaf_probs, axis=1)
            peak_counts = np.bincount(peak_idx, minlength=leaf_probs.shape[1])
            ax_count.plot(
                bin_centers,
                peak_counts,
                color=c_peak,
                linewidth=1.2,
                linestyle="--" if args.theme == "minimal_blue" else "-",
                label=lab_peak,
            )
        if args.show_y_hist:
            y_hist, _ = np.histogram(y_true, bins=bin_edges)
            ax_count.bar(
                bin_centers,
                y_hist,
                width=bin_edges[1:] - bin_edges[:-1],
                alpha=0.55 if args.theme == "minimal_blue" else 0.25,
                color=c_hist,
                edgecolor="none",
                label=lab_hist,
            )

        if not args.count_only:
            ax_count.set_ylabel("count")
            ax_count.set_xlabel(x_label)
            ax_count.grid(alpha=0.2, linestyle="--", linewidth=0.5)
            ax_count.legend(loc="upper right", fontsize=8, ncol=2)
        else:
            # Global labels for count-only to avoid repeating text.
            ax_count.set_xlabel("")
            ax_count.set_ylabel("")
            ax_count.grid(alpha=0.08 if args.theme == "minimal_blue" else 0.2, linestyle="--", linewidth=0.5)

    if args.title and not args.no_suptitle:
        fig.suptitle(args.title)
    if args.count_only:
        # Global axis labels (Matplotlib version compatible).
        try:
            fig.supxlabel("y (bin center)")
            fig.supylabel("count")
        except Exception:
            fig.text(0.5, 0.02, "y (bin center)", ha="center", va="center")
            fig.text(0.02, 0.5, "count", ha="center", va="center", rotation=90)

        # Single legend for the whole figure.
        handles, labels = axes[0].get_legend_handles_labels()
        # Drop nolegend sentinels if any slipped through.
        keep = [(h, l) for (h, l) in zip(handles, labels) if l and l != "_nolegend_"]
        if keep:
            handles, labels = zip(*keep)
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels),
                bbox_to_anchor=(0.5, 0.98),
                fontsize=9,
            )

    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.94) if args.count_only else None)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
