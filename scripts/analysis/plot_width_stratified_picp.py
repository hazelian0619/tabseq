#!/usr/bin/env python3
"""
Plot width-stratified PICP bars from metrics JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from math import sqrt
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"expected #RRGGBB, got: {hex_color}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return r, g, b


def _blend_hex(a: str, b: str, t: float) -> Tuple[float, float, float]:
    """Blend two hex colors, returning rgb in [0,1]. t=0->a, t=1->b."""
    t = float(max(0.0, min(1.0, t)))
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    return (ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def _default_label(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _width_key_sort_key(key: str) -> Tuple[float, float]:
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", key)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return (0.0, 0.0)


def _parse_value(raw: str) -> Tuple[float, int]:
    val_match = re.search(r"[-+]?[0-9]*\.?[0-9]+", raw)
    n_match = re.search(r"n=([0-9]+)", raw)
    val = float(val_match.group(0)) if val_match else 0.0
    n = int(n_match.group(1)) if n_match else 0
    return val, n


def _compact_width_label(key: str) -> str:
    """
    Make width-bin labels less verbose.
    Example: "Width [0.4, 0.8)" -> "0.4–0.8"
    """
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", key)
    if len(nums) >= 2:
        # ASCII-only to avoid font/encoding issues in report toolchains.
        return f"{nums[0]}-{nums[1]}"
    if len(nums) == 1:
        return str(nums[0])
    return str(key)


def _norm_ppf(p: float) -> float:
    """
    Approximate inverse CDF (quantile) for standard normal.

    Acklam's approximation; accurate enough for CIs without scipy.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients in rational approximations.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = sqrt(-2.0 * np.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = sqrt(-2.0 * np.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -(num / den)

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def _wilson_ci(p_hat: float, n: int, *, z: float) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p_hat = float(p_hat)
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half = (z * sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def _family_from_label(label: str) -> str:
    t = str(label).lower()
    if "greedy" in t:
        return "greedy"
    if "beam" in t:
        return "beam"
    if "full" in t:
        return "full"
    if "teacher" in t or "tf" in t:
        return "teacher"
    return "other"


def _load_widths(path: str) -> Tuple[List[str], List[float], List[int]]:
    data = json.load(open(path))
    widths = data.get("width_stratified_PICP") or {}
    keys = sorted(widths.keys(), key=_width_key_sort_key)
    vals: List[float] = []
    ns: List[int] = []
    for k in keys:
        val, n = _parse_value(str(widths[k]))
        vals.append(val)
        ns.append(n)
    return keys, vals, ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="metrics_val*.json files")
    ap.add_argument("--labels", nargs="*", default=None, help="optional labels for each input")
    ap.add_argument("--title", default=None, help="figure title")
    ap.add_argument("--target", type=float, default=0.9, help="target PICP line")
    ap.add_argument(
        "--style",
        choices=[
            "bars",
            "bars_rug",
            "bars_facets_clean",
            "bubble",
            "line",
            "line_annot_light",
            "line_annot_full",
            "lollipop",
            "dual",
            "heatmap",
            "reliability",
        ],
        default="bars",
        help="plot style",
    )
    ap.add_argument("--color", default="#a6cbe3", help="main series color")
    ap.add_argument("--accent", default="#5fa8d3", help="accent color")
    ap.add_argument("--target-color", default="#e07a5f", help="target line color")
    ap.add_argument(
        "--n-mode",
        choices=["label", "size", "bar"],
        default="label",
        help="how to encode n (label/size/bar)",
    )
    ap.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="confidence level for error bars in reliability style (e.g. 0.95)",
    )
    ap.add_argument(
        "--n-scale",
        choices=["linear", "log"],
        default="linear",
        help="y-scale for the n panel in reliability style",
    )
    ap.add_argument(
        "--n-warn",
        type=int,
        default=30,
        help="highlight bins with n < n_warn in reliability style",
    )
    ap.add_argument(
        "--reliability-layout",
        choices=["auto", "stacked", "grouped"],
        default="auto",
        help="layout for reliability style: stacked(2 rows per series) or grouped(2xK panels)",
    )
    ap.add_argument("--out", required=True, help="output image path (png/pdf)")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("labels length must match inputs length")

    items = []
    for idx, path in enumerate(args.inputs):
        label = args.labels[idx] if args.labels else _default_label(path)
        keys, vals, ns = _load_widths(path)
        items.append((label, keys, vals, ns))

    n_plots = len(items)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    def _mix(c1: Tuple[float, float, float], c2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        t = float(np.clip(t, 0.0, 1.0))
        return (c1[0] * (1 - t) + c2[0] * t, c1[1] * (1 - t) + c2[1] * t, c1[2] * (1 - t) + c2[2] * t)

    def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
        h = h.strip().lstrip("#")
        if len(h) != 6:
            return (0.65, 0.80, 0.90)
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)

    def _n_to_color(n: int, *, n_min: int, n_max: int, base_hex: str, dark_hex: str) -> Tuple[float, float, float]:
        if n <= 0 or n_max <= n_min:
            return _hex_to_rgb01(base_hex)
        # Use sqrt scaling to give low-n still visible differences without going too dark.
        t = (sqrt(n) - sqrt(n_min)) / (sqrt(n_max) - sqrt(n_min) + 1e-9)
        # Keep it light overall; never reach fully "dark".
        t = 0.15 + 0.65 * float(np.clip(t, 0.0, 1.0))
        return _mix(_hex_to_rgb01(base_hex), _hex_to_rgb01(dark_hex), t)

    # Bars facets (clean): one row per input, n encoded by bar shade, n label inside bar.
    if args.style == "bars_facets_clean":
        base_hex = str(args.color)
        dark_hex = str(args.accent)
        all_ns: List[int] = []
        for _, _, _, ns in items:
            all_ns.extend([int(v) for v in ns if int(v) > 0])
        n_min = min(all_ns) if all_ns else 1
        n_max = max(all_ns) if all_ns else 1

        fig = plt.figure(figsize=(12.0, max(4.5, 2.6 * n_plots)), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n_plots, ncols=1, hspace=0.22)

        for i, (label, keys, vals, ns) in enumerate(items):
            ax = fig.add_subplot(gs[i, 0])
            x = np.arange(len(keys), dtype=float)
            y = np.asarray(vals, dtype=float)

            colors = [_n_to_color(int(nv), n_min=n_min, n_max=n_max, base_hex=base_hex, dark_hex=dark_hex) for nv in ns]
            bars = ax.bar(
                x,
                y,
                color=colors,
                edgecolor="#9dbfd4",
                linewidth=0.8,
                alpha=0.95,
                zorder=2,
            )

            # Target line + label above the line (avoid overlap).
            ax.axhline(float(args.target), color=args.target_color, linestyle="--", linewidth=1.2, alpha=0.95, zorder=1)
            ax.text(
                0.99,
                float(args.target) + 0.02,
                f"target={float(args.target):.2f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=9,
                color=args.target_color,
                clip_on=False,
            )

            # n labels inside bars (centered). If bar is too short, place near the top inside.
            for b, nv, yv in zip(bars, ns, y.tolist()):
                if int(nv) <= 0:
                    continue
                yv = float(yv)
                if yv <= 0:
                    continue
                # Prefer consistent placement: near the top inside the bar.
                if yv < 0.08:
                    yy = yv * 0.55
                    va = "center"
                else:
                    yy = max(0.02, yv - 0.035)
                    va = "top"
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    yy,
                    f"n={int(nv)}",
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="#1f4e6a",
                )

            ax.set_title(f"{label} | width-stratified PICP", fontsize=11, pad=10)
            ax.set_ylabel("PICP")
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([_compact_width_label(k) for k in keys], rotation=20, ha="right", fontsize=9)

        if args.title:
            fig.suptitle(args.title, fontsize=12, y=1.02)

        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print("saved:", out_path)
        return

    # Bubble (single panel): x=width bin, y=PICP, color/marker=series, size~n.
    # This is a lightweight structural change from bars: fewer repeated axes, clearer "support".
    if args.style == "bubble":
        # Union bins across series (keeps a single shared x-axis).
        all_keys: List[str] = []
        for _, keys, _, _ in items:
            for k in keys:
                if k not in all_keys:
                    all_keys.append(k)
        all_keys = sorted(all_keys, key=_width_key_sort_key)
        x = np.arange(len(all_keys), dtype=float)

        # Global n range (for comparable sizing).
        all_ns: List[int] = []
        for _, _, _, ns in items:
            all_ns.extend([int(v) for v in ns if int(v) > 0])
        n_min = min(all_ns) if all_ns else 1
        n_max = max(all_ns) if all_ns else 1

        def size_from_n(n: int) -> float:
            if n <= 0:
                return 0.0
            s_min, s_max = 28.0, 520.0
            if n_max == n_min:
                return 120.0
            t = (sqrt(n) - sqrt(n_min)) / (sqrt(n_max) - sqrt(n_min) + 1e-9)
            return float(s_min + t * (s_max - s_min))

        fig, ax = plt.subplots(figsize=(11.0, 7.2))
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.axhline(float(args.target), color=args.target_color, linestyle="--", linewidth=1.4, alpha=0.9, zorder=1)
        ax.text(len(all_keys) - 0.35, float(args.target) + 0.015, f"target={float(args.target):.2f}", ha="right", va="bottom", fontsize=10, color=args.target_color)

        palette = plt.get_cmap("tab10")
        markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
        jitter = np.linspace(-0.22, 0.22, num=max(1, len(items)))

        for si, (label, keys, vals, ns) in enumerate(items):
            key_to_idx = {k: i for i, k in enumerate(keys)}
            xs, ys, ss = [], [], []
            low_xs, low_ys, low_ss = [], [], []
            for j, k in enumerate(all_keys):
                if k not in key_to_idx:
                    continue
                idx = key_to_idx[k]
                yv = float(vals[idx])
                nv = int(ns[idx])
                s = size_from_n(nv)
                xj = float(x[j] + jitter[si])
                xs.append(xj)
                ys.append(yv)
                ss.append(s)
                if args.n_warn and int(args.n_warn) > 0 and 0 < nv < int(args.n_warn):
                    low_xs.append(xj)
                    low_ys.append(yv)
                    low_ss.append(s)

            color = palette(si % 10)
            ax.scatter(
                xs,
                ys,
                s=ss,
                marker=markers[si % len(markers)],
                color=color,
                edgecolor="#2f3b46",
                linewidth=0.75,
                alpha=0.9,
                label=str(label),
                zorder=3,
            )
            # low-n emphasis: red ring
            if low_xs:
                ax.scatter(
                    low_xs,
                    low_ys,
                    s=[v + 50.0 for v in low_ss],
                    marker=markers[si % len(markers)],
                    facecolors="none",
                    edgecolors="#b23a48",
                    linewidth=1.15,
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("PICP (coverage)")
        ax.set_xlabel("Interval width bin")
        ax.set_xticks(x)
        ax.set_xticklabels([_compact_width_label(k) for k in all_keys], rotation=0, ha="center")
        ax.set_title(args.title or "Width-stratified coverage (bubble: size=n)")

        # Legend: series labels + support-size guide.
        leg1 = ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92, ncol=2)
        ax.add_artist(leg1)

        size_refs = [n_min, int((n_min + n_max) / 2), n_max]
        size_handles = [
            ax.scatter([], [], s=size_from_n(v), marker="o", color="#cbd5e1", edgecolor="#2f3b46", linewidth=0.75, label=f"n={v}")
            for v in size_refs
        ]
        ax.legend(handles=size_handles, loc="lower right", fontsize=9, frameon=True, framealpha=0.92, title="support")

        fig.tight_layout()
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=170)
        print("saved:", out_path)
        return

    # Reliability-style: (top) PICP with binomial CI, (bottom) sample count panel.
    # This aligns with common "reliability diagram + count histogram" layouts.
    if args.style == "reliability":
        if not (0.0 < float(args.ci) < 1.0):
            raise ValueError("--ci must be in (0, 1)")
        z = _norm_ppf(0.5 + float(args.ci) / 2.0)

        layout = str(args.reliability_layout)
        if layout == "auto":
            # Many stacked panels become visually noisy; prefer grouped small-multiples.
            layout = "grouped" if n_plots > 3 else "stacked"

        if layout == "stacked":
            # 2 rows per series: coverage panel + count panel.
            n_rows = n_plots * 2
            fig = plt.figure(figsize=(11.5, max(3.8, 2.7 * n_plots)), constrained_layout=True)
            gs = fig.add_gridspec(
                nrows=n_rows,
                ncols=1,
                height_ratios=[3.0, 1.15] * n_plots,
                hspace=0.18,
            )

            for i, (label, keys, vals, ns) in enumerate(items):
                ax = fig.add_subplot(gs[2 * i])
                ax_n = fig.add_subplot(gs[2 * i + 1], sharex=ax)

                x = np.arange(len(keys), dtype=float)
                y = np.asarray(vals, dtype=float)
                n_arr = np.asarray(ns, dtype=float)

                ci_lo = np.full_like(y, np.nan, dtype=float)
                ci_hi = np.full_like(y, np.nan, dtype=float)
                for j, (p_hat, n_val) in enumerate(zip(y.tolist(), ns)):
                    lo, hi = _wilson_ci(p_hat, int(n_val), z=float(z))
                    ci_lo[j] = lo
                    ci_hi[j] = hi

                # Color code points by whether they are clearly under/over target (CI excludes target).
                good = (ci_lo <= float(args.target)) & (ci_hi >= float(args.target))
                under = ci_hi < float(args.target)
                over = ci_lo > float(args.target)

                ax.axhline(
                    float(args.target),
                    color=args.target_color,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"target={float(args.target):.2f}",
                    zorder=1,
                )

                # Base line for continuity.
                ax.plot(x, y, color=args.accent, linewidth=1.6, alpha=0.9, zorder=2)

                # Error bars + markers (split by category to keep legend clean).
                def _plot_mask(mask: np.ndarray, *, color: str, label_txt: str) -> None:
                    idx = np.where(mask)[0]
                    if idx.size == 0:
                        return
                    yerr = np.vstack([y[idx] - ci_lo[idx], ci_hi[idx] - y[idx]])
                    ax.errorbar(
                        x[idx],
                        y[idx],
                        yerr=yerr,
                        fmt="o",
                        color=color,
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=2.4,
                        markersize=4.8,
                        label=label_txt,
                        zorder=3,
                    )

                _plot_mask(good, color=args.accent, label_txt=f"{int(args.ci * 100)}% CI (incl. target)")
                _plot_mask(under, color="#d55e00", label_txt="CI below target")
                _plot_mask(over, color="#0072b2", label_txt="CI above target")

                ax.set_title(f"{label} | width-stratified coverage (PICP)")
                ax.set_ylabel("PICP")
                ax.set_ylim(0.0, 1.0)
                ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.5)
                ax.tick_params(axis="x", which="both", labelbottom=False)

                # n panel: count bars (optionally log-scale) + low-n highlighting.
                n_colors = np.array(["#edf2f7"] * len(ns), dtype=object)
                if args.n_warn and int(args.n_warn) > 0:
                    for j, n_val in enumerate(ns):
                        if int(n_val) < int(args.n_warn):
                            n_colors[j] = "#fde2e4"  # low-n highlight

                ax_n.bar(
                    x,
                    np.maximum(n_arr, 1.0) if args.n_scale == "log" else n_arr,
                    color=n_colors.tolist(),
                    edgecolor="#d5e1ec",
                    linewidth=0.6,
                    alpha=0.92,
                    zorder=1,
                )
                ax_n.set_ylabel("n", color="#6b7b8a")
                ax_n.tick_params(axis="y", colors="#6b7b8a")
                ax_n.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.5)
                ax_n.set_xticks(x)
                ax_n.set_xticklabels(keys, rotation=20, ha="right")
                if args.n_scale == "log":
                    ax_n.set_yscale("log")
                    ax_n.set_ylim(bottom=1.0)
                else:
                    ax_n.set_ylim(bottom=0.0)

                # Labels are very dense; only annotate low-n bins in reliability mode.
                if args.n_mode == "label" and args.n_warn and int(args.n_warn) > 0:
                    for j, n_val in enumerate(ns):
                        if int(n_val) >= int(args.n_warn):
                            continue
                        y0 = float(n_arr[j])
                        ax_n.text(
                            x[j],
                            y0 + (0.02 * max(1.0, float(np.nanmax(n_arr)))) if args.n_scale == "linear" else y0 * 1.05,
                            f"{int(n_val)}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#4c4c4c",
                            rotation=0,
                        )

                # Keep legend compact: only show on first panel.
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8, frameon=True)

            if args.title:
                fig.suptitle(args.title)
            out_path = args.out
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=170)
            print("saved:", out_path)
            return

        # Grouped layout: 2 rows (PICP / n) x K columns (families). Less busy, more "story".
        fam_order = ["greedy", "beam/full", "teacher", "other"]
        fam_to_series: Dict[str, List[Tuple[str, List[str], List[float], List[int]]]] = {k: [] for k in fam_order}
        for (label, keys, vals, ns) in items:
            fam = _family_from_label(label)
            fam = "beam/full" if fam in ("beam", "full") else fam
            if fam not in fam_to_series:
                fam_to_series[fam] = []
                fam_order.append(fam)
            fam_to_series[fam].append((label, keys, vals, ns))

        active_fams = [f for f in fam_order if fam_to_series.get(f)]
        n_cols = len(active_fams)
        fig = plt.figure(figsize=(12.6, max(4.6, 3.2 + 1.6 * n_cols)), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=n_cols, height_ratios=[3.0, 1.25], wspace=0.25, hspace=0.06)

        palette = plt.get_cmap("tab10")
        for col, fam in enumerate(active_fams):
            series = fam_to_series[fam]
            # Union bins for this family.
            all_keys: List[str] = []
            for _, keys, _, _ in series:
                for k in keys:
                    if k not in all_keys:
                        all_keys.append(k)
            all_keys = sorted(all_keys, key=_width_key_sort_key)
            x = np.arange(len(all_keys), dtype=float)

            ax = fig.add_subplot(gs[0, col])
            ax_n = fig.add_subplot(gs[1, col], sharex=ax)

            ax.axhline(
                float(args.target),
                color=args.target_color,
                linestyle="--",
                linewidth=1.25,
                alpha=0.9,
                zorder=1,
            )

            for si, (label, keys, vals, ns) in enumerate(series):
                color = palette(si % 10)
                key_to_idx = {k: i for i, k in enumerate(keys)}
                y = np.full(len(all_keys), np.nan, dtype=float)
                n_arr = np.zeros(len(all_keys), dtype=int)
                for j, k in enumerate(all_keys):
                    if k not in key_to_idx:
                        continue
                    idx = key_to_idx[k]
                    y[j] = float(vals[idx])
                    n_arr[j] = int(ns[idx])

                # CI whiskers: thin + semi-transparent to avoid clutter.
                ci_lo = np.full_like(y, np.nan, dtype=float)
                ci_hi = np.full_like(y, np.nan, dtype=float)
                for j in range(len(all_keys)):
                    if not (y[j] == y[j]):  # NaN
                        continue
                    lo, hi = _wilson_ci(float(y[j]), int(n_arr[j]), z=float(z))
                    ci_lo[j] = lo
                    ci_hi[j] = hi

                # Marker size: optionally encode n (log-compressed); otherwise keep consistent.
                if args.n_mode == "size" and np.any(n_arr > 0):
                    n_min = int(np.min(n_arr[n_arr > 0])) if np.any(n_arr > 0) else 1
                    n_max = int(np.max(n_arr)) if np.any(n_arr > 0) else 1
                    sizes = []
                    for nv in n_arr.tolist():
                        if nv <= 0 or n_max == n_min:
                            sizes.append(28.0)
                        else:
                            s = 24.0 + 90.0 * (
                                (np.log1p(nv) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
                            )
                            sizes.append(float(s))
                    sizes = np.asarray(sizes, dtype=float)
                else:
                    sizes = np.full(len(all_keys), 36.0, dtype=float)

                # Low-n emphasis: dark edge for low support.
                edgecolors = []
                for nv in n_arr.tolist():
                    if args.n_warn and int(args.n_warn) > 0 and 0 < int(nv) < int(args.n_warn):
                        edgecolors.append("#b23a48")
                    else:
                        edgecolors.append("#263238")

                # Plot line (masked NaNs) + points + whiskers.
                ax.plot(x, y, color=color, linewidth=1.35, alpha=0.9, zorder=2)
                idx = np.where(~np.isnan(y))[0]
                if idx.size:
                    yerr = np.vstack([y[idx] - ci_lo[idx], ci_hi[idx] - y[idx]])
                    ax.errorbar(
                        x[idx],
                        y[idx],
                        yerr=yerr,
                        fmt="none",
                        ecolor=color,
                        elinewidth=0.9,
                        capsize=2.0,
                        alpha=0.32,
                        zorder=2,
                    )
                    ax.scatter(
                        x[idx],
                        y[idx],
                        s=sizes[idx],
                        color=color,
                        edgecolor=np.array(edgecolors, dtype=object)[idx].tolist(),
                        linewidth=0.65,
                        alpha=0.95,
                        label=label,
                        zorder=3,
                    )

                # n panel: side-by-side bars with high transparency (support only).
                m = max(1, len(series))
                bw = min(0.78 / m, 0.26)
                offset = (si - (m - 1) / 2.0) * bw
                n_plot = np.maximum(n_arr.astype(float), 1.0) if args.n_scale == "log" else n_arr.astype(float)
                ax_n.bar(
                    x + offset,
                    n_plot,
                    width=bw,
                    color=color,
                    edgecolor="#546e7a",
                    linewidth=0.4,
                    alpha=0.28,
                    zorder=1,
                )
                if args.n_mode == "label" and args.n_warn and int(args.n_warn) > 0:
                    for j, nv in enumerate(n_arr.tolist()):
                        if not (0 < int(nv) < int(args.n_warn)):
                            continue
                        y0 = float(n_plot[j])
                        ax_n.text(
                            x[j] + offset,
                            y0 + (0.02 * max(1.0, float(np.nanmax(n_plot)))) if args.n_scale == "linear" else y0 * 1.08,
                            f"{int(nv)}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#4c4c4c",
                        )

            ax.set_title(fam, fontsize=11)
            ax.set_ylabel("PICP" if col == 0 else "")
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.20, linestyle="--", linewidth=0.5)
            ax.tick_params(axis="x", which="both", labelbottom=False)
            ax.legend(loc="upper right", fontsize=8, frameon=True)

            ax_n.set_ylabel("n" if col == 0 else "", color="#6b7b8a")
            ax_n.tick_params(axis="y", colors="#6b7b8a")
            ax_n.grid(axis="y", alpha=0.16, linestyle="--", linewidth=0.5)
            ax_n.set_xticks(x)
            ax_n.set_xticklabels(all_keys, rotation=20, ha="right")
            if args.n_scale == "log":
                ax_n.set_yscale("log")
                ax_n.set_ylim(bottom=1.0)
            else:
                ax_n.set_ylim(bottom=0.0)

        if args.title:
            fig.suptitle(args.title)
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print("saved:", out_path)
        return

    if args.style == "heatmap":
        labels = [it[0] for it in items]
        all_keys: List[str] = []
        for _, keys, _, _ in items:
            for k in keys:
                if k not in all_keys:
                    all_keys.append(k)
        all_keys = sorted(all_keys, key=_width_key_sort_key)
        mat = []
        n_mat = []
        for _, keys, vals, ns in items:
            row = []
            n_row = []
            key_to_idx = {k: i for i, k in enumerate(keys)}
            for k in all_keys:
                if k in key_to_idx:
                    idx = key_to_idx[k]
                    row.append(vals[idx])
                    n_row.append(ns[idx])
                else:
                    row.append(float("nan"))
                    n_row.append(0)
            mat.append(row)
            n_mat.append(n_row)

        fig, ax = plt.subplots(figsize=(12, max(2.6, 1.1 * len(labels))))
        data = plt.cm.Blues(mat)
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels(all_keys, rotation=20, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("width bin")
        ax.set_ylabel("mode")
        for i in range(len(labels)):
            for j in range(len(all_keys)):
                val = mat[i][j]
                n_val = n_mat[i][j]
                if not (val == val):  # NaN
                    continue
                txt = f"{val:.2f}\\n(n={n_val})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#1f2d3d")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("PICP")
        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout()
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print("saved:", out_path)
        return

    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        # Keep the multi-row figure compact; we hide intermediate x tick labels.
        figsize=(11.2, max(3.8, 1.55 * n_plots + 1.1)),
        sharex=False,
    )
    if n_plots == 1:
        axes = [axes]

    for plot_idx, (ax, (label, keys, vals, ns)) in enumerate(zip(axes, items)):
        x = list(range(len(keys)))
        bars = [None] * len(vals)
        if args.style in ("bars", "bars_rug"):
            bars = ax.bar(x, vals, color=args.color, edgecolor="#8fbcd6", linewidth=0.6, alpha=0.95, zorder=2)
        elif args.style == "dual":
            # Clean dual-axis layout:
            # - Right axis: background bars show support n (very light blue).
            # - Left axis: foreground bars show PICP (light blue), narrower for separation.
            ax2 = ax.twinx()
            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

            n_bars = ax2.bar(
                x,
                ns,
                width=0.84,
                color="#eaf3fb",
                edgecolor="#d6e6f2",
                linewidth=0.6,
                alpha=1.0,
                zorder=0,
            )
            ax2.set_ylabel("n" if plot_idx == 0 else "", color="#6b7b8a")
            ax2.tick_params(axis="y", colors="#6b7b8a", labelsize=9)
            ax2.grid(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_color("#d6e6f2")

            bars = ax.bar(
                x,
                vals,
                width=0.54,
                color="#bfe3f7",
                edgecolor="#88c3e6",
                linewidth=0.7,
                alpha=0.98,
                zorder=2,
            )
        elif args.style in ("line", "line_annot_light", "line_annot_full"):
            sizes = None
            if args.n_mode == "size":
                n_min = min(ns) if ns else 0
                n_max = max(ns) if ns else 1
                sizes = []
                for n_val in ns:
                    if n_max == n_min:
                        s = 40.0
                    else:
                        s = 30.0 + 70.0 * (
                            (np.log1p(n_val) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
                        )
                    sizes.append(s)
            if args.style == "line":
                # Encode n by marker lightness (still within the same light-blue family).
                # This is a low-ink cue that avoids sprinkling text everywhere.
                n_pos = [int(v) for v in ns if int(v) > 0]
                n_min = min(n_pos) if n_pos else 1
                n_max = max(n_pos) if n_pos else 1
                colors = []
                for n_val in ns:
                    if n_max == n_min or int(n_val) <= 0:
                        t = 0.0
                    else:
                        t = (np.log1p(int(n_val)) - np.log1p(int(n_min))) / (np.log1p(int(n_max)) - np.log1p(int(n_min)))
                    colors.append(_blend_hex("#dff2fd", str(args.accent), float(t)))
            else:
                # Annotation-oriented variant: keep the line/markers consistent (less to decode).
                colors = [args.accent] * len(vals)

            ax.plot(x, vals, color=args.accent, linewidth=1.9, zorder=3, alpha=0.95)
            ax.scatter(x, vals, s=sizes or 44, color=colors, edgecolor="#5f9fc5", linewidth=0.7, zorder=4)
            ax.fill_between(x, vals, [0.0] * len(vals), color=args.color, alpha=0.22, zorder=1)

            if args.style == "line_annot_full":
                # Add compact numeric PICP labels near markers (still low-ink).
                for xi, yv in zip(x, vals):
                    yv = float(yv)
                    yy = min(0.985, yv + 0.035)
                    ax.text(
                        xi,
                        yy,
                        f"{yv:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#1f4e6a",
                    )
        else:
            ax.vlines(x, [0.0] * len(vals), vals, color=args.color, linewidth=2.2, zorder=2)
            sizes = None
            if args.n_mode == "size":
                n_min = min(ns) if ns else 0
                n_max = max(ns) if ns else 1
                sizes = []
                for n_val in ns:
                    if n_max == n_min:
                        s = 40.0
                    else:
                        s = 30.0 + 70.0 * (
                            (np.log1p(n_val) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
                        )
                    sizes.append(s)
            ax.scatter(x, vals, s=sizes or 42, color=args.accent, zorder=3)

        ax.axhline(args.target, color=args.target_color, linestyle="--", linewidth=1.2, zorder=0, alpha=0.95)
        # Place target label above the dashed line (avoid overlapping with the dash).
        ax.annotate(
            f"target={float(args.target):.2f}",
            xy=(1.0, float(args.target)),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=args.target_color,
        )
        ax.set_title(str(label), loc="left", fontsize=11, pad=4)
        ax.set_ylabel("PICP")
        if args.style == "bars_rug":
            ax.set_ylim(-0.14, 1.0)  # room for support "rug" below axis
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        # Keep x labels compact and aligned; long "Width [...]" is visual noise.
        ax.set_xticklabels([_compact_width_label(k) for k in keys], rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Reduce clutter in multi-row figures: only show x tick labels on the last panel.
        if n_plots > 1 and plot_idx != (n_plots - 1):
            ax.tick_params(axis="x", which="both", labelbottom=False, length=0)

        if args.style == "bars_rug":
            # "Fun but light": encode n as a dot-rug below the baseline.
            # - Dot size ~ log(n) (per-subplot) so n=6 vs n=810 is obvious.
            # - Red edge highlights low support (n < n_warn).
            y_rug = -0.085
            n_arr = np.asarray(ns, dtype=float)
            n_pos = n_arr[n_arr > 0]
            n_min = float(np.min(n_pos)) if n_pos.size else 1.0
            n_max = float(np.max(n_pos)) if n_pos.size else 1.0
            sizes = []
            edgecolors = []
            for n_val in ns:
                if n_val <= 0 or n_max == n_min:
                    s = 26.0
                else:
                    s = 22.0 + 110.0 * (
                        (np.log1p(n_val) - np.log1p(n_min)) / (np.log1p(n_max) - np.log1p(n_min))
                    )
                sizes.append(float(s))
                if args.n_warn and int(args.n_warn) > 0 and 0 < int(n_val) < int(args.n_warn):
                    edgecolors.append("#b23a48")
                else:
                    edgecolors.append("#64748b")

            ax.scatter(
                x,
                [y_rug] * len(x),
                s=sizes,
                color="#cbd5e1",
                edgecolor=edgecolors,
                linewidth=0.9,
                zorder=4,
                alpha=0.95,
            )
            ax.axhline(0.0, color="#94a3b8", linewidth=0.7, alpha=0.55, zorder=1)
            ax.set_yticks([0.0, 0.5, 0.9, 1.0])
            ax.tick_params(axis="x", length=0)

            # Only label very small-n bins (otherwise it's text soup).
            if args.n_mode == "label" and args.n_warn and int(args.n_warn) > 0:
                for xi, n_val in zip(x, ns):
                    if not (0 < int(n_val) < int(args.n_warn)):
                        continue
                    ax.text(
                        xi,
                        y_rug + 0.03,
                        f"n={int(n_val)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#334155",
                    )

            # Avoid repeated legend noise; keep target label as subtle text in first panel only.
            if plot_idx == 0:
                ax.text(
                    max(x) + 0.15 if x else 0.0,
                    float(args.target),
                    f"target={float(args.target):.2f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    color=args.target_color,
                )
        elif args.n_mode == "bar" and args.style != "dual":
            ax2 = ax.twinx()
            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            # Very light background bars for n; optionally shade by n for subtle hierarchy.
            n_pos = [int(v) for v in ns if int(v) > 0]
            n_min = min(n_pos) if n_pos else 1
            n_max = max(n_pos) if n_pos else 1
            bar_colors = []
            for n_val in ns:
                if n_max == n_min or int(n_val) <= 0:
                    t = 0.0
                else:
                    t = (sqrt(int(n_val)) - sqrt(int(n_min))) / (sqrt(int(n_max)) - sqrt(int(n_min)) + 1e-9)
                # keep it light: only blend part-way to accent
                t = 0.15 + 0.55 * float(np.clip(t, 0.0, 1.0))
                bar_colors.append(_blend_hex("#f3f9fe", str(args.accent), float(t)))

            ax2.bar(x, ns, color=bar_colors, edgecolor="#d6e6f2", linewidth=0.6, alpha=1.0, zorder=0)
            # Keep the right axis minimal (n is encoded visually + low-n labels).
            ax2.set_ylabel("n" if plot_idx == 0 else "", color="#6b7b8a")
            ax2.tick_params(axis="y", colors="#6b7b8a", labelsize=9)
            ax2.grid(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_color("#d6e6f2")

            ymax = max([int(v) for v in ns] + [1])
            ax2.set_ylim(0.0, ymax * 1.08)

            # In annotation-oriented line styles, label all n inside bars (still readable; bins are few).
            # Otherwise, only label low-n bins to avoid turning into text soup.
            label_all = args.style in ("line_annot_light", "line_annot_full")
            if label_all or (args.n_warn and int(args.n_warn) > 0):
                for xi, n_val in zip(x, ns):
                    if int(n_val) <= 0:
                        continue
                    if (not label_all) and (not (0 < int(n_val) < int(args.n_warn))):
                        continue
                    ax2.text(
                        xi,
                        max(1.0, float(n_val) * 0.62),
                        f"n={int(n_val)}",
                        ha="center",
                        va="center",
                        fontsize=9 if label_all else 8,
                        color="#1f4e6a" if label_all else "#334155",
                        alpha=0.9 if label_all else 0.95,
                    )

            # Tiny legend hint (first panel only): line=PICP, bars=support n.
            if plot_idx == 0 and args.style in ("line_annot_light", "line_annot_full"):
                handles = [
                    Line2D([0], [0], color=args.accent, linewidth=2.0, label="PICP"),
                    Patch(facecolor=_blend_hex("#f3f9fe", str(args.accent), 0.35), edgecolor="#d6e6f2", label="support n"),
                ]
                ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.92)
        elif args.n_mode == "label":
            # Default label mode is too busy for report figures; only label low-n bins.
            if args.n_warn and int(args.n_warn) > 0:
                for idx, n in enumerate(ns):
                    if not (0 < int(n) < int(args.n_warn)):
                        continue
                    y_val = vals[idx] if idx < len(vals) else 0.0
                    ax.text(
                        x[idx],
                        max(0.02, float(y_val) - 0.06),
                        f"n={int(n)}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="#4c4c4c",
                    )

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout(pad=0.6, h_pad=0.25)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
