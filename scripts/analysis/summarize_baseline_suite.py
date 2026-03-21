#!/usr/bin/env python3
"""
Generate a compact markdown report for a baseline suite run_id.

Expected layout:
  outputs/baselines/<dataset>/<model>/run_<run_id>/metrics_val.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{float(x):.{nd}f}"
    return str(x)


def _find_metrics(dataset: str, run_id: str, models: List[str], root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in models:
        path = os.path.join(root, dataset, m, f"run_{run_id}", "metrics_val.json")
        if os.path.isfile(path):
            out[m] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", default="outputs/baselines")
    ap.add_argument(
        "--models",
        default="mlp,quantile,ft_transformer,catboost,realmlp",
        help="comma-separated; report will include those that exist",
    )
    ap.add_argument("--out", default=None, help="write markdown to file (optional)")
    args = ap.parse_args()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    metrics_paths = _find_metrics(str(args.dataset), str(args.run_id), models, str(args.root))
    if not metrics_paths:
        raise FileNotFoundError("no metrics found for given dataset/run_id")

    rows: List[Dict[str, Any]] = []
    eval_spec: Optional[Dict[str, Any]] = None
    for m, p in metrics_paths.items():
        data = _load_json(p)
        if eval_spec is None and isinstance(data.get("eval_spec"), dict):
            eval_spec = data["eval_spec"]
        rows.append(
            {
                "model": m,
                "MAE": data.get("MAE"),
                "RMSE": data.get("RMSE"),
                "PICP": data.get("PICP"),
                "MPIW": data.get("MPIW"),
                "bin_acc_0.2": data.get("bin_acc_0.2"),
                "bin_acc_0.4": data.get("bin_acc_0.4"),
                "width_stratified_PICP": data.get("width_stratified_PICP") or {},
                "path": p,
            }
        )

    lines: List[str] = []
    lines.append(f"# Baseline Suite Report: {args.dataset} / run_id={args.run_id}")
    if eval_spec:
        lines.append("")
        lines.append("## Eval Spec (TabSeq-aligned)")
        for k in ["seed", "confidence", "v_min", "v_max", "clip_range"]:
            if k in eval_spec:
                lines.append(f"- {k}: {eval_spec[k]}")

    lines.append("")
    lines.append("## Point + Interval Metrics")
    lines.append("")
    lines.append("| model | MAE | RMSE | PICP | MPIW | bin_acc_0.2 | bin_acc_0.4 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(rows, key=lambda x: x["model"]):
        lines.append(
            "| {model} | {MAE} | {RMSE} | {PICP} | {MPIW} | {b02} | {b04} |".format(
                model=r["model"],
                MAE=_fmt(r["MAE"]),
                RMSE=_fmt(r["RMSE"]),
                PICP=_fmt(r["PICP"]),
                MPIW=_fmt(r["MPIW"]),
                b02=_fmt(r["bin_acc_0.2"]),
                b04=_fmt(r["bin_acc_0.4"]),
            )
        )

    lines.append("")
    lines.append("## width_stratified_PICP (Diagnostics)")
    for r in sorted(rows, key=lambda x: x["model"]):
        lines.append("")
        lines.append(f"### {r['model']}")
        widths = r["width_stratified_PICP"] or {}
        if not widths:
            lines.append("- (missing width_stratified_PICP)")
            continue
        for k in sorted(widths.keys()):
            lines.append(f"- {k}: {widths[k]}")

    lines.append("")
    lines.append("## Artifact Paths")
    for r in sorted(rows, key=lambda x: x["model"]):
        lines.append(f"- {r['model']}: `{os.path.dirname(r['path'])}`")

    text = "\n".join(lines).rstrip() + "\n"
    if args.out:
        os.makedirs(os.path.dirname(str(args.out)), exist_ok=True)
        with open(str(args.out), "w", encoding="utf-8") as f:
            f.write(text)
        print("saved:", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()

