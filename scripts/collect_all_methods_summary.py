#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "tabseq",
    "catboost",
    "xgboost",
    "lightgbm",
    "catboost_quantile",
    "xgboost_quantile",
    "lightgbm_quantile",
)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(float(value))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6f}".rstrip("0").rstrip("."))
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join([header, sep, *body]) + "\n", encoding="utf-8")


def _parse_list(raw: Optional[str], *, fallback: Iterable[str]) -> list[str]:
    if raw is None or str(raw).strip() == "":
        return list(fallback)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _latest_tabseq_summary(batch_runs_root: Path) -> Path:
    candidates = [path for path in batch_runs_root.glob("*/summary.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no summary.csv found under {batch_runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_tabseq_rows(summary_csv: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for src in reader:
            dataset = str(src["dataset"])
            rows[dataset] = {
                "dataset": dataset,
                "method": "tabseq",
                "split": src.get("test.split") or "val",
                "confidence": _to_float(src.get("test.confidence") or src.get("best_confidence")),
                "target_confidence": _to_float(src.get("test.target_confidence") or src.get("train_summary.target_confidence")),
                "calibrated_confidence": _to_float(
                    src.get("best_calibrated_confidence") or src.get("train_summary.calibrated_confidence") or src.get("test.confidence")
                ),
                "depth": _to_int(src.get("best_depth")),
                "interval_method": src.get("test.interval_method") or src.get("best_interval_method"),
                "bin_acc": _to_float(src.get("test.bin_acc")),
                "tol_bin_acc@1": _to_float(src.get("test.tol_bin_acc@1")),
                "avg_coverage": _to_float(src.get("test.avg_coverage")),
                "avg_length": _to_float(src.get("test.avg_length")),
                "best_epoch": _to_int(src.get("train_summary.best_epoch") or src.get("val.best_epoch")),
                "beam_size": _to_int(src.get("test.beam_size") or src.get("best_beam_size")),
                "leaf_prior_weight": _to_float(src.get("test.leaf_prior_weight") or src.get("best_leaf_prior_weight")),
                "run_dir": src.get("best_run_dir"),
                "source_file": str(summary_csv.resolve()),
                "batch_id": src.get("batch_id"),
                "status": src.get("status") or "ok",
            }
    return rows


def _latest_baseline_metrics(baseline_root: Path, dataset: str, method: str) -> Optional[Path]:
    method_dir = baseline_root / dataset / method
    if not method_dir.is_dir():
        return None
    candidates = [path for path in method_dir.glob("run_*/metrics_val.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_baseline_row(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        src = json.load(f)
    return {
        "dataset": str(src["dataset"]),
        "method": str(src["model"]),
        "split": src.get("split") or "val",
        "confidence": _to_float(src.get("confidence")),
        "target_confidence": _to_float(src.get("target_confidence") or src.get("confidence")),
        "calibrated_confidence": _to_float(src.get("calibrated_confidence") or src.get("confidence")),
        "depth": _to_int(src.get("depth")),
        "interval_method": src.get("interval_method"),
        "bin_acc": _to_float(src.get("bin_acc")),
        "tol_bin_acc@1": _to_float(src.get("tol_bin_acc@1")),
        "avg_coverage": _to_float(src.get("avg_coverage")),
        "avg_length": _to_float(src.get("avg_length")),
        "best_epoch": None,
        "beam_size": None,
        "leaf_prior_weight": None,
        "run_dir": str(metrics_path.parent.resolve()),
        "source_file": str(metrics_path.resolve()),
        "batch_id": None,
        "status": "ok",
    }


def _collect_dataset_names(
    *,
    requested: Optional[list[str]],
    tabseq_rows: dict[str, dict[str, Any]],
    baseline_root: Path,
) -> list[str]:
    if requested:
        return sorted(dict.fromkeys(requested))
    names = set(tabseq_rows.keys())
    if baseline_root.is_dir():
        for path in baseline_root.iterdir():
            if path.is_dir():
                names.add(path.name)
    return sorted(names)


def _build_wide_rows(long_rows: list[dict[str, Any]], datasets: list[str], methods: list[str]) -> list[dict[str, Any]]:
    by_dataset_method = {(row["dataset"], row["method"]): row for row in long_rows}
    metric_fields = [
        "bin_acc",
        "tol_bin_acc@1",
        "avg_coverage",
        "avg_length",
        "confidence",
        "target_confidence",
        "calibrated_confidence",
        "depth",
        "best_epoch",
        "beam_size",
        "leaf_prior_weight",
        "interval_method",
        "run_dir",
    ]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        row: dict[str, Any] = {"dataset": dataset}
        for method in methods:
            data = by_dataset_method.get((dataset, method))
            for field in metric_fields:
                row[f"{method}.{field}"] = None if data is None else data.get(field)
        rows.append(row)
    return rows


def _build_metrics_rows(long_rows: list[dict[str, Any]], datasets: list[str], methods: list[str]) -> list[dict[str, Any]]:
    by_dataset_method = {(row["dataset"], row["method"]): row for row in long_rows}
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            data = by_dataset_method.get((dataset, method))
            if data is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "bin_acc": data.get("bin_acc"),
                    "tol_bin_acc@1": data.get("tol_bin_acc@1"),
                    "avg_coverage": data.get("avg_coverage"),
                    "avg_length": data.get("avg_length"),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabseq-summary", type=str, default=None, help="TabSeq summary.csv; default is latest under outputs/batch_runs")
    ap.add_argument("--baseline-root", type=str, default="outputs/baselines_four_metrics")
    ap.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names; default is union of discovered datasets")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--out-dir", type=str, default=None, help="default: outputs/comparisons/all_methods_<timestamp>")
    args = ap.parse_args()

    methods = _parse_list(args.methods, fallback=DEFAULT_METHODS)
    batch_runs_root = ROOT / "outputs" / "batch_runs"
    tabseq_summary = Path(args.tabseq_summary).resolve() if args.tabseq_summary else _latest_tabseq_summary(batch_runs_root)
    baseline_root = (ROOT / args.baseline_root).resolve() if not Path(args.baseline_root).is_absolute() else Path(args.baseline_root)

    tabseq_rows = _load_tabseq_rows(tabseq_summary) if "tabseq" in methods else {}
    datasets = _collect_dataset_names(
        requested=_parse_list(args.datasets, fallback=()) if args.datasets else None,
        tabseq_rows=tabseq_rows,
        baseline_root=baseline_root,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ROOT / "outputs" / "comparisons" / f"all_methods_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for method in methods:
            if method == "tabseq":
                row = tabseq_rows.get(dataset)
            else:
                metrics_path = _latest_baseline_metrics(baseline_root, dataset, method)
                row = _load_baseline_row(metrics_path) if metrics_path is not None else None
            if row is None:
                missing_rows.append({"dataset": dataset, "method": method, "reason": "result_not_found"})
                continue
            long_rows.append(row)

    long_rows.sort(key=lambda row: (str(row["dataset"]), str(row["method"])))
    wide_rows = _build_wide_rows(long_rows, datasets, methods)
    metrics_rows = _build_metrics_rows(long_rows, datasets, methods)

    long_csv = out_dir / "summary_long.csv"
    long_json = out_dir / "summary_long.json"
    wide_csv = out_dir / "summary_wide.csv"
    wide_json = out_dir / "summary_wide.json"
    metrics_csv = out_dir / "summary_metrics.csv"
    metrics_md = out_dir / "summary_metrics.md"
    missing_csv = out_dir / "missing.csv"
    missing_json = out_dir / "missing.json"

    _write_csv(long_csv, long_rows)
    _write_json(long_json, long_rows)
    _write_csv(wide_csv, wide_rows)
    _write_json(wide_json, wide_rows)
    _write_csv(metrics_csv, metrics_rows)
    _write_markdown_table(
        metrics_md,
        metrics_rows,
        ["dataset", "method", "bin_acc", "tol_bin_acc@1", "avg_coverage", "avg_length"],
    )
    _write_csv(missing_csv, missing_rows)
    _write_json(missing_json, missing_rows)

    print(f"tabseq_summary: {tabseq_summary.resolve()}")
    print(f"baseline_root: {baseline_root}")
    print(f"datasets: {datasets}")
    print(f"methods: {methods}")
    print(f"saved long csv: {long_csv.resolve()}")
    print(f"saved wide csv: {wide_csv.resolve()}")
    print(f"saved metrics csv: {metrics_csv.resolve()}")
    print(f"saved metrics md: {metrics_md.resolve()}")
    print(f"saved missing csv: {missing_csv.resolve()}")
    print(f"found_rows={len(long_rows)} missing_rows={len(missing_rows)}")


if __name__ == "__main__":
    main()
