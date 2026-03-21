import argparse
import csv
import json
import os
from typing import List, Optional


def _load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "greedy" in data and "teacher_forcing" in data:
        out = dict(data["greedy"])
        out["mode"] = "greedy"
        return out
    return data


def _pick_metrics_file(dirpath: str, filenames: List[str]) -> Optional[str]:
    candidates = [
        name
        for name in filenames
        if name.startswith("metrics_val") and name.endswith(".json") and "prefix" not in name
    ]
    if not candidates:
        return None
    paths = [os.path.join(dirpath, name) for name in candidates]
    return max(paths, key=os.path.getmtime)


def _load_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _infer_dataset_from_path(run_dir: str) -> Optional[str]:
    # Common layouts:
    # - outputs/<dataset>/run_YYYY.../
    # - outputs/baselines/<dataset>/<model>/run_YYYY.../
    parts = os.path.normpath(run_dir).split(os.sep)
    try:
        # .../outputs/baselines/<dataset>/...
        idx = parts.index("baselines")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    try:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", default="outputs/benchmark.csv")
    ap.add_argument("--seed", type=int, default=None, help="only include runs with this seed")
    ap.add_argument("--models", type=str, default=None, help="comma-separated model names to include")
    ap.add_argument("--latest", action="store_true", help="keep only the latest run per model")
    ap.add_argument(
        "--latest-per-dataset",
        action="store_true",
        help="keep only the latest run per (dataset, model) pair",
    )
    ap.add_argument("--include-extras", action="store_true", help="include width/bin diagnostics")
    args = ap.parse_args()

    allow_models = None
    if args.models:
        allow_models = {name.strip() for name in args.models.split(",") if name.strip()}

    rows = []
    for dirpath, _, filenames in os.walk(args.root):
        metrics_path = _pick_metrics_file(dirpath, filenames)
        if metrics_path is None:
            continue
        metrics = _load_metrics(metrics_path)
        cfg = _load_config(dirpath)
        model_name = metrics.get("model") or cfg.get("model") or "unknown"
        dataset_name = cfg.get("dataset") or metrics.get("dataset") or _infer_dataset_from_path(dirpath)
        if allow_models is not None and model_name not in allow_models:
            continue
        if args.seed is not None:
            seed = cfg.get("seed")
            if seed is None or int(seed) != int(args.seed):
                continue
        extras = {}
        if args.include_extras:
            width_picp = metrics.get("width_stratified_PICP")
            extras["width_stratified_PICP"] = (
                json.dumps(width_picp, ensure_ascii=False) if isinstance(width_picp, dict) else width_picp
            )
            extras["bin_acc_0.2"] = metrics.get("bin_acc_0.2")
            extras["bin_acc_0.4"] = metrics.get("bin_acc_0.4")
        rows.append(
            {
                "run_dir": dirpath,
                "dataset": dataset_name,
                "model": model_name,
                "mode": metrics.get("mode", "n/a"),
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "PICP": metrics.get("PICP"),
                "MPIW": metrics.get("MPIW"),
                "confidence": metrics.get("confidence"),
                "temperature": metrics.get("temperature"),
                "acm_enabled": cfg.get("use_confidence_masking"),
                "alpha_depth_mode": cfg.get("alpha_depth_mode"),
                "alpha_min": cfg.get("alpha_min"),
                "alpha_max": cfg.get("alpha_max"),
                "seed": cfg.get("seed"),
                **extras,
            }
        )

    rows.sort(key=lambda r: r["run_dir"])
    if args.latest and args.latest_per_dataset:
        raise SystemExit("choose only one: --latest or --latest-per-dataset")
    if args.latest:
        latest_by_model = {}
        for row in rows:
            latest_by_model[row["model"]] = row
        rows = list(latest_by_model.values())
        rows.sort(key=lambda r: r["run_dir"])
    if args.latest_per_dataset:
        latest_by_key = {}
        for row in rows:
            key = (row.get("dataset"), row.get("model"))
            latest_by_key[key] = row
        rows = list(latest_by_key.values())
        rows.sort(key=lambda r: r["run_dir"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "run_dir",
        "dataset",
        "model",
        "mode",
        "MAE",
        "RMSE",
        "PICP",
        "MPIW",
        "confidence",
        "temperature",
        "acm_enabled",
        "alpha_depth_mode",
        "alpha_min",
        "alpha_max",
        "seed",
    ]
    if args.include_extras:
        fieldnames.extend(["width_stratified_PICP", "bin_acc_0.2", "bin_acc_0.4"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("saved:", args.out)


if __name__ == "__main__":
    main()
