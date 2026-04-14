#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.data.datasets import DEFAULT_LOCAL_DATA_ROOTS, load_dataset_split
from tabseq.inference import constrained_interval_rank, greedy_metric_rank
from tabseq.training import TrainConfig, train_tabseq_model
from tabseq.utils.config import choose, load_config, resolve_section, write_json


DEFAULT_DATASETS = ["diamonds"]
DEFAULT_EPOCH_GRID = [60]
DEFAULT_TEMPERATURE_GRID = [0.8, 1.0, 1.2]
AUTO_DEPTH_MIN = 4
AUTO_DEPTH_MAX = 9


def _list_complete_local_datasets() -> list[str]:
    names: set[str] = set()
    for root in DEFAULT_LOCAL_DATA_ROOTS:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir() or path.name.startswith("_"):
                continue
            if (path / "table.csv.gz").is_file() and (path / "metadata.json").is_file():
                names.add(path.name)
    return sorted(names)


def _default_selected_datasets(available: list[str]) -> list[str]:
    selected = [name for name in DEFAULT_DATASETS if name in available]
    if selected:
        return selected
    return available


def _latest_metrics_file(run_dir: str) -> str:
    candidates = []
    for name in os.listdir(run_dir):
        if name.startswith("metrics_test_") and name.endswith(".json"):
            candidates.append(os.path.join(run_dir, name))
    if not candidates:
        raise FileNotFoundError(f"no metrics_test_*.json found under {run_dir}")
    return max(candidates, key=os.path.getmtime)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[f"{prefix}{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, list):
            out[f"{prefix}{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            out[f"{prefix}{key}"] = value
    return out


def _pick_width_bins(raw: Any, defaults: Iterable[float]) -> tuple[float, ...]:
    width_bins = choose(raw, list(defaults))
    if isinstance(width_bins, str):
        width_bins = [float(x.strip()) for x in width_bins.split(",") if x.strip()]
    elif isinstance(width_bins, tuple):
        width_bins = list(width_bins)
    return tuple(float(x) for x in width_bins)


def _parse_grid(raw: Optional[str], *, fallback: Any, caster: Callable[[Any], Any]) -> list[Any]:
    if raw is None or str(raw).strip() == "":
        if isinstance(fallback, (list, tuple)):
            values = [caster(x) for x in fallback]
        else:
            values = [caster(fallback)]
        if not values:
            raise ValueError("grid must contain at least one value")
        return values
    values = [caster(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _parse_optional_float_grid(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    else:
        values = [float(raw)]
    return tuple(values)


def _fmt_float_tag(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _resolve_seed_and_val_size(args: argparse.Namespace, train_cfg: dict[str, Any], defaults: TrainConfig) -> tuple[int, float]:
    seed = int(choose(args.seed, train_cfg.get("seed"), defaults.seed))
    val_size = float(choose(args.val_size, train_cfg.get("val_size"), defaults.val_size))
    return seed, val_size


def _derive_auto_depth_grid(
    *,
    dataset: str,
    seed: int,
    val_size: float,
) -> tuple[list[int], dict[str, float]]:
    split = load_dataset_split(dataset, random_state=int(seed), val_size=float(val_size))
    y_train = np.asarray(split.y_train, dtype=np.float64).reshape(-1)
    if y_train.size == 0:
        return [5, 6], {"y_range": 0.0, "y_iqr": 0.0, "base_depth": 5.0}

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = max(y_max - y_min, 1e-12)
    q25, q75 = np.percentile(y_train, [25.0, 75.0])
    y_iqr = max(float(q75 - q25), 1e-12)

    target_bin_width = max(y_iqr / 6.0, y_range / 128.0, 1e-12)
    target_bins = int(np.clip(np.ceil(y_range / target_bin_width), 2 ** AUTO_DEPTH_MIN, 2 ** (AUTO_DEPTH_MAX - 2)))
    raw_base_depth = int(np.clip(np.round(np.log2(target_bins)), AUTO_DEPTH_MIN, AUTO_DEPTH_MAX - 1))
    shifted_base_depth = int(np.clip(raw_base_depth + 1, AUTO_DEPTH_MIN, AUTO_DEPTH_MAX - 1))

    grid = [int(np.clip(shifted_base_depth, AUTO_DEPTH_MIN, AUTO_DEPTH_MAX))]
    return grid, {
        "y_range": y_range,
        "y_iqr": y_iqr,
        "raw_base_depth": float(raw_base_depth),
        "base_depth": float(shifted_base_depth),
    }


def _build_train_config(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    dataset: str,
    *,
    run_id: str,
    epochs: int,
    depth: int,
    temperature: float,
) -> TrainConfig:
    defaults = TrainConfig()
    return TrainConfig(
        dataset=str(dataset),
        seed=int(choose(args.seed, train_cfg.get("seed"), defaults.seed)),
        val_size=float(choose(args.val_size, train_cfg.get("val_size"), defaults.val_size)),
        batch_size=int(choose(args.batch_size, train_cfg.get("batch_size"), defaults.batch_size)),
        epochs=int(epochs),
        lr=float(choose(args.lr, train_cfg.get("lr"), defaults.lr)),
        weight_decay=float(choose(args.weight_decay, train_cfg.get("weight_decay"), defaults.weight_decay)),
        depth=int(depth),
        binning_strategy=str(choose(args.binning_strategy, train_cfg.get("binning_strategy"), defaults.binning_strategy)),
        encoder_type=str(choose(args.encoder_type, train_cfg.get("encoder_type"), defaults.encoder_type)),
        d_model=int(choose(args.d_model, train_cfg.get("d_model"), defaults.d_model)),
        n_heads=int(choose(args.n_heads, train_cfg.get("n_heads"), defaults.n_heads)),
        n_layers=int(choose(args.n_layers, train_cfg.get("n_layers"), defaults.n_layers)),
        dropout=float(choose(args.dropout, train_cfg.get("dropout"), defaults.dropout)),
        confidence=float(choose(args.confidence, train_cfg.get("confidence"), defaults.confidence)),
        confidence_grid=_parse_optional_float_grid(
            choose(args.confidence_grid, train_cfg.get("confidence_grid"), eval_cfg.get("confidence_grid"), defaults.confidence_grid)
        ),
        temperature=float(temperature),
        interval_method=str(
            choose(args.interval_method, train_cfg.get("interval_method"), eval_cfg.get("interval_method"), defaults.interval_method)
        ),
        peak_merge_alpha=float(
            choose(
                args.peak_merge_alpha,
                train_cfg.get("peak_merge_alpha"),
                eval_cfg.get("peak_merge_alpha"),
                defaults.peak_merge_alpha,
            )
        ),
        mask_outside=float(
            choose(args.mask_outside, train_cfg.get("mask_outside"), eval_cfg.get("mask_outside"), defaults.mask_outside)
        ),
        beam_size=int(choose(args.beam_size, train_cfg.get("beam_size"), eval_cfg.get("beam_size"), defaults.beam_size)),
        leaf_prior_weight=float(
            choose(
                args.leaf_prior_weight,
                train_cfg.get("leaf_prior_weight"),
                eval_cfg.get("leaf_prior_weight"),
                defaults.leaf_prior_weight,
            )
        ),
        selection_warmup_epochs=int(
            choose(args.selection_warmup_epochs, train_cfg.get("selection_warmup_epochs"), defaults.selection_warmup_epochs)
        ),
        out_root=str(choose(args.out_root, train_cfg.get("out_root"), defaults.out_root)),
        run_id=str(run_id),
        device=choose(args.device, train_cfg.get("device"), defaults.device),
        num_workers=int(choose(args.num_workers, train_cfg.get("num_workers"), defaults.num_workers)),
        bin_step_02=float(choose(args.bin_step_02, train_cfg.get("bin_step_02"), defaults.bin_step_02)),
        bin_step_04=float(choose(args.bin_step_04, train_cfg.get("bin_step_04"), defaults.bin_step_04)),
        width_bins=_pick_width_bins(choose(args.width_bins, train_cfg.get("width_bins")), defaults.width_bins),
    )


def _run_test(
    *,
    run_dir: str,
    dataset: str,
    config_path: Optional[str],
    batch_size: Optional[int],
    confidence: Optional[float],
    temperature: Optional[float],
    interval_method: Optional[str],
    peak_merge_alpha: Optional[float],
    mask_outside: Optional[float],
    beam_size: Optional[int],
    leaf_prior_weight: Optional[float],
    device: Optional[str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "test.py"),
        "--ckpt",
        run_dir,
        "--dataset",
        dataset,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if confidence is not None:
        cmd.extend(["--confidence", str(confidence)])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if interval_method is not None:
        cmd.extend(["--interval-method", str(interval_method)])
    if peak_merge_alpha is not None:
        cmd.extend(["--peak-merge-alpha", str(peak_merge_alpha)])
    if mask_outside is not None:
        cmd.extend(["--mask-outside", str(mask_outside)])
    if beam_size is not None:
        cmd.extend(["--beam-size", str(beam_size)])
    if leaf_prior_weight is not None:
        cmd.extend(["--leaf-prior-weight", str(leaf_prior_weight)])
    if device is not None:
        cmd.extend(["--device", str(device)])

    subprocess.run(cmd, check=True, cwd=ROOT)
    return _load_json(_latest_metrics_file(run_dir))


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_better_candidate(candidate: dict[str, Any], best: Optional[dict[str, Any]], *, confidence: float) -> bool:
    if best is None:
        return True
    candidate_rank = constrained_interval_rank(candidate, target_confidence=float(confidence), tolerance_bins=1)
    best_rank = constrained_interval_rank(best, target_confidence=float(confidence), tolerance_bins=1)
    if candidate_rank != best_rank:
        return candidate_rank > best_rank
    candidate_rank = greedy_metric_rank(candidate, confidence=float(confidence), tolerance_bins=1)
    best_rank = greedy_metric_rank(best, confidence=float(confidence), tolerance_bins=1)
    return candidate_rank > best_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names; default is all complete local datasets")
    ap.add_argument("--list-datasets", action="store_true")
    ap.add_argument("--max-datasets", type=int, default=None)
    ap.add_argument("--batch-id", type=str, default=None)
    ap.add_argument("--summary-dir", type=str, default=None, help="where to save the merged table; default outputs/batch_runs/<batch_id>")
    ap.add_argument("--fail-fast", action="store_true")

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-size", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epochs-grid", type=str, default=None, help="comma-separated epoch candidates")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--depth-grid", type=str, default=None, help="comma-separated depth candidates")
    ap.add_argument("--binning-strategy", choices=["uniform", "quantile"], default=None)
    ap.add_argument("--encoder-type", type=str, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--confidence-grid", type=str, default=None, help="comma-separated inference confidences for val calibration")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--temperature-grid", type=str, default=None, help="comma-separated temperature candidates")
    ap.add_argument("--interval-method", choices=["cdf", "shortest_mass", "relaxed_mass", "peak_merge"], default=None)
    ap.add_argument("--peak-merge-alpha", type=float, default=None)
    ap.add_argument("--mask-outside", type=float, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--leaf-prior-weight", type=float, default=None)
    ap.add_argument("--selection-warmup-epochs", type=int, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--bin-step-02", type=float, default=None)
    ap.add_argument("--bin-step-04", type=float, default=None)
    ap.add_argument("--width-bins", type=str, default=None)

    ap.add_argument("--test-batch-size", type=int, default=None)
    args = ap.parse_args()

    available = _list_complete_local_datasets()
    if args.list_datasets:
        for name in available:
            print(name, flush=True)
        return

    if args.datasets:
        datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    else:
        datasets = _default_selected_datasets(available)
    if args.max_datasets is not None:
        datasets = datasets[: int(args.max_datasets)]
    if not datasets:
        raise SystemExit("no datasets selected")

    config = load_config(args.config)
    train_cfg = resolve_section(config, "train")
    eval_cfg = resolve_section(config, "eval")
    defaults = TrainConfig()

    epoch_grid = _parse_grid(
        args.epochs_grid,
        fallback=choose(args.epochs, DEFAULT_EPOCH_GRID),
        caster=int,
    )
    temperature_grid = _parse_grid(
        args.temperature_grid,
        fallback=choose(args.temperature, DEFAULT_TEMPERATURE_GRID),
        caster=float,
    )
    seed, val_size = _resolve_seed_and_val_size(args, train_cfg, defaults)
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.abspath(args.summary_dir or os.path.join("outputs", "batch_runs", batch_id))
    candidates_dir = os.path.join(summary_dir, "candidates")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    beam_size = int(choose(args.beam_size, train_cfg.get("beam_size"), eval_cfg.get("beam_size"), defaults.beam_size))

    print(f"[BATCH] datasets={datasets}", flush=True)
    print(f"[BATCH] summary_dir={summary_dir}", flush=True)
    if args.depth_grid is not None or args.depth is not None:
        manual_depth_grid = _parse_grid(
            args.depth_grid,
            fallback=int(args.depth),
            caster=int,
        )
        print(
            f"[BATCH] sweep epochs={epoch_grid} depth={manual_depth_grid} temperature={temperature_grid} beam_size={beam_size}",
            flush=True,
        )
    else:
        manual_depth_grid = None
        print(
            f"[BATCH] sweep epochs={epoch_grid} depth=auto-per-dataset temperature={temperature_grid} beam_size={beam_size}",
            flush=True,
        )

    for idx, dataset in enumerate(datasets, start=1):
        print(f"[BATCH] ({idx}/{len(datasets)}) tune+test {dataset}", flush=True)
        if manual_depth_grid is None:
            depth_grid, depth_meta = _derive_auto_depth_grid(dataset=dataset, seed=seed, val_size=val_size)
            print(
                f"[BATCH]   auto depth grid for {dataset}: {depth_grid} "
                f"(base={int(depth_meta['base_depth'])}, y_range={depth_meta['y_range']:.6g}, y_iqr={depth_meta['y_iqr']:.6g})",
                flush=True,
            )
        else:
            depth_grid = manual_depth_grid
            depth_meta = {}

        combos = list(product(epoch_grid, depth_grid, temperature_grid))
        row: dict[str, Any] = {
            "dataset": dataset,
            "batch_id": batch_id,
            "status": "ok",
            "n_candidates": len(combos),
            "depth_grid_used": json.dumps(depth_grid),
            "auto_depth_base": depth_meta.get("base_depth"),
            "auto_depth_y_range": depth_meta.get("y_range"),
            "auto_depth_y_iqr": depth_meta.get("y_iqr"),
        }
        try:
            candidate_rows: list[dict[str, Any]] = []
            best_candidate_row: Optional[dict[str, Any]] = None
            best_val_metrics: Optional[dict[str, Any]] = None

            for cand_idx, (epochs, depth, temperature) in enumerate(combos, start=1):
                run_id = f"{batch_id}_{dataset}_ep{epochs}_d{depth}_t{_fmt_float_tag(float(temperature))}"
                print(
                    f"[BATCH]   candidate ({cand_idx}/{len(combos)}) "
                    f"epochs={epochs} depth={depth} temperature={temperature}"
                , flush=True)
                train_config = _build_train_config(
                    args,
                    train_cfg,
                    eval_cfg,
                    dataset,
                    run_id=run_id,
                    epochs=int(epochs),
                    depth=int(depth),
                    temperature=float(temperature),
                )
                run_dir = train_tabseq_model(train_config)
                train_summary = _load_json(os.path.join(run_dir, "train_summary.json"))
                val_metrics = _load_json(os.path.join(run_dir, "metrics_val_beam.json"))

                candidate_row = {
                    "dataset": dataset,
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "checkpoint_path": os.path.join(run_dir, "checkpoint.pt"),
                    "epochs": int(epochs),
                    "depth": int(depth),
                    "temperature": float(temperature),
                    "binning_strategy": str(train_config.binning_strategy),
                    "confidence": float(train_config.confidence),
                    "interval_method": str(train_config.interval_method),
                    "peak_merge_alpha": float(train_config.peak_merge_alpha),
                    "mask_outside": float(train_config.mask_outside),
                    "beam_size": int(train_config.beam_size),
                    "leaf_prior_weight": float(train_config.leaf_prior_weight),
                    "target_confidence": float(train_config.confidence),
                    "calibrated_confidence": float(val_metrics.get("calibrated_confidence", train_config.confidence)),
                }
                candidate_row.update(_flatten("train_summary.", train_summary))
                candidate_row.update(_flatten("val.", val_metrics))
                candidate_rows.append(candidate_row)

                if _is_better_candidate(val_metrics, best_val_metrics, confidence=float(train_config.confidence)):
                    best_candidate_row = candidate_row
                    best_val_metrics = val_metrics

            if best_candidate_row is None or best_val_metrics is None:
                raise RuntimeError(f"no successful candidate for dataset={dataset}")

            candidate_csv = os.path.join(candidates_dir, f"{dataset}.csv")
            candidate_json = os.path.join(candidates_dir, f"{dataset}.json")
            _write_csv(candidate_csv, candidate_rows)
            write_json(
                candidate_json,
                {
                    "dataset": dataset,
                    "batch_id": batch_id,
                    "rows": candidate_rows,
                },
            )

            test_metrics = _run_test(
                run_dir=str(best_candidate_row["run_dir"]),
                dataset=dataset,
                config_path=args.config,
                batch_size=args.test_batch_size,
                confidence=float(best_candidate_row.get("calibrated_confidence", best_candidate_row["confidence"])),
                temperature=float(best_candidate_row["temperature"]),
                interval_method=str(best_candidate_row["interval_method"]),
                peak_merge_alpha=float(best_candidate_row["peak_merge_alpha"]),
                mask_outside=float(best_candidate_row["mask_outside"]),
                beam_size=int(best_candidate_row["beam_size"]),
                leaf_prior_weight=float(best_candidate_row["leaf_prior_weight"]),
                device=args.device,
            )

            row.update(
                {
                    "best_run_id": best_candidate_row["run_id"],
                    "best_run_dir": best_candidate_row["run_dir"],
                    "best_checkpoint_path": best_candidate_row["checkpoint_path"],
                    "best_epochs": best_candidate_row["epochs"],
                    "best_depth": best_candidate_row["depth"],
                    "best_temperature": best_candidate_row["temperature"],
                    "best_binning_strategy": best_candidate_row["binning_strategy"],
                    "best_confidence": best_candidate_row["confidence"],
                    "best_calibrated_confidence": best_candidate_row.get("calibrated_confidence", best_candidate_row["confidence"]),
                    "best_interval_method": best_candidate_row["interval_method"],
                    "best_peak_merge_alpha": best_candidate_row["peak_merge_alpha"],
                    "best_mask_outside": best_candidate_row["mask_outside"],
                    "best_beam_size": best_candidate_row["beam_size"],
                    "best_leaf_prior_weight": best_candidate_row["leaf_prior_weight"],
                    "candidate_csv": candidate_csv,
                    "candidate_json": candidate_json,
                }
            )
            row.update({k: v for k, v in best_candidate_row.items() if k.startswith("train_summary.")})
            row.update({k: v for k, v in best_candidate_row.items() if k.startswith("val.")})
            row.update(_flatten("test.", test_metrics))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            failures.append({"dataset": dataset, "error": str(exc)})
            if args.fail_fast:
                rows.append(row)
                break

        rows.append(row)
        csv_path = os.path.join(summary_dir, "summary.csv")
        json_path = os.path.join(summary_dir, "summary.json")
        _write_csv(csv_path, rows)
        write_json(
            json_path,
            {
                "batch_id": batch_id,
                "datasets": datasets,
                "epoch_grid": epoch_grid,
                "depth_grid_mode": "manual" if manual_depth_grid is not None else "auto_per_dataset",
                "manual_depth_grid": manual_depth_grid,
                "temperature_grid": temperature_grid,
                "rows": rows,
                "failures": failures,
            },
        )

    csv_path = os.path.join(summary_dir, "summary.csv")
    json_path = os.path.join(summary_dir, "summary.json")
    _write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "batch_id": batch_id,
            "datasets": datasets,
            "epoch_grid": epoch_grid,
            "depth_grid_mode": "manual" if manual_depth_grid is not None else "auto_per_dataset",
            "manual_depth_grid": manual_depth_grid,
            "temperature_grid": temperature_grid,
            "rows": rows,
            "failures": failures,
        },
    )

    print(f"[BATCH] saved csv: {csv_path}", flush=True)
    print(f"[BATCH] saved json: {json_path}", flush=True)
    if failures:
        print(f"[BATCH] failures={len(failures)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
