#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "openml_regression"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    preset: str
    feature_type: str
    note: str


CURATED_REGRESSION_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="diabetes",
        source="sklearn_builtin",
        preset="core",
        feature_type="numeric",
        note="我们之前一直在用的基准数据集，适合先做快速训练和调试。",
    ),
    DatasetSpec(
        name="california_housing",
        source="sklearn_builtin",
        preset="core",
        feature_type="numeric",
        note="我们之前一直在用的房价回归数据集，适合做主实验。",
    ),
    DatasetSpec(
        name="abalone",
        source="openml",
        preset="core",
        feature_type="mixed",
        note="小中型，经典 mixed-type 回归，适合先做 smoke/perf 对比。",
    ),
    DatasetSpec(
        name="elevators",
        source="openml",
        preset="core",
        feature_type="numeric",
        note="中型，纯数值回归，适合看 encoder/decoder 主干是否稳定。",
    ),
    DatasetSpec(
        name="MiamiHousing2016",
        source="openml",
        preset="core",
        feature_type="numeric",
        note="中型房价回归，数值特征为主，和我们任务比较贴近。",
    ),
    DatasetSpec(
        name="house_sales",
        source="openml",
        preset="core",
        feature_type="mixed",
        note="中型房价回归，含类别和数值特征，适合测 mixed 表格能力。",
    ),
    DatasetSpec(
        name="Bike_Sharing_Demand",
        source="openml",
        preset="core",
        feature_type="mixed",
        note="中型 demand 回归，表格结构标准，适合做主实验。",
    ),
    DatasetSpec(
        name="diamonds",
        source="openml",
        preset="core",
        feature_type="mixed",
        note="中大型 mixed-type 回归，零缺失，适合做稳定 benchmark。",
    ),
    DatasetSpec(
        name="Brazilian_houses",
        source="openml",
        preset="extended",
        feature_type="mixed",
        note="中型房价回归，mixed-type，和 housing 场景接近。",
    ),
    DatasetSpec(
        name="house_prices_nominal",
        source="openml",
        preset="extended",
        feature_type="mixed",
        note="较多类别特征，适合看 categorical tokenization 表现。",
    ),
    DatasetSpec(
        name="sulfur",
        source="openml",
        preset="extended",
        feature_type="numeric",
        note="中型纯数值回归，规模适中，适合补充数值场景。",
    ),
    DatasetSpec(
        name="medical_charges",
        source="openml",
        preset="extended",
        feature_type="numeric",
        note="较大规模纯数值回归，适合看样本量变大后的表现。",
    ),
    DatasetSpec(
        name="superconduct",
        source="openml",
        preset="extended",
        feature_type="numeric",
        note="中高维纯数值回归，适合看特征维度上来后 encoder 的表现。",
    ),
    DatasetSpec(
        name="wine_quality",
        source="openml",
        preset="extended",
        feature_type="numeric",
        note="小而干净的回归基准，适合做稳定 sanity benchmark。",
    ),
)


OPENML_TARGET_FALLBACKS: dict[str, str] = {
    "house_sales": "price",
}


class _SimpleProgressBar:
    def __init__(self, total: int) -> None:
        self.total = max(int(total), 1)
        self.current = 0
        self.description = ""
        self._render()

    def set_description(self, description: str) -> None:
        self.description = str(description)
        self._render()

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + int(n))
        self._render()

    def write(self, message: str) -> None:
        sys.stdout.write("\n" + str(message) + "\n")
        self._render()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self) -> None:
        width = 24
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        line = f"\r[{bar}] {self.current}/{self.total}"
        if self.description:
            line += f" | {self.description}"
        sys.stdout.write(line)
        sys.stdout.flush()


def _make_progress(total: int):
    if tqdm is not None:
        return tqdm(total=total, unit="dataset", dynamic_ncols=True)
    return _SimpleProgressBar(total=total)


def _progress_write(progress: Any, message: str) -> None:
    if tqdm is not None and isinstance(progress, tqdm):
        progress.write(message)
    else:
        progress.write(message)


def _set_socket_timeout(timeout_seconds: Optional[float]) -> Optional[float]:
    previous = socket.getdefaulttimeout()
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        socket.setdefaulttimeout(float(timeout_seconds))
    return previous


def _restore_socket_timeout(previous: Optional[float]) -> None:
    socket.setdefaulttimeout(previous)


def _resolve_specs(preset: str, datasets_arg: Optional[str]) -> list[DatasetSpec]:
    by_name = {spec.name: spec for spec in CURATED_REGRESSION_DATASETS}
    if datasets_arg:
        names = [x.strip() for x in str(datasets_arg).split(",") if x.strip()]
        specs = []
        for name in names:
            specs.append(
                by_name.get(
                    name,
                    DatasetSpec(
                        name=name,
                        source="openml",
                        preset="custom",
                        feature_type="unknown",
                        note="用户自定义数据集，脚本只负责下载和保存。",
                    ),
                )
            )
        return specs

    if preset == "core":
        return [spec for spec in CURATED_REGRESSION_DATASETS if spec.preset == "core"]
    if preset == "extended":
        return list(CURATED_REGRESSION_DATASETS)
    raise ValueError(f"unknown preset={preset!r}")


def _filter_specs_by_source(specs: Sequence[DatasetSpec], source: str) -> list[DatasetSpec]:
    if source == "all":
        return list(specs)
    if source == "openml":
        return [spec for spec in specs if spec.source == "openml"]
    if source == "sklearn_builtin":
        return [spec for spec in specs if spec.source == "sklearn_builtin"]
    raise ValueError(f"unknown source={source!r}")


def _resolve_target_name(bunch: Any) -> str:
    target_names = getattr(bunch, "target_names", None)
    if isinstance(target_names, str) and target_names:
        return target_names
    if isinstance(target_names, Sequence) and not isinstance(target_names, (str, bytes)) and len(target_names) > 0:
        first = target_names[0]
        if isinstance(first, str) and first:
            return first

    details = getattr(bunch, "details", None)
    if isinstance(details, dict):
        value = details.get("default_target_attribute")
        if isinstance(value, str) and value:
            return value

    return "target"


def _to_feature_frame(data: Any, feature_names: Optional[Sequence[str]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "toarray"):
        data = data.toarray()

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D tabular features, got shape={arr.shape}")

    names = list(feature_names) if feature_names is not None else None
    if names is not None and len(names) == arr.shape[1]:
        return pd.DataFrame(arr, columns=names)
    return pd.DataFrame(arr)


def _to_numeric_target(target: Any) -> pd.Series:
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(f"expected a single regression target column, got shape={target.shape}")
        target = target.iloc[:, 0]

    if isinstance(target, pd.Series):
        series = target.copy()
    else:
        arr = np.asarray(target)
        if arr.ndim > 1:
            if arr.shape[1] != 1:
                raise ValueError(f"expected a single regression target, got shape={arr.shape}")
            arr = arr[:, 0]
        series = pd.Series(arr)

    series = pd.to_numeric(series, errors="raise")
    if series.isna().any():
        raise ValueError("target contains NaN after numeric coercion")
    return series.astype(np.float32)


def _dataset_dir(root: Path, name: str) -> Path:
    safe_name = name.replace("/", "__").replace(" ", "_")
    return root / safe_name


def _load_builtin_dataset(name: str, cache_root: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if name == "diabetes":
        bunch = load_diabetes(as_frame=True)
    elif name == "california_housing":
        bunch = fetch_california_housing(as_frame=True, data_home=str(cache_root))
    else:
        raise ValueError(f"unknown sklearn builtin dataset: {name}")

    X = bunch.data.copy()
    y = _to_numeric_target(bunch.target)
    details = {
        "id": None,
        "source_url": None,
        "target_name": getattr(bunch, "target_names", None),
    }
    return X, y, details


def _download_one(spec: DatasetSpec, root: Path, overwrite: bool) -> dict[str, Any]:
    dataset_dir = _dataset_dir(root, spec.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table_path = dataset_dir / "table.csv.gz"
    metadata_path = dataset_dir / "metadata.json"

    if table_path.exists() and metadata_path.exists() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["status"] = "skipped_existing"
        return metadata

    if spec.source == "openml":
        bunch = fetch_openml(
            name=spec.name,
            as_frame=True,
            data_home=str(root / "_sklearn_openml_cache"),
        )
        X = _to_feature_frame(bunch.data, getattr(bunch, "feature_names", None))
        if bunch.target is None:
            target_name = OPENML_TARGET_FALLBACKS.get(spec.name)
            if target_name is None or target_name not in X.columns:
                raise ValueError(
                    f"OpenML dataset {spec.name!r} has no parsed target; "
                    "please add an explicit fallback target column name"
                )
            y = _to_numeric_target(X[target_name])
            X = X.drop(columns=[target_name])
        else:
            y = _to_numeric_target(bunch.target)
            target_name = _resolve_target_name(bunch)
        details = getattr(bunch, "details", {})
        if not isinstance(details, dict):
            details = {}
        data_id = details.get("id")
        source_url = None if data_id is None else f"https://api.openml.org/d/{int(data_id)}"
    elif spec.source == "sklearn_builtin":
        X, y, details = _load_builtin_dataset(spec.name, root / "_sklearn_builtin_cache")
        target_name = details.get("target_name") or "target"
        if isinstance(target_name, Sequence) and not isinstance(target_name, (str, bytes)):
            target_name = target_name[0] if target_name else "target"
        data_id = details.get("id")
        source_url = details.get("source_url")
    else:
        raise ValueError(f"unknown source={spec.source!r}")

    if target_name in X.columns:
        target_name = f"{target_name}__target"

    frame = X.copy()
    frame[target_name] = y.to_numpy(copy=False)
    frame.to_csv(table_path, index=False, compression="gzip")

    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns.tolist() if c not in num_cols]

    metadata = {
        "name": spec.name,
        "source": spec.source,
        "preset": spec.preset,
        "feature_type": spec.feature_type,
        "note": spec.note,
        "status": "downloaded",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "openml_data_id": int(data_id) if data_id is not None else None,
        "source_url": source_url,
        "target_name": target_name,
        "n_rows": int(frame.shape[0]),
        "n_features": int(X.shape[1]),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "n_missing_values": int(X.isna().sum().sum()),
        "table_path": str(table_path),
        "numeric_feature_names": num_cols,
        "categorical_feature_names": cat_cols,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def _print_specs(specs: Sequence[DatasetSpec]) -> None:
    for spec in specs:
        print(
            f"{spec.name:24s} source={spec.source:15s} "
            f"preset={spec.preset:8s} type={spec.feature_type:7s} {spec.note}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download a curated set of OpenML tabular regression datasets and "
            "save them under the local data directory."
        )
    )
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--preset", choices=["core", "extended"], default="core")
    ap.add_argument("--source", choices=["openml", "sklearn_builtin", "all"], default="openml")
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="comma-separated dataset names; if set, overrides --preset",
    )
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="network socket timeout in seconds for each dataset download; <=0 disables it",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="list curated datasets and exit")
    args = ap.parse_args()

    specs = _filter_specs_by_source(_resolve_specs(args.preset, args.datasets), args.source)
    if args.list:
        _print_specs(specs)
        return

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    ok_count = 0
    skipped_count = 0
    fail_count = 0

    timeout_desc = "disabled" if float(args.timeout_seconds) <= 0 else f"{float(args.timeout_seconds):g}s"
    print(
        f"[RUN] start downloading {len(specs)} dataset(s) to {root} "
        f"(source={args.source}, preset={args.preset}, timeout={timeout_desc})",
        flush=True,
    )

    progress = _make_progress(total=len(specs))
    try:
        for idx, spec in enumerate(specs, start=1):
            progress.set_description(f"{idx}/{len(specs)} {spec.name}")
            _progress_write(progress, f"[START] ({idx}/{len(specs)}) downloading {spec.name}")
            try:
                previous_timeout = _set_socket_timeout(
                    None if float(args.timeout_seconds) <= 0 else float(args.timeout_seconds)
                )
                try:
                    metadata = _download_one(spec, root=root, overwrite=bool(args.overwrite))
                finally:
                    _restore_socket_timeout(previous_timeout)
                manifest.append(metadata)
                if metadata.get("status") == "skipped_existing":
                    skipped_count += 1
                    _progress_write(
                        progress,
                        f"[DONE] {spec.name} skipped_existing path={metadata['table_path']}",
                    )
                else:
                    ok_count += 1
                    _progress_write(
                        progress,
                        f"[DONE] {spec.name} success rows={metadata['n_rows']} "
                        f"features={metadata['n_features']} target={metadata['target_name']}",
                    )
            except Exception as exc:
                fail_count += 1
                failures.append({"name": spec.name, "error": str(exc)})
                _progress_write(progress, f"[DONE] {spec.name} failed: {exc}")
            finally:
                progress.update(1)
    finally:
        progress.close()

    manifest_payload = {
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": int(len(specs)),
            "ok": int(ok_count),
            "skipped": int(skipped_count),
            "failed": int(fail_count),
        },
        "selection_rule": (
            "tabular regression only; include our two existing sklearn baselines plus "
            "OpenML datasets with moderate size and a mix of numeric-only / mixed-type tables"
        ),
        "datasets": [asdict(spec) for spec in specs],
        "downloads": manifest,
        "failures": failures,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    print(
        f"[RUN] finished total={len(specs)} ok={ok_count} skipped={skipped_count} failed={fail_count}\n"
        f"saved manifest: {manifest_path}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
