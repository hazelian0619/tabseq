#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "openml_regression"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_data_id: Optional[int] = None


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


def _parse_openml_ids(raw: str) -> list[int]:
    ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not ids:
        raise ValueError("openml id list must not be empty")
    return ids


def _resolve_specs(datasets_arg: Optional[str], openml_ids_arg: Optional[str]) -> list[DatasetSpec]:
    if openml_ids_arg:
        return [
            DatasetSpec(
                name=str(data_id),
                openml_data_id=int(data_id),
            )
            for data_id in _parse_openml_ids(openml_ids_arg)
        ]
    if datasets_arg:
        names = [x.strip() for x in str(datasets_arg).split(",") if x.strip()]
        return [DatasetSpec(name=name) for name in names]
    raise ValueError("please provide --openml-ids or --datasets")


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
    safe_name = name.replace("/", "__").replace("\\", "__")
    return root / safe_name


def _download_one(spec: DatasetSpec, root: Path, overwrite: bool) -> dict[str, Any]:
    if spec.openml_data_id is not None:
        bunch = fetch_openml(
            data_id=int(spec.openml_data_id),
            as_frame=True,
            data_home=str(root / "_sklearn_openml_cache"),
        )
    else:
        bunch = fetch_openml(
            name=spec.name,
            as_frame=True,
            data_home=str(root / "_sklearn_openml_cache"),
        )
    X = _to_feature_frame(bunch.data, getattr(bunch, "feature_names", None))
    details = getattr(bunch, "details", {})
    if not isinstance(details, dict):
        details = {}
    source_dataset_name = str(details.get("name") or spec.name)
    local_dataset_name = source_dataset_name if spec.openml_data_id is not None else spec.name
    if bunch.target is None:
        target_name = OPENML_TARGET_FALLBACKS.get(source_dataset_name) or OPENML_TARGET_FALLBACKS.get(spec.name)
        if target_name is None or target_name not in X.columns:
            raise ValueError(
                f"OpenML dataset {source_dataset_name!r} has no parsed target; "
                "please add an explicit fallback target column name"
            )
        y = _to_numeric_target(X[target_name])
        X = X.drop(columns=[target_name])
    else:
        y = _to_numeric_target(bunch.target)
        target_name = _resolve_target_name(bunch)
    data_id = details.get("id")
    source_url = None if data_id is None else f"https://api.openml.org/d/{int(data_id)}"

    dataset_dir = _dataset_dir(root, local_dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table_path = dataset_dir / "table.csv.gz"
    metadata_path = dataset_dir / "metadata.json"

    if table_path.exists() and metadata_path.exists() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["status"] = "skipped_existing"
        return metadata

    if target_name in X.columns:
        target_name = f"{target_name}__target"

    frame = X.copy()
    frame[target_name] = y.to_numpy(copy=False)
    frame.to_csv(table_path, index=False, compression="gzip")

    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns.tolist() if c not in num_cols]

    metadata = {
        "name": local_dataset_name,
        "source_dataset_name": source_dataset_name,
        "source": "openml",
        "status": "downloaded",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "openml_data_id": int(data_id) if data_id is not None else None,
        "requested_openml_data_id": int(spec.openml_data_id) if spec.openml_data_id is not None else None,
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
        id_suffix = f" openml_id={spec.openml_data_id}" if spec.openml_data_id is not None else ""
        print(f"{spec.name:24s} source=openml{id_suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download tabular regression datasets and save them under the local data directory. "
            "For OpenML, prefer specifying --openml-ids."
        )
    )
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="comma-separated OpenML dataset names",
    )
    ap.add_argument(
        "--openml-ids",
        type=str,
        default=None,
        help="comma-separated OpenML data ids; datasets will be saved under their actual dataset names",
    )
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="network socket timeout in seconds for each dataset download; <=0 disables it",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.datasets and args.openml_ids:
        raise SystemExit("use either --datasets or --openml-ids, not both")

    specs = _resolve_specs(args.datasets, args.openml_ids)

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
        f"(timeout={timeout_desc})",
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
                        f"[DONE] {metadata['name']} success rows={metadata['n_rows']} "
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
        "datasets": [
            {
                "name": spec.name,
                "source": "openml",
                "openml_data_id": spec.openml_data_id,
            }
            for spec in specs
        ],
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
