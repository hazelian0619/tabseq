from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_DATA_ROOTS = (
    REPO_ROOT / "data" / "openml_regression",
    REPO_ROOT / "data",
)


def _local_dataset_dir_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__")


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_cat_train: Optional[np.ndarray] = None
    X_cat_val: Optional[np.ndarray] = None
    num_feature_names: Sequence[str] = ()
    cat_feature_names: Sequence[str] = ()
    cat_cardinalities: Sequence[int] = ()


def _coerce_target(y: object) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"expected a single target column, got shape={y.shape}")
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        values = pd.to_numeric(y, errors="coerce").to_numpy()
    else:
        values = np.asarray(y)
        if values.ndim > 1:
            if values.shape[1] != 1:
                raise ValueError(f"expected 1D target, got shape={values.shape}")
            values = values[:, 0]
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if np.isnan(values).any():
        raise ValueError("target contains NaN or non-numeric values")
    return values


def _candidate_local_dataset_dirs(dataset: str) -> list[Path]:
    name = str(dataset)
    normalized_name = _local_dataset_dir_name(name)
    candidates = []
    for root in DEFAULT_LOCAL_DATA_ROOTS:
        candidates.append(root / name)
        if normalized_name != name:
            candidates.append(root / normalized_name)
    return candidates


def _load_local_frame(dataset: str) -> Optional[tuple[pd.DataFrame, np.ndarray]]:
    for dataset_dir in _candidate_local_dataset_dirs(dataset):
        table_path = dataset_dir / "table.csv.gz"
        metadata_path = dataset_dir / "metadata.json"
        if not table_path.is_file() or not metadata_path.is_file():
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        target_name = metadata.get("target_name")
        if not isinstance(target_name, str) or not target_name:
            raise ValueError(f"metadata.json for {dataset!r} must contain a non-empty target_name")

        frame = pd.read_csv(table_path)
        if target_name not in frame.columns:
            raise ValueError(f"target_name={target_name!r} not found in local dataset columns for {dataset!r}")

        y = _coerce_target(frame.pop(target_name))
        return frame, y

    return None


def _load_raw_frame(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    name = str(dataset)
    local = _load_local_frame(name)
    if local is not None:
        return local

    if name.startswith("openml_"):
        data_id = int(name.split("_", 1)[1])
        bunch = fetch_openml(data_id=data_id, as_frame=True)
    else:
        bunch = fetch_openml(name=name, as_frame=True)

    X = bunch.data.copy()
    y = _coerce_target(bunch.target)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X, y


def _split_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in frame.columns.tolist() if c not in num_cols]
    return frame[num_cols].copy(), frame[cat_cols].copy()


def _fill_missing_numeric(train_arr: np.ndarray, val_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_arr.shape[1] == 0:
        return train_arr.astype(np.float32, copy=False), val_arr.astype(np.float32, copy=False)

    medians = np.nanmedian(train_arr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    train_arr = np.where(np.isnan(train_arr), medians, train_arr)
    val_arr = np.where(np.isnan(val_arr), medians, val_arr)
    return train_arr.astype(np.float32, copy=False), val_arr.astype(np.float32, copy=False)


def load_dataset_split(
    dataset: str,
    *,
    random_state: int = 0,
    val_size: float = 0.2,
) -> DatasetSplit:
    if not (0.0 < float(val_size) < 1.0):
        raise ValueError("val_size must be in (0, 1)")

    frame, y = _load_raw_frame(dataset)
    X_num_df, X_cat_df = _split_columns(frame)

    indices = np.arange(len(y))
    idx_train, idx_val = train_test_split(indices, test_size=float(val_size), random_state=int(random_state))

    X_num_train = X_num_df.iloc[idx_train].to_numpy(dtype=np.float32, copy=True)
    X_num_val = X_num_df.iloc[idx_val].to_numpy(dtype=np.float32, copy=True)
    X_num_train, X_num_val = _fill_missing_numeric(X_num_train, X_num_val)

    if X_num_train.shape[1] > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_num_train).astype(np.float32)
        X_val = scaler.transform(X_num_val).astype(np.float32)
    else:
        X_train = np.zeros((len(idx_train), 0), dtype=np.float32)
        X_val = np.zeros((len(idx_val), 0), dtype=np.float32)

    X_cat_train: Optional[np.ndarray] = None
    X_cat_val: Optional[np.ndarray] = None
    cat_cardinalities: tuple[int, ...] = ()

    if X_cat_df.shape[1] > 0:
        train_cat = X_cat_df.iloc[idx_train].fillna("__nan__").astype(str)
        val_cat = X_cat_df.iloc[idx_val].fillna("__nan__").astype(str)
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        )
        X_cat_train = encoder.fit_transform(train_cat).astype(np.int64) + 1
        X_cat_val = encoder.transform(val_cat).astype(np.int64) + 1
        cat_cardinalities = tuple(len(categories) + 1 for categories in encoder.categories_)

    y_train = y[idx_train].astype(np.float32, copy=False)
    y_val = y[idx_val].astype(np.float32, copy=False)

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_cat_train=X_cat_train,
        X_cat_val=X_cat_val,
        num_feature_names=tuple(X_num_df.columns.tolist()),
        cat_feature_names=tuple(X_cat_df.columns.tolist()),
        cat_cardinalities=cat_cardinalities,
    )
