#!/usr/bin/env python3
"""
Download OpenML regression datasets and store as numpy arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


OPENML_CTR23 = {
    "diamonds": 44979,
    "fifa": 45012,
    "superconductivity": 44964,
    "kin8nm": 44980,
    "naval_propulsion": 44969,
}


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _split_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def _encode_categorical(series: pd.Series) -> Tuple[np.ndarray, int]:
    cat = series.astype("category")
    if cat.isna().any():
        cat = cat.cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
    codes = cat.cat.codes.to_numpy(dtype=np.int64)
    cardinality = int(len(cat.cat.categories))
    return codes, cardinality


def _prepare_arrays(df: pd.DataFrame, target: pd.Series) -> Dict:
    num_cols, cat_cols = _split_features(df)

    if num_cols:
        x_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
        x_num = x_num.astype(np.float32)
        if x_num.isna().any().any():
            x_num = x_num.fillna(x_num.mean(numeric_only=True))
        x_num_arr = x_num.to_numpy(dtype=np.float32)
    else:
        x_num_arr = np.zeros((len(df), 0), dtype=np.float32)

    if cat_cols:
        cat_arrays = []
        cat_cardinalities = []
        for col in cat_cols:
            codes, cardinality = _encode_categorical(df[col])
            cat_arrays.append(codes)
            cat_cardinalities.append(cardinality)
        x_cat_arr = np.stack(cat_arrays, axis=1).astype(np.int64)
    else:
        x_cat_arr = np.zeros((len(df), 0), dtype=np.int64)
        cat_cardinalities = []

    y = pd.to_numeric(target, errors="coerce").astype(np.float32)
    if y.isna().any():
        raise ValueError("target contains non-numeric values after coercion")
    y_arr = y.to_numpy(dtype=np.float32)

    meta = {
        "n_samples": int(len(df)),
        "n_num_features": int(x_num_arr.shape[1]),
        "n_cat_features": int(x_cat_arr.shape[1]),
        "cat_cardinalities": cat_cardinalities,
        "num_feature_names": num_cols,
        "cat_feature_names": cat_cols,
    }
    return {
        "X_num": x_num_arr,
        "X_cat": x_cat_arr,
        "y": y_arr,
        "meta": meta,
    }


def _resolve_dataset(spec: str) -> Tuple[str, int]:
    if spec in OPENML_CTR23:
        return spec, int(OPENML_CTR23[spec])
    if spec.isdigit():
        return f"openml_{spec}", int(spec)
    raise ValueError(f"unknown dataset spec: {spec}")


def download_dataset(name: str, data_id: int, out_root: str) -> str:
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
        except Exception:
            certifi = None
        if certifi is not None:
            os.environ["SSL_CERT_FILE"] = certifi.where()
    data_home = os.path.join(out_root, "_sklearn_cache")
    dataset = fetch_openml(data_id=data_id, as_frame=True, data_home=data_home, parser="auto")
    df = dataset.data
    target = dataset.target
    pack = _prepare_arrays(df, target)

    out_dir = os.path.join(out_root, _slugify(name))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_num.npy"), pack["X_num"])
    np.save(os.path.join(out_dir, "X_cat.npy"), pack["X_cat"])
    np.save(os.path.join(out_dir, "y.npy"), pack["y"])

    meta = {
        "openml_id": int(data_id),
        "openml_name": dataset.details.get("name", name),
        "target_name": dataset.details.get("default_target_attribute"),
    }
    meta.update(pack["meta"])
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True, help="name or OpenML data_id")
    parser.add_argument("--out-dir", default="data/openml")
    args = parser.parse_args()

    for spec in args.datasets:
        name, data_id = _resolve_dataset(spec)
        out_dir = download_dataset(name, data_id, args.out_dir)
        print(f"downloaded: {name} (id={data_id}) -> {out_dir}")


if __name__ == "__main__":
    main()
