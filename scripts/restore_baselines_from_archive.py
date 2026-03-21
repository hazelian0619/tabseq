#!/usr/bin/env python3
"""
Restore baseline artifacts from an archived cleanup folder back into outputs/baselines,
using the *canonical* layout:

  outputs/baselines/<dataset>/<model>/run_<run_id>/

and creating backward-compatible symlink aliases:

  outputs/baselines/baseline_<model>_<run_id>  ->  <dataset>/<model>/run_<run_id>

This fixes "broken doc references" without re-training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple


_BASELINE_DIR_RE = re.compile(r"^baseline_(?P<model>[a-z0-9_]+)_(?P<run_id>\d{8}_\d{6})$")


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _infer_dataset(baseline_dir: str, cfg: Dict[str, Any], *, default_dataset: str) -> str:
    # Prefer explicit fields (newer baselines).
    for key in ("dataset", "data_split"):
        if cfg.get(key):
            return str(cfg[key])
    # Older torch baselines didn't record dataset; those were California-only at that time.
    return default_dataset


def _canonical_dst(out_root: str, *, dataset: str, model: str, run_id: str) -> str:
    return os.path.join(out_root, dataset, model, f"run_{run_id}")


def _legacy_alias(out_root: str, *, model: str, run_id: str) -> str:
    return os.path.join(out_root, f"baseline_{model}_{run_id}")


def _rel_symlink(target: str, link_path: str) -> None:
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    rel = os.path.relpath(target, os.path.dirname(link_path))
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(rel, link_path)


def _copytree_overwrite(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _collect_archive_dirs(archive_root: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for name in os.listdir(archive_root):
        m = _BASELINE_DIR_RE.match(name)
        if not m:
            continue
        path = os.path.join(archive_root, name)
        if not os.path.isdir(path):
            continue
        out.append((path, m.group("model"), m.group("run_id")))
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive-root",
        default="outputs/_archive/20260210_cleanup/baselines",
        help="archive baselines dir",
    )
    ap.add_argument("--out-root", default="outputs/baselines")
    ap.add_argument(
        "--default-dataset",
        default="california_housing",
        help="dataset name for legacy baselines without dataset in config.json",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing restored dirs")
    args = ap.parse_args()

    archive_root = str(args.archive_root)
    out_root = str(args.out_root)

    if not os.path.isdir(archive_root):
        raise FileNotFoundError(f"archive_root not found: {archive_root}")

    restored: List[Dict[str, Any]] = []
    for src_dir, model, run_id in _collect_archive_dirs(archive_root):
        cfg = _read_json(os.path.join(src_dir, "config.json"))
        dataset = _infer_dataset(src_dir, cfg, default_dataset=str(args.default_dataset))
        dst_dir = _canonical_dst(out_root, dataset=dataset, model=model, run_id=run_id)
        alias = _legacy_alias(out_root, model=model, run_id=run_id)

        if os.path.exists(dst_dir) and not bool(args.force):
            print("skip (exists):", dst_dir)
        else:
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            _copytree_overwrite(src_dir, dst_dir)
            # Provenance marker.
            with open(os.path.join(dst_dir, "RESTORED_FROM_ARCHIVE.txt"), "w", encoding="utf-8") as f:
                f.write(src_dir + "\n")

        _rel_symlink(dst_dir, alias)
        restored.append(
            {
                "model": model,
                "run_id": run_id,
                "dataset": dataset,
                "src": src_dir,
                "dst": dst_dir,
                "alias": alias,
            }
        )
        print("restored:", alias, "->", dst_dir)

    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "_restored_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"archive_root": archive_root, "items": restored}, f, ensure_ascii=False, indent=2)
    print("saved:", manifest_path)


if __name__ == "__main__":
    main()
