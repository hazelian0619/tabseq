#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.baselines.suite import load_eval_spec_from_tabseq_run, run_suite


def _pick_latest_run_dir(dataset: str, outputs_root: str = "outputs") -> str:
    base = os.path.join(outputs_root, dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"dataset outputs dir not found: {base}")
    candidates: List[str] = []
    for name in os.listdir(base):
        if name.startswith("run_"):
            path = os.path.join(base, name)
            if os.path.isdir(path):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no run_* dirs found under: {base}")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. california_housing, kin8nm")
    ap.add_argument(
        "--tabseq-run",
        default=None,
        help="TabSeq standard run dir (defaults to latest outputs/<dataset>/run_*)",
    )
    ap.add_argument(
        "--models",
        default="mlp,quantile,ft_transformer,catboost,realmlp",
        help="comma-separated list: mlp,quantile,ft_transformer,catboost,realmlp",
    )
    ap.add_argument("--out-root", default="outputs/baselines")
    ap.add_argument("--run-id", default=None, help="override run_id (default: timestamp)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-epochs", type=int, default=10)
    ap.add_argument("--no-clip-range", action="store_true", help="disable clip_range (not recommended)")
    args = ap.parse_args()

    tabseq_run_dir: Optional[str] = args.tabseq_run
    if tabseq_run_dir is None:
        tabseq_run_dir = _pick_latest_run_dir(args.dataset)

    spec = load_eval_spec_from_tabseq_run(
        tabseq_run_dir,
        dataset_fallback=str(args.dataset),
        clip_range=not bool(args.no_clip_range),
    )
    models = [m.strip() for m in str(args.models).split(",") if m.strip()]

    results = run_suite(
        spec=spec,
        models=models,
        out_root=str(args.out_root),
        run_id=args.run_id,
        device=str(args.device),
        torch_epochs=int(args.torch_epochs),
    )

    print("eval_spec:", spec.to_dict())
    for m, path in results.items():
        print(f"saved[{m}]: {path}")


if __name__ == "__main__":
    main()
