# Repository Guidelines

This repository is a work-in-progress implementation of **TabSeq/Trace Regression**: reframing tabular regression as **binary decision sequence prediction** over a fixed binning of the target. Notebooks provide the original prototypes; the goal is a reproducible, scriptable pipeline (train/eval + benchmarks).

## Project Structure & Module Organization

- `docs/` — project docs and execution plan (`docs/EXECUTION.md` is the source of truth).
- `src/` — (to be added) production code:
  - `src/labels/` target encoding/decoding (e.g., `TraceLabelEncoder`)
  - `src/data/` dataset loading and `TabSeqDataset`
  - `src/models/` tabular encoders + sequence decoders
  - `src/metrics/` evaluation metrics (MAE/RMSE, PICP/MPIW, bucket accuracy)
- `scripts/` — (to be added) entry points: `scripts/train.py`, `scripts/eval.py`
- `tests/` — (to be added) unit + smoke tests
- `outputs/` — (local, ignored) experiment artifacts (checkpoints, metrics, logs)
- Current prototypes: `tabseq_trace_design.ipynb`, `quantile_regression_extended_benchmark.ipynb`

## Build, Test, and Development Commands

Planned commands (once `src/` exists):
- `python -m pytest -q` — run unit tests
- `python scripts/train.py --config configs/default.yaml` — train and save to `outputs/`
- `python scripts/eval.py --ckpt outputs/<exp>/checkpoint.pt` — evaluate and write metrics JSON/CSV

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints where helpful, keep functions small and testable.
- Naming:
  - files: `snake_case.py` (e.g., `trace_encoder.py`)
  - classes: `PascalCase` (e.g., `TraceLabelEncoder`)
  - configs: `configs/<dataset>.yaml`, experiments under `outputs/<dataset>/<model>/<timestamp>/`

## Testing Guidelines

- Framework: `pytest` (planned).
- Tests live in `tests/` and are named `test_*.py`.
- Minimum coverage expectations:
  - `TraceLabelEncoder`: `encode()` + `decode_sequence()` round-trip sanity; `encode_multi_hot()` shape and monotonic shrinking.
  - Smoke: one mini-batch through model forward pass without error.

## Commit & Pull Request Guidelines

- Commit messages follow a lightweight conventional style (seen in history):
  - `docs: ...`, `feat(...): ...`, `fix(...): ...`, `init: ...`
- PRs should include:
  - what changed + why, how to reproduce (exact command), and metrics/outputs if applicable.

## Architecture Notes (Quick)

Core pipeline: tabular features `X` → encode `y` into `(y_seq, y_mht)` → model predicts per-step logits `[depth, n_bins]` → reconstruct leaf distribution → compute point + interval metrics.

