# Repository Guidelines

This repository prototypes **TabSeq-Trace regression**: turn tabular regression (continuous `y`) into predicting a **binary decision trace** over a fixed binning. The end goal is a reproducible pipeline that outputs both point accuracy (MAE/RMSE) and interval quality (PICP/MPIW).

## Project Structure & Module Organization

- `tabseq_trace_design.ipynb` — label encoding (`TraceLabelEncoder`), dataset packing (`TabSeqDataset`), and evaluation (`ExtendedHolographicMetric`).
- `quantile_regression_extended_benchmark.ipynb` — baseline interval benchmark (CatBoost; optional RealMLP if installed).
- `docs/EXECUTION.md` — execution plan and milestones (M0→M3); treat as source of truth.
- `tabseq.pdf` / `Revisiting Deep Learning Models for Tabular Data.pdf` — method + background.
- Planned (not yet added): `src/`, `scripts/`, `tests/`, local `outputs/` (ignored).

## Build, Test, and Development Commands

- Environment + notebooks:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install numpy pandas torch scikit-learn jupyter catboost` (add `pytabkit` if using RealMLP)
  - `jupyter lab` — run the prototype notebooks
- Planned (once `scripts/` exists):
  - `python scripts/train.py --config configs/default.yaml` — train and write to `outputs/`
  - `python scripts/eval.py --ckpt outputs/<exp>/checkpoint.pt` — evaluate and emit JSON/CSV
  - `python -m pytest -q` — run tests

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep pure logic in `src/` (avoid notebook-only implementations); add type hints where helpful.
- Naming:
  - files: `snake_case.py` (e.g., `trace_encoder.py`)
  - classes: `PascalCase` (e.g., `TraceLabelEncoder`)
  - configs: `configs/<dataset>.yaml`, experiments under `outputs/<dataset>/<model>/<timestamp>/`

## Testing Guidelines

- Current state: no test suite yet. When adding one, use `pytest` with `tests/test_*.py`.
- Minimum targets: `TraceLabelEncoder` round-trip sanity; metric shape checks; one-batch smoke train/eval.

## Commit & Pull Request Guidelines

- Commit messages follow a lightweight conventional style (seen in history): `init: ...`, `docs: ...`, `feat(...): ...`, `fix(...): ...`
- PRs should include:
  - what changed + why, how to reproduce (exact command), and metrics/outputs if applicable.

## Architecture Notes (Quick)

Core pipeline: tabular features `X` → encode `y` into `(y_seq, y_mht)` → model predicts per-step probabilities → reconstruct leaf distribution → compute point + interval metrics.

## Agent-Specific Instructions (for Codex)

- Start from `docs/EXECUTION.md` (M0→M1) and restate each milestone in plain language before coding.
- Implement the smallest runnable baseline first (e.g., `x_num -> per-step probs`), then iterate toward a Transformer-based model.
- Prefer small, verifiable steps: one-command entry points, deterministic seeds, and at least one smoke test.
