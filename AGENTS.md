# Repository Guidelines

This repository prototypes **TabSeq-Trace regression**: turn tabular regression
(continuous `y`) into predicting a **binary decision trace** over a fixed binning.
The end goal is a reproducible pipeline that outputs both point accuracy
(MAE/RMSE) and interval quality (PICP/MPIW).

## Project Structure & Module Organization

- `src/tabseq/` — core library (data/labels/models/metrics/utils).
- `scripts/` — training/eval/inference entrypoints.
- `scripts/analysis/` — analysis utilities (beam search, temperature sweep, plots).
- `configs/` — runnable configs (eval/inference/sweeps).
- `docs/` — execution plan, dev log, reports, and architecture notes.
- `pre/` — legacy notebooks and reference PDFs.
- `outputs/` — local experiment outputs (ignored).

## Build, Test, and Development Commands

- Environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install -e .`
  - `python -m pip install numpy pandas torch scikit-learn`
- Train:
  - `python scripts/train.py --dataset california_housing`
- Eval:
  - `python scripts/eval.py --ckpt outputs/<dataset>/run_<timestamp>/`
- Tests:
  - `python -m pytest -q`

## Coding Style & Naming Conventions

- Python: 4-space indentation; type hints where helpful.
- Naming:
  - files: `snake_case.py` (e.g., `trace_encoder.py`)
  - classes: `PascalCase` (e.g., `TraceLabelEncoder`)
  - configs: `configs/<task>.yaml`, experiments under `outputs/<dataset>/<model>/<timestamp>/`

## Testing Guidelines

- Use `pytest` with `test_*.py`.
- Minimum targets: `TraceLabelEncoder` round-trip sanity; metric shape checks;
  one-batch smoke train/eval.

## Commit & Pull Request Guidelines

- Commit messages follow a lightweight conventional style: `init: ...`, `docs: ...`,
  `feat(...): ...`, `fix(...): ...`.
- PRs should include:
  - what changed + why, how to reproduce (exact command), metrics/outputs if applicable.

## Architecture Notes (Quick)

Core pipeline: tabular features `X` → encode `y` into `(y_seq, y_mht)` → model
predicts per-step probabilities → reconstruct leaf distribution → compute point
and interval metrics.
