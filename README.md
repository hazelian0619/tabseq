# TabSeq-Trace Regression

This repository turns tabular regression (continuous `y`) into predicting a
binary decision trace over a fixed binning. The pipeline outputs:

- Point accuracy: MAE / RMSE
- Interval quality: PICP / MPIW

## Project Layout

- `src/tabseq/` core library (data, labels, models, metrics)
- `scripts/` training/eval/inference entrypoints
- `scripts/analysis/` analysis utilities (beam, temperature sweep, plots)
- `docs/` execution plan, dev log, and reports
- `configs/` runnable configs (eval/inference/sweeps)
- `pre/` legacy notebooks and reference PDFs

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install numpy pandas torch scikit-learn
```

Train (minimal run):

```bash
python scripts/train.py --dataset california_housing
```

Evaluate:

```bash
python scripts/eval.py --ckpt outputs/<dataset>/run_<timestamp>/
```

## Notes

- The main execution milestones live in `docs/EXECUTION.md`.
- The architecture walkthrough is in `docs/ARCHITECTURE.md`.
