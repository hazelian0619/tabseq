# Execution Plan

This document is the working source of truth for turning the prototype into a runnable, reproducible project.

## M0: Restore The Engineering Skeleton

Plain language:
- make every referenced module importable
- move notebook-only core pieces into `src/`
- ensure at least one smoke path works without manual notebook steps

Exit criteria:
- `TabSeqDataset`, `TransformerTabSeqModel`, config helpers, and dataset loaders exist in `src/`
- at least one smoke forward/backward pass works

## M1: Make The Minimal TabSeq Loop Runnable

Plain language:
- train a smallest viable TabSeq model from tabular `X`
- save a checkpoint and config under `outputs/<dataset>/run_<timestamp>/`
- run evaluation from that checkpoint and compute MAE/RMSE/PICP/MPIW

Exit criteria:
- `python scripts/train.py --config configs/default.yaml`
- `python scripts/eval.py --ckpt outputs/<dataset>/run_<timestamp>/ --config configs/default.yaml`
- training and evaluation complete without notebook code

## M2: Reproduce Comparison Runs

Plain language:
- run baseline models under the same split and interval settings
- compare TabSeq against MLP / Quantile / FT-Transformer / CatBoost / RealMLP

Exit criteria:
- `scripts/baseline_suite.py` consumes a TabSeq run dir
- baseline outputs are stored under `outputs/baselines/<dataset>/<model>/run_<run_id>/`

## M3: Improve Calibration And Analysis

Plain language:
- study why greedy / prefix inference undercovers
- compare interval extraction strategies and calibration knobs
- produce analysis figures and width-stratified diagnostics

Exit criteria:
- temperature / alpha / beam sweeps can be rerun from saved checkpoints
- reports in `reports/metrics/` and analysis scripts can be regenerated from local outputs

## Recommended Commands

Local smoke:

```bash
conda run -n tabr1 python scripts/train.py --config configs/default.yaml
conda run -n tabr1 python scripts/eval.py --ckpt outputs/diabetes/run_<timestamp>/ --config configs/default.yaml
```

Main experiments:

```bash
conda run -n tabr1 python scripts/train.py --config configs/default.yaml --dataset california_housing
conda run -n tabr1 python scripts/run_eval_suite.sh --ckpt outputs/california_housing/run_<timestamp>/
```
