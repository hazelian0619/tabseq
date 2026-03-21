from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tabseq.baselines.suite import load_eval_spec_from_tabseq_run
from tabseq.models.baselines import FTTransformerRegressor, MLPRegressor, QuantileMLP


def test_baseline_models_forward_shapes() -> None:
    x = torch.randn(8, 5)

    mlp = MLPRegressor(input_dim=5, hidden_dims=(16, 16), dropout=0.0)
    y = mlp(x)
    assert y.shape == (8,)

    q = QuantileMLP(input_dim=5, quantiles=(0.1, 0.9), hidden_dims=(16,), dropout=0.0)
    yq = q(x)
    assert yq.shape == (8, 2)

    ft = FTTransformerRegressor(n_num_features=5, d_model=16, n_heads=4, n_layers=1, dropout=0.0, head_hidden_dims=(8,))
    yft = ft(x)
    assert yft.shape == (8,)


def test_load_eval_spec_from_existing_tabseq_run_if_present() -> None:
    # This repo is often used locally with existing outputs/ runs. In CI the folder may not exist.
    run_dir = os.path.join("outputs", "california_housing", "run_20260202_112549")
    if not os.path.isdir(run_dir):
        pytest.skip("local TabSeq run dir not present")

    spec = load_eval_spec_from_tabseq_run(run_dir, dataset_fallback="california_housing", clip_range=True)
    assert spec.dataset == "california_housing"
    assert spec.seed == 0
    assert np.isfinite(spec.confidence) and 0.0 < spec.confidence < 1.0
    assert spec.v_min is not None and spec.v_max is not None
    assert spec.v_min < spec.v_max

