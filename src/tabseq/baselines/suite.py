from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from tabseq.baselines.spec import BaselineEvalSpec
from tabseq.data.datasets import DatasetSplit, load_dataset_split
from tabseq.metrics.regression import compute_point_interval_metrics, pinball_loss
from tabseq.models.baselines import FTTransformerRegressor, MLPRegressor, QuantileMLP
from tabseq.utils.git import get_git_hash
from tabseq.utils.seed import set_seed


def _pick_latest_json(run_dir: str, prefix: str) -> Optional[str]:
    candidates: List[str] = []
    for name in os.listdir(run_dir):
        if name.startswith(prefix) and name.endswith(".json"):
            candidates.append(os.path.join(run_dir, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_eval_spec_from_tabseq_run(
    tabseq_run_dir: str,
    *,
    dataset_fallback: Optional[str] = None,
    clip_range: bool = True,
) -> BaselineEvalSpec:
    cfg_path = os.path.join(tabseq_run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"TabSeq config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f) or {}

    dataset = cfg.get("dataset") or dataset_fallback
    if not dataset:
        raise ValueError("dataset not found in TabSeq config.json; pass dataset_fallback")

    seed = int(cfg.get("seed", 0))
    v_min = cfg.get("v_min")
    v_max = cfg.get("v_max")
    v_min_f = float(v_min) if v_min is not None else None
    v_max_f = float(v_max) if v_max is not None else None

    # Confidence lives in eval_config*.json (TabSeq can have many).
    eval_cfg_path = os.path.join(tabseq_run_dir, "eval_config.json")
    if not os.path.isfile(eval_cfg_path):
        eval_cfg_path = _pick_latest_json(tabseq_run_dir, "eval_config")
    confidence = 0.9
    if eval_cfg_path and os.path.isfile(eval_cfg_path):
        with open(eval_cfg_path, "r", encoding="utf-8") as f:
            eval_cfg = json.load(f) or {}
        if eval_cfg.get("confidence") is not None:
            confidence = float(eval_cfg["confidence"])
        width_bins = eval_cfg.get("width_bins")
        bin_step_02 = eval_cfg.get("bin_step_02")
        bin_step_04 = eval_cfg.get("bin_step_04")
    else:
        width_bins = None
        bin_step_02 = None
        bin_step_04 = None

    if isinstance(width_bins, str):
        width_bins = [float(v) for v in width_bins.split(",") if str(v).strip()]
    elif isinstance(width_bins, (list, tuple)):
        width_bins = [float(v) for v in width_bins]
    else:
        width_bins = None

    return BaselineEvalSpec(
        dataset=str(dataset),
        seed=seed,
        confidence=float(confidence),
        v_min=v_min_f,
        v_max=v_max_f,
        clip_range=bool(clip_range),
        width_bins=tuple(width_bins) if width_bins else (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0),
        bin_step_02=float(bin_step_02) if bin_step_02 is not None else None,
        bin_step_04=float(bin_step_04) if bin_step_04 is not None else None,
    )


def _resolve_bin_edges(spec: BaselineEvalSpec) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if spec.bin_step_02 is None and spec.bin_step_04 is None:
        return None, None
    if spec.v_min is None or spec.v_max is None:
        raise ValueError("bin_step_02/bin_step_04 require v_min/v_max in eval spec")
    edges_02 = None
    edges_04 = None
    if spec.bin_step_02 is not None:
        edges_02 = np.arange(spec.v_min, spec.v_max + spec.bin_step_02, spec.bin_step_02)
    if spec.bin_step_04 is not None:
        edges_04 = np.arange(spec.v_min, spec.v_max + spec.bin_step_04, spec.bin_step_04)
    return edges_02, edges_04


def default_suite_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(out_root: str, *, dataset: str, model: str, run_id: str) -> str:
    # Canonical structure:
    #   outputs/baselines/<dataset>/<model>/run_<run_id>/
    return os.path.join(out_root, dataset, model, f"run_{run_id}")


def make_legacy_alias_path(out_root: str, *, model: str, run_id: str) -> str:
    # Backward-compat alias for older docs:
    #   outputs/baselines/baseline_<model>_<run_id> -> <dataset>/<model>/run_<run_id>
    return os.path.join(out_root, f"baseline_{model}_{run_id}")


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_git_txt(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")


def ensure_legacy_alias(out_root: str, *, dataset: str, model: str, run_id: str) -> str:
    alias = make_legacy_alias_path(out_root, model=model, run_id=run_id)
    target = os.path.relpath(make_run_dir(out_root, dataset=dataset, model=model, run_id=run_id), os.path.dirname(alias))

    # Replace any existing file/symlink to keep the alias correct.
    if os.path.lexists(alias):
        try:
            os.remove(alias)
        except IsADirectoryError:
            # Should not happen in normal usage; keep safe.
            raise RuntimeError(f"Refusing to replace directory alias at {alias}; please delete manually.")

    os.symlink(target, alias)
    return alias


def _split_train_cal(
    split: DatasetSplit, *, seed: int, calibration_fraction: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not (0.0 < float(calibration_fraction) < 1.0):
        raise ValueError("calibration_fraction must be in (0,1)")
    arrays: List[np.ndarray] = [split.X_train, split.y_train]
    has_cat = split.X_cat_train is not None and split.X_cat_train.shape[1] > 0
    if has_cat:
        arrays.insert(1, split.X_cat_train)

    parts = train_test_split(*arrays, test_size=float(calibration_fraction), random_state=int(seed))
    if not has_cat:
        X_tr, X_cal, y_tr, y_cal = parts
        return X_tr, X_cal, y_tr, y_cal, None, None

    X_tr, X_cal, X_cat_tr, X_cat_cal, y_tr, y_cal = parts
    return X_tr, X_cal, y_tr, y_cal, X_cat_tr, X_cat_cal


def _torch_clip_if_needed(
    y_true: torch.Tensor,
    y_lower: torch.Tensor,
    y_upper: torch.Tensor,
    *,
    v_min: Optional[float],
    v_max: Optional[float],
    clip_range: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not clip_range or v_min is None or v_max is None:
        return y_true, y_lower, y_upper
    lo = float(v_min)
    hi = float(v_max)
    return torch.clamp(y_true, lo, hi), torch.clamp(y_lower, lo, hi), torch.clamp(y_upper, lo, hi)


def _compute_residual_quantile_point_model(
    *,
    model: torch.nn.Module,
    x_cal_num: torch.Tensor,
    y_cal: torch.Tensor,
    confidence: float,
    x_cal_cat: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> float:
    model.eval()
    with torch.no_grad():
        if x_cal_cat is None:
            pred = model(x_cal_num.to(device)).view(-1)
        else:
            pred = model(x_cal_num.to(device), x_cal_cat.to(device)).view(-1)
    residuals = torch.abs(pred.cpu() - y_cal.view(-1).cpu()).numpy()
    return float(np.quantile(residuals, float(confidence)))


def _predict_batches_point_model(
    *,
    model: torch.nn.Module,
    dl: DataLoader,
    x_has_cat: bool,
    device: str,
) -> torch.Tensor:
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dl:
            if x_has_cat:
                x_num, x_cat, _y = batch
                outs.append(model(x_num.to(device), x_cat.to(device)).detach().cpu())
            else:
                x_num, _y = batch
                outs.append(model(x_num.to(device)).detach().cpu())
    return torch.cat(outs, dim=0)


def _predict_batches_quantile_model(model: torch.nn.Module, dl: DataLoader, device: str) -> torch.Tensor:
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for x_num, _y in dl:
            outs.append(model(x_num.to(device)).detach().cpu())
    return torch.cat(outs, dim=0)


def run_torch_baseline(
    *,
    spec: BaselineEvalSpec,
    model: str,
    out_root: str,
    run_id: str,
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    dropout: float = 0.1,
    hidden_dims: Sequence[int] = (128, 128),
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    calibration_fraction: float = 0.2,
) -> str:
    if model not in {"mlp", "quantile", "ft_transformer", "tab_transformer"}:
        raise ValueError(f"unsupported torch baseline model: {model}")

    set_seed(int(spec.seed))
    split = load_dataset_split(spec.dataset, random_state=int(spec.seed))

    # Only FT-Transformer baseline supports categorical features in this repo.
    has_cat = split.X_cat_train is not None and split.X_cat_train.shape[1] > 0
    if has_cat and model not in {"ft_transformer", "tab_transformer"}:
        raise ValueError(f"dataset {spec.dataset} has categorical features; use ft_transformer (got {model})")

    X_tr, X_cal, y_tr, y_cal, X_cat_tr, X_cat_cal = _split_train_cal(
        split, seed=int(spec.seed), calibration_fraction=float(calibration_fraction)
    )

    x_tr_num = torch.from_numpy(np.asarray(X_tr)).float()
    y_tr_t = torch.from_numpy(np.asarray(y_tr)).float()
    x_cal_num = torch.from_numpy(np.asarray(X_cal)).float()
    y_cal_t = torch.from_numpy(np.asarray(y_cal)).float()

    x_val_num = torch.from_numpy(np.asarray(split.X_val)).float()
    y_val_t = torch.from_numpy(np.asarray(split.y_val)).float()

    x_tr_cat_t = None
    x_cal_cat_t = None
    x_val_cat_t = None
    cat_cardinalities: List[int] = []
    if has_cat:
        x_tr_cat_t = torch.from_numpy(np.asarray(X_cat_tr)).long()
        x_cal_cat_t = torch.from_numpy(np.asarray(X_cat_cal)).long()
        x_val_cat_t = torch.from_numpy(np.asarray(split.X_cat_val)).long()
        cat_cardinalities = list(split.cat_cardinalities or [])

    if model == "tab_transformer":
        # Keep compatibility with older naming.
        model = "ft_transformer"

    input_dim = int(x_tr_num.shape[1])
    if model == "mlp":
        net = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=float(dropout))
        loss_fn = torch.nn.MSELoss()
        quantiles = None
    elif model == "quantile":
        alpha = 1.0 - float(spec.confidence)
        quantiles = [alpha / 2.0, 1.0 - (alpha / 2.0)]
        net = QuantileMLP(input_dim=input_dim, quantiles=quantiles, hidden_dims=hidden_dims, dropout=float(dropout))
        loss_fn = None
    else:
        net = FTTransformerRegressor(
            n_num_features=input_dim,
            cat_cardinalities=cat_cardinalities,
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            dropout=float(dropout),
            head_hidden_dims=hidden_dims,
        )
        loss_fn = torch.nn.MSELoss()
        quantiles = None

    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))

    if has_cat:
        ds = TensorDataset(x_tr_num, x_tr_cat_t, y_tr_t)  # type: ignore[arg-type]
    else:
        ds = TensorDataset(x_tr_num, y_tr_t)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True)

    net.train()
    for _epoch in range(int(epochs)):
        for batch in dl:
            if has_cat:
                x_num_b, x_cat_b, y_b = batch
            else:
                x_num_b, y_b = batch

            if model != "quantile":
                if has_cat:
                    pred = net(x_num_b.to(device), x_cat_b.to(device))
                else:
                    pred = net(x_num_b.to(device))
                loss = loss_fn(pred.view(-1), y_b.to(device).view(-1))  # type: ignore[misc]
            else:
                preds = net(x_num_b.to(device))
                loss = torch.zeros((), dtype=preds.dtype, device=preds.device)
                assert quantiles is not None
                for i, q in enumerate(quantiles):
                    loss = loss + pinball_loss(y_b.to(device), preds[:, i], float(q))

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Build val dataloader.
    if has_cat:
        ds_val = TensorDataset(x_val_num, x_val_cat_t, y_val_t)  # type: ignore[arg-type]
        dl_val = DataLoader(ds_val, batch_size=int(batch_size), shuffle=False)
    else:
        ds_val = TensorDataset(x_val_num, y_val_t)
        dl_val = DataLoader(ds_val, batch_size=int(batch_size), shuffle=False)

    if model != "quantile":
        residual_q = _compute_residual_quantile_point_model(
            model=net,
            x_cal_num=x_cal_num,
            x_cal_cat=x_cal_cat_t,
            y_cal=y_cal_t,
            confidence=float(spec.confidence),
            device=device,
        )
        preds = _predict_batches_point_model(model=net, dl=dl_val, x_has_cat=has_cat, device=device).view(-1)
        y_lower = preds - float(residual_q)
        y_upper = preds + float(residual_q)
        y_pred = preds
        interval_meta: Dict[str, Any] = {"residual_quantile": float(residual_q)}
    else:
        preds_q = _predict_batches_quantile_model(net, dl_val, device=device)
        y_lower = torch.minimum(preds_q[:, 0], preds_q[:, 1]).view(-1)
        y_upper = torch.maximum(preds_q[:, 0], preds_q[:, 1]).view(-1)
        y_pred = (y_lower + y_upper) / 2.0
        interval_meta = {"quantiles": quantiles}

    y_true = y_val_t.view(-1)
    y_true, y_lower, y_upper = _torch_clip_if_needed(
        y_true, y_lower, y_upper, v_min=spec.v_min, v_max=spec.v_max, clip_range=bool(spec.clip_range)
    )

    bin_edges_02, bin_edges_04 = _resolve_bin_edges(spec)
    metrics = compute_point_interval_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        confidence=float(spec.confidence),
        return_extras=True,
        width_bins=spec.width_bins,
        bin_edges_02=bin_edges_02,
        bin_edges_04=bin_edges_04,
        v_min=spec.v_min,
        v_max=spec.v_max,
    )
    metrics.update({"model": model, "dataset": spec.dataset, "run_id": run_id, "eval_spec": asdict(spec)})

    run_dir = make_run_dir(out_root, dataset=spec.dataset, model=model, run_id=run_id)
    os.makedirs(run_dir, exist_ok=True)
    write_git_txt(run_dir)

    cfg: Dict[str, Any] = {
        "dataset": spec.dataset,
        "model": model,
        "run_id": run_id,
        "seed": int(spec.seed),
        "confidence": float(spec.confidence),
        "v_min": spec.v_min,
        "v_max": spec.v_max,
        "clip_range": bool(spec.clip_range),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "dropout": float(dropout),
        "hidden_dims": list(map(int, hidden_dims)),
        "calibration_fraction": float(calibration_fraction),
        "n_num_features": int(input_dim),
        "cat_cardinalities": cat_cardinalities,
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "n_layers": int(n_layers),
        **interval_meta,
    }

    # Save checkpoint for torch baselines.
    ckpt = {"model_state_dict": net.state_dict(), "config": cfg}
    torch.save(ckpt, os.path.join(run_dir, "checkpoint.pt"))

    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(os.path.join(run_dir, "metrics_val.json"), metrics)

    # Save raw arrays for eval-only recomputation (and debugging).
    np.save(os.path.join(run_dir, "y_true_val.npy"), y_true.detach().cpu().numpy())
    np.save(os.path.join(run_dir, "y_pred_val.npy"), y_pred.detach().cpu().numpy())
    np.save(os.path.join(run_dir, "y_lower_val.npy"), y_lower.detach().cpu().numpy())
    np.save(os.path.join(run_dir, "y_upper_val.npy"), y_upper.detach().cpu().numpy())

    ensure_legacy_alias(out_root, dataset=spec.dataset, model=model, run_id=run_id)
    return run_dir


def run_catboost_baseline(
    *,
    spec: BaselineEvalSpec,
    out_root: str,
    run_id: str,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.1,
    l2_leaf_reg: float = 3.0,
) -> str:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("catboost is required for the CatBoost baseline") from exc

    set_seed(int(spec.seed))
    split = load_dataset_split(spec.dataset, random_state=int(spec.seed))

    X_train = split.X_train
    y_train = split.y_train
    X_val = split.X_val
    y_val = split.y_val

    # CatBoost supports categorical features, but only if we provide them explicitly.
    cat_features: List[int] = []
    if split.X_cat_train is not None and split.X_cat_train.shape[1] > 0:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for catboost with categorical features") from exc
        n_num = split.X_train.shape[1]
        X_num_train = X_train
        X_num_val = X_val
        X_cat_train = split.X_cat_train.astype(np.int64)
        X_cat_val = split.X_cat_val.astype(np.int64)
        num_cols = [f"num_{i}" for i in range(n_num)]
        cat_cols = [f"cat_{i}" for i in range(X_cat_train.shape[1])]
        X_train = pd.concat(
            [pd.DataFrame(X_num_train, columns=num_cols), pd.DataFrame(X_cat_train, columns=cat_cols)],
            axis=1,
        )
        X_val = pd.concat(
            [pd.DataFrame(X_num_val, columns=num_cols), pd.DataFrame(X_cat_val, columns=cat_cols)],
            axis=1,
        )
        cat_features = list(range(n_num, n_num + X_cat_train.shape[1]))

    alpha = 1.0 - float(spec.confidence)
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    model = CatBoostRegressor(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function=f"MultiQuantile:alpha={q_lower},{q_upper}",
        random_seed=int(spec.seed),
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features or None)

    preds = np.asarray(model.predict(X_val))
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError(f"expected MultiQuantile predictions with shape (N, 2), got {preds.shape}")
    y_lower = np.minimum(preds[:, 0], preds[:, 1]).astype(np.float32)
    y_upper = np.maximum(preds[:, 0], preds[:, 1]).astype(np.float32)
    y_true = np.asarray(y_val, dtype=np.float32)

    if spec.clip_range and spec.v_min is not None and spec.v_max is not None:
        y_true = np.clip(y_true, spec.v_min, spec.v_max)
        y_lower = np.clip(y_lower, spec.v_min, spec.v_max)
        y_upper = np.clip(y_upper, spec.v_min, spec.v_max)

    bin_edges_02, bin_edges_04 = _resolve_bin_edges(spec)
    metrics = compute_point_interval_metrics(
        y_true=torch.from_numpy(y_true),
        y_lower=torch.from_numpy(y_lower),
        y_upper=torch.from_numpy(y_upper),
        confidence=float(spec.confidence),
        return_extras=True,
        width_bins=spec.width_bins,
        bin_edges_02=bin_edges_02,
        bin_edges_04=bin_edges_04,
        v_min=spec.v_min,
        v_max=spec.v_max,
    )
    metrics.update({"model": "catboost", "dataset": spec.dataset, "run_id": run_id, "eval_spec": asdict(spec)})

    run_dir = make_run_dir(out_root, dataset=spec.dataset, model="catboost", run_id=run_id)
    os.makedirs(run_dir, exist_ok=True)
    write_git_txt(run_dir)

    cfg: Dict[str, Any] = {
        "dataset": spec.dataset,
        "model": "catboost",
        "run_id": run_id,
        "seed": int(spec.seed),
        "confidence": float(spec.confidence),
        "quantiles": [float(q_lower), float(q_upper)],
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": float(l2_leaf_reg),
        "v_min": spec.v_min,
        "v_max": spec.v_max,
        "clip_range": bool(spec.clip_range),
    }

    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(os.path.join(run_dir, "metrics_val.json"), metrics)
    model.save_model(os.path.join(run_dir, "model.cbm"))

    np.save(os.path.join(run_dir, "y_true_val.npy"), y_true)
    np.save(os.path.join(run_dir, "y_lower_val.npy"), y_lower)
    np.save(os.path.join(run_dir, "y_upper_val.npy"), y_upper)

    ensure_legacy_alias(out_root, dataset=spec.dataset, model="catboost", run_id=run_id)
    return run_dir


def run_realmlp_baseline(
    *,
    spec: BaselineEvalSpec,
    out_root: str,
    run_id: str,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 256,
) -> str:
    try:
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pytabkit is required for the RealMLP baseline") from exc

    set_seed(int(spec.seed))
    split = load_dataset_split(spec.dataset, random_state=int(spec.seed))
    if split.X_cat_train is not None and split.X_cat_train.shape[1] > 0:
        raise ValueError("RealMLP baseline currently supports numeric-only datasets in this repo")

    X_train = split.X_train
    y_train = split.y_train
    X_val = split.X_val
    y_val = split.y_val

    alpha = 1.0 - float(spec.confidence)
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    model = RealMLP_TD_Regressor(
        train_metric_name=f"multi_pinball({q_lower},{q_upper})",
        random_state=int(spec.seed),
        device=str(device),
        n_epochs=int(epochs),
        batch_size=int(batch_size),
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    preds = np.asarray(model.predict(X_val))
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError(f"expected RealMLP predictions with shape (N, 2), got {preds.shape}")
    y_lower = np.minimum(preds[:, 0], preds[:, 1]).astype(np.float32)
    y_upper = np.maximum(preds[:, 0], preds[:, 1]).astype(np.float32)
    y_true = np.asarray(y_val, dtype=np.float32)

    if spec.clip_range and spec.v_min is not None and spec.v_max is not None:
        y_true = np.clip(y_true, spec.v_min, spec.v_max)
        y_lower = np.clip(y_lower, spec.v_min, spec.v_max)
        y_upper = np.clip(y_upper, spec.v_min, spec.v_max)

    bin_edges_02, bin_edges_04 = _resolve_bin_edges(spec)
    metrics = compute_point_interval_metrics(
        y_true=torch.from_numpy(y_true),
        y_lower=torch.from_numpy(y_lower),
        y_upper=torch.from_numpy(y_upper),
        confidence=float(spec.confidence),
        return_extras=True,
        width_bins=spec.width_bins,
        bin_edges_02=bin_edges_02,
        bin_edges_04=bin_edges_04,
        v_min=spec.v_min,
        v_max=spec.v_max,
    )
    metrics.update({"model": "realmlp", "dataset": spec.dataset, "run_id": run_id, "eval_spec": asdict(spec)})

    run_dir = make_run_dir(out_root, dataset=spec.dataset, model="realmlp", run_id=run_id)
    os.makedirs(run_dir, exist_ok=True)
    write_git_txt(run_dir)

    cfg: Dict[str, Any] = {
        "dataset": spec.dataset,
        "model": "realmlp",
        "run_id": run_id,
        "seed": int(spec.seed),
        "confidence": float(spec.confidence),
        "quantiles": [float(q_lower), float(q_upper)],
        "device": str(device),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "v_min": spec.v_min,
        "v_max": spec.v_max,
        "clip_range": bool(spec.clip_range),
    }

    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(os.path.join(run_dir, "metrics_val.json"), metrics)

    # RealMLP object may not be trivially serializable across versions.
    # Save raw outputs for eval-only recomputation.
    np.save(os.path.join(run_dir, "y_true_val.npy"), y_true)
    np.save(os.path.join(run_dir, "y_lower_val.npy"), y_lower)
    np.save(os.path.join(run_dir, "y_upper_val.npy"), y_upper)
    np.save(os.path.join(run_dir, "preds_val.npy"), preds.astype(np.float32))

    ensure_legacy_alias(out_root, dataset=spec.dataset, model="realmlp", run_id=run_id)
    return run_dir


def run_suite(
    *,
    spec: BaselineEvalSpec,
    models: Sequence[str],
    out_root: str = "outputs/baselines",
    run_id: Optional[str] = None,
    device: str = "cpu",
    torch_epochs: int = 10,
) -> Dict[str, str]:
    """
    Run a baseline suite (train+eval) under a single run_id.

    Returns: {model_name: run_dir}.
    """
    rid = run_id or default_suite_run_id()
    results: Dict[str, str] = {}
    for m in models:
        name = m.strip().lower()
        if not name:
            continue
        if name in {"mlp", "quantile", "ft_transformer", "tab_transformer"}:
            results[name] = run_torch_baseline(
                spec=spec,
                model=name,
                out_root=out_root,
                run_id=rid,
                device=device,
                epochs=int(torch_epochs),
            )
        elif name == "catboost":
            results[name] = run_catboost_baseline(spec=spec, out_root=out_root, run_id=rid)
        elif name == "realmlp":
            results[name] = run_realmlp_baseline(spec=spec, out_root=out_root, run_id=rid, device=device)
        else:
            raise ValueError(f"unknown baseline model: {name}")
    return results
