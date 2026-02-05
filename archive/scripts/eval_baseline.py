import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from tabseq.data.datasets import load_california_housing_split
from tabseq.metrics.regression import compute_point_interval_metrics
from tabseq.models.baselines import FTTransformerRegressor, MLPRegressor, QuantileMLP
from tabseq.utils.seed import set_seed


def _resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint.pt not found in dir: {path}")
        return ckpt_path
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="baseline run dir or checkpoint.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    set_seed(int(cfg.get("seed", 0)))
    split = load_california_housing_split(random_state=int(cfg.get("seed", 0)))
    X_val = torch.from_numpy(split.X_val)
    y_val = torch.from_numpy(split.y_val)

    ds = TensorDataset(X_val, y_val)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    input_dim = int(cfg["n_num_features"])
    if cfg["model"] == "mlp":
        model = MLPRegressor(input_dim=input_dim, hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"])
    elif cfg["model"] == "quantile":
        model = QuantileMLP(
            input_dim=input_dim,
            quantiles=cfg["quantiles"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )
    elif cfg["model"] in ("ft_transformer", "tab_transformer"):
        model = FTTransformerRegressor(
            n_num_features=input_dim,
            cat_cardinalities=cfg.get("cat_cardinalities", []),
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            head_hidden_dims=cfg.get("hidden_dims", [128, 128]),
        )
    else:
        raise ValueError(f"unknown baseline model: {cfg['model']}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds_all = []
    with torch.no_grad():
        for x_batch, _ in dl:
            preds_all.append(model(x_batch))
    preds = torch.cat(preds_all, dim=0)

    if cfg["model"] in ("mlp", "ft_transformer", "tab_transformer"):
        y_pred = preds.view(-1)
        q = float(cfg["residual_quantile"])
        y_lower = y_pred - q
        y_upper = y_pred + q
    else:
        y_pred = None
        y_low = preds[:, 0]
        y_high = preds[:, 1]
        y_lower = torch.minimum(y_low, y_high)
        y_upper = torch.maximum(y_low, y_high)

    metrics = compute_point_interval_metrics(
        y_true=y_val,
        y_pred=y_pred,
        y_lower=y_lower,
        y_upper=y_upper,
        confidence=float(cfg["confidence"]),
        return_extras=True,
    )
    metrics["model"] = cfg["model"]

    out_dir = os.path.dirname(ckpt_path)
    out_path = os.path.join(out_dir, "metrics_val.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
