import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from tabseq.data.datasets import load_california_housing_split
from tabseq.metrics.regression import pinball_loss
from tabseq.models.baselines import FTTransformerRegressor, MLPRegressor, QuantileMLP
from tabseq.utils.git import get_git_hash
from tabseq.utils.seed import set_seed


def _compute_residual_quantile(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, confidence: float) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).view(-1)
    residuals = torch.abs(pred - y.view(-1)).cpu().numpy()
    return float(np.quantile(residuals, float(confidence)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "quantile", "ft_transformer", "tab_transformer"], default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--confidence", type=float, default=0.90)
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    args = ap.parse_args()

    set_seed(args.seed)

    split = load_california_housing_split(random_state=args.seed)
    X_train_full = split.X_train
    y_train_full = split.y_train
    if not (0.0 < args.calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0,1)")
    X_train_np, X_cal_np, y_train_np, y_cal_np = train_test_split(
        X_train_full,
        y_train_full,
        test_size=float(args.calibration_fraction),
        random_state=args.seed,
    )
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_cal = torch.from_numpy(X_cal_np)
    y_cal = torch.from_numpy(y_cal_np)

    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    if args.model == "tab_transformer":
        print("warning: 'tab_transformer' is deprecated; use 'ft_transformer' instead")
    if args.model == "mlp":
        model = MLPRegressor(input_dim=input_dim, hidden_dims=args.hidden_dims, dropout=args.dropout)
        loss_fn = torch.nn.MSELoss()
        quantiles = None
    elif args.model == "quantile":
        alpha = 1.0 - float(args.confidence)
        quantiles = [alpha / 2.0, 1.0 - (alpha / 2.0)]
        model = QuantileMLP(
            input_dim=input_dim,
            quantiles=quantiles,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
        loss_fn = None
    else:
        model = FTTransformerRegressor(
            n_num_features=input_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            head_hidden_dims=args.hidden_dims,
        )
        loss_fn = torch.nn.MSELoss()
        quantiles = None

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for x_batch, y_batch in dl:
            if args.model in ("mlp", "ft_transformer", "tab_transformer"):
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
            else:
                preds = model(x_batch)
                loss = torch.zeros((), dtype=preds.dtype, device=preds.device)
                for i, q in enumerate(quantiles):
                    loss = loss + pinball_loss(y_batch, preds[:, i], q)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch={epoch} loss={loss.item():.6f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model
    out_dir = os.path.join("outputs", f"baseline_{model_name}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "model": model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "hidden_dims": args.hidden_dims,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "confidence": float(args.confidence),
        "n_num_features": int(input_dim),
        "cat_cardinalities": [],
    }

    if model_name in ("mlp", "ft_transformer", "tab_transformer"):
        residual_q = _compute_residual_quantile(model, X_cal, y_cal, args.confidence)
        cfg["residual_quantile"] = residual_q
    else:
        cfg["quantiles"] = quantiles
    cfg["calibration_fraction"] = float(args.calibration_fraction)

    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")
    with open(os.path.join(out_dir, "metrics_train.json"), "w", encoding="utf-8") as f:
        json.dump({"final_loss": float(loss.item())}, f, ensure_ascii=False, indent=2)

    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()
