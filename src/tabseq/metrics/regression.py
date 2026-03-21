from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    分位数回归损失（pinball loss）。
    - quantile 越大，模型更偏向预测上界；越小，偏向预测下界。
    """
    q = float(quantile)
    diff = y_true - y_pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def compute_point_interval_metrics(
    *,
    y_true: torch.Tensor,
    y_pred: Optional[torch.Tensor] = None,
    y_lower: Optional[torch.Tensor] = None,
    y_upper: Optional[torch.Tensor] = None,
    confidence: float,
    return_extras: bool = False,
    width_bins: Optional[Sequence[float]] = None,
    bin_edges_02: Optional[Sequence[float]] = None,
    bin_edges_04: Optional[Sequence[float]] = None,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
) -> Dict:
    """
    基线评估统一口径：
      - MAE/RMSE：点预测误差（越小越好）
      - PICP：区间覆盖率，应接近 confidence（越高越可靠）
      - MPIW：区间平均宽度（越小越尖锐）
    诊断项：
      - width_stratified_PICP：按区间宽度分桶的覆盖率
      - bin_acc_0.2/0.4：把 y 与 y_hat 量化后的一致性命中率

    注意：
    - 若传入 v_min/v_max，则 bin_edges 采用该范围（与 TabSeq 口径对齐）。
    - 若未传入 bin_edges 与 v_min/v_max，则使用 y_true 的最小/最大值推断边界。
    """
    y_true = y_true.view(-1)
    if y_pred is None:
        if y_lower is None or y_upper is None:
            raise ValueError("y_pred or (y_lower, y_upper) must be provided")
        y_pred = (y_lower + y_upper) / 2.0
    y_pred = y_pred.view(-1)

    # 点预测误差：MAE 更直观，RMSE 更惩罚大误差
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    if y_lower is None or y_upper is None:
        raise ValueError("y_lower and y_upper are required for PICP/MPIW")

    y_lower = y_lower.view(-1)
    y_upper = y_upper.view(-1)

    y_true_np = y_true.detach().cpu().numpy()
    y_lower_np = y_lower.detach().cpu().numpy()
    y_upper_np = y_upper.detach().cpu().numpy()

    # 区间覆盖率与宽度：可靠性 vs 尖锐度的权衡
    covered = (y_true_np >= y_lower_np) & (y_true_np <= y_upper_np)
    picp = float(np.mean(covered))
    widths = y_upper_np - y_lower_np
    mpiw = float(np.mean(widths))

    out = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "confidence": float(confidence),
        "PICP": picp,
        "MPIW": mpiw,
    }
    if not return_extras:
        return out

    width_bins = list(width_bins or [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0])
    if bin_edges_02 is None or bin_edges_04 is None:
        if v_min is None or v_max is None:
            v_min = float(np.min(y_true_np))
            v_max = float(np.max(y_true_np))
        if bin_edges_02 is None:
            bin_edges_02 = np.arange(v_min, v_max + 0.2, 0.2)
        if bin_edges_04 is None:
            bin_edges_04 = np.arange(v_min, v_max + 0.4, 0.4)
    bin_edges_02 = np.asarray(bin_edges_02)
    bin_edges_04 = np.asarray(bin_edges_04)

    # 诊断：宽度分桶后的覆盖率（窄区间通常更难命中）
    width_bins_idx = np.digitize(widths, width_bins) - 1
    width_stratified_picp: Dict[str, str] = {}
    for i in range(len(width_bins) - 1):
        mask = width_bins_idx == i
        count = int(np.sum(mask))
        if count > 0:
            local_cov = float(np.mean(covered[mask]))
            range_str = f"Width [{width_bins[i]:.1f}, {width_bins[i + 1]:.1f})"
            width_stratified_picp[range_str] = f"{local_cov:.4f} (n={count})"

    # 诊断：固定 bin 的粗粒度命中率（越高代表点预测更稳定）
    y_pred_np = y_pred.detach().cpu().numpy()
    true_bins_02 = np.digitize(y_true_np, bin_edges_02)
    pred_bins_02 = np.digitize(y_pred_np, bin_edges_02)
    bin_acc_02 = float(np.mean(true_bins_02 == pred_bins_02))

    true_bins_04 = np.digitize(y_true_np, bin_edges_04)
    pred_bins_04 = np.digitize(y_pred_np, bin_edges_04)
    bin_acc_04 = float(np.mean(true_bins_04 == pred_bins_04))

    out["width_stratified_PICP"] = width_stratified_picp
    out["bin_acc_0.2"] = bin_acc_02
    out["bin_acc_0.4"] = bin_acc_04
    return out
