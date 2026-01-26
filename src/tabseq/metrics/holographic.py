from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
from tabseq.labels.trace_encoder import TraceLabelEncoder

@dataclass(frozen=True)
class IntervalResult:
    L: torch.Tensor
    U: torch.Tensor

class ExtendedHolographicMetric:

    def __init__(self, encoder: TraceLabelEncoder, *, width_bins: Optional[list[float]]=None, bin_edges_02: Optional[np.ndarray]=None, bin_edges_04: Optional[np.ndarray]=None):
        self.encoder = encoder
        self.width_bins = width_bins or [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0]
        self.bin_edges_02 = bin_edges_02 if bin_edges_02 is not None else np.arange(0, 5.21, 0.2)
        self.bin_edges_04 = bin_edges_04 if bin_edges_04 is not None else np.arange(0, 5.21, 0.4)

    def leaf_probs_from_step_probs(self, step_probs: torch.Tensor) -> torch.Tensor:
        leaf_unnorm_probs = torch.prod(step_probs, dim=1)
        leaf_probs = leaf_unnorm_probs / (torch.sum(leaf_unnorm_probs, dim=1, keepdim=True) + 1e-09)
        return leaf_probs

    def interval_from_leaf_probs(self, leaf_probs: torch.Tensor, *, confidence: float) -> IntervalResult:
        cdf = torch.cumsum(leaf_probs, dim=1)
        alpha = 1.0 - float(confidence)
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        lower_indices = torch.argmax((cdf >= lower_q).int(), dim=1)
        upper_indices = torch.argmax((cdf >= upper_q).int(), dim=1)
        bin_values = torch.tensor([self.encoder.decode_bin_index(i) for i in range(leaf_probs.shape[1])], dtype=torch.float32, device=leaf_probs.device)
        L = bin_values[lower_indices]
        U = bin_values[upper_indices]
        return IntervalResult(L=L, U=U)

    def compute_metrics(self, model_probs: torch.Tensor, y_true: torch.Tensor, *, confidence: float=0.9, return_extras: bool=True) -> Dict:
        if model_probs.ndim != 3:
            raise ValueError(f'model_probs must be 3D (B, depth, n_bins), got shape={tuple(model_probs.shape)}')
        if y_true.ndim != 1:
            y_true = y_true.view(-1)
        if model_probs.shape[0] != y_true.shape[0]:
            raise ValueError(f'batch mismatch: model_probs.shape[0]={model_probs.shape[0]} vs y_true.shape[0]={y_true.shape[0]}')
        device = model_probs.device
        y_true = y_true.to(device)
        leaf_probs = self.leaf_probs_from_step_probs(model_probs)
        n_bins = leaf_probs.shape[1]
        bin_values = torch.tensor([self.encoder.decode_bin_index(i) for i in range(n_bins)], dtype=torch.float32, device=device)
        y_pred_point = torch.sum(leaf_probs * bin_values, dim=1)
        mae = torch.mean(torch.abs(y_pred_point - y_true)).item()
        rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()
        interval = self.interval_from_leaf_probs(leaf_probs, confidence=confidence)
        L_pred, U_pred = (interval.L, interval.U)
        y_true_np = y_true.detach().cpu().numpy()
        L_pred_np = L_pred.detach().cpu().numpy()
        U_pred_np = U_pred.detach().cpu().numpy()
        y_pred_point_np = y_pred_point.detach().cpu().numpy()
        covered = (y_true_np >= L_pred_np) & (y_true_np <= U_pred_np)
        picp = float(np.mean(covered))
        widths = U_pred_np - L_pred_np
        mpiw = float(np.mean(widths))
        out: Dict = {'MAE': float(mae), 'RMSE': float(rmse), 'confidence': float(confidence), 'PICP': picp, 'MPIW': mpiw}
        if not return_extras:
            return out
        fixed_bins = self.width_bins
        bin_indices = np.digitize(widths, fixed_bins) - 1
        width_stratified_picp: Dict[str, str] = {}
        for i in range(len(fixed_bins) - 1):
            mask = bin_indices == i
            count = int(np.sum(mask))
            if count > 0:
                local_cov = float(np.mean(covered[mask]))
                range_str = f'Width [{fixed_bins[i]:.1f}, {fixed_bins[i + 1]:.1f})'
                width_stratified_picp[range_str] = f'{local_cov:.4f} (n={count})'
        true_bins_02 = np.digitize(y_true_np, self.bin_edges_02)
        pred_bins_02 = np.digitize(y_pred_point_np, self.bin_edges_02)
        bin_acc_02 = float(np.mean(true_bins_02 == pred_bins_02))
        true_bins_04 = np.digitize(y_true_np, self.bin_edges_04)
        pred_bins_04 = np.digitize(y_pred_point_np, self.bin_edges_04)
        bin_acc_04 = float(np.mean(true_bins_04 == pred_bins_04))
        out['width_stratified_PICP'] = width_stratified_picp
        out['bin_acc_0.2'] = bin_acc_02
        out['bin_acc_0.4'] = bin_acc_04
        return out
