from __future__ import annotations  # 允许类型标注用“类名字符串”前置引用

from dataclasses import dataclass  # dataclass 用于声明简单结构体
from typing import Dict, Optional  # Dict 用于返回指标结构，Optional 表示可选参数

import numpy as np  # 数值计算（PICP/MPIW 分桶统计等）
import torch  # 计算图与张量运算（模型输出、区间计算）

from tabseq.labels.trace_encoder import TraceLabelEncoder  # 把叶子桶索引映射回连续值

# 术语速记（给新人/快速复盘用）：
# - bin（桶）：把连续 y 划分成离散区间后的“叶子桶”，索引范围是 [0, n_bins)。
# - depth（深度）：二叉决策树的深度；n_bins = 2^depth。
# - step_probs：每一步 t 对所有叶子桶的软约束概率（越像“走到该叶子”）。
# - leaf_probs：把所有 step_probs 相乘并归一化后的“叶子分布”。
# - CDF：累计分布函数，用来取分位点区间 [L, U]。
# - PICP：真实值落在区间内的比例（覆盖率）。
# - MPIW：区间平均宽度（越大越保守）。
# - width_bins：按“区间宽度”分桶的边界，仅用于诊断，不影响核心指标。


@dataclass(frozen=True)  # 冻结数据类：返回区间时不允许修改
class IntervalResult:
    L: torch.Tensor  # (B,) 区间下界，按 batch 对齐
    U: torch.Tensor  # (B,) 区间上界，按 batch 对齐


class ExtendedHolographicMetric:
    """
    统一评估口径（与 notebook 对齐）。

    输入：
      - model_probs: (B, depth, n_bins)，每一步对叶子桶的软约束概率
      - y_true     : (B,) 连续真实值

    输出核心指标（必须）：
      - MAE：平均绝对误差，衡量点预测偏差（越小越好）
      - RMSE：均方根误差，更强调大误差（越小越好）
      - PICP：区间覆盖率，期望接近 confidence（越高越可靠）
      - MPIW：区间平均宽度，表示区间“保守程度”（越小越尖锐）

    诊断指标（可选）：
      - width_stratified_PICP：按区间宽度分桶后的覆盖率，用于判断“窄区间是否系统性漏掉真值”
      - bin_acc_0.2 / bin_acc_0.4：把 y 与 y_hat 量化到固定 bin 后的粗粒度命中率

    设计原因（为什么这样写）：
      - TabSeq 的模型输出是“逐步约束”而非直接分布，所以需要先把 step_probs 合成为 leaf_probs。
      - 点预测用叶子分布的期望，区间预测用 CDF 分位点，便于与 notebook 对齐。
      - 诊断指标用于解释“覆盖率高但区间太宽”或“窄区间是否系统性漏真值”的问题。
    """

    def __init__(
        self,
        encoder: TraceLabelEncoder,  # 标签编码器：叶子桶索引 -> 连续值
        *,
        width_bins: Optional[list[float]] = None,  # 宽度分桶边界（诊断用）
        bin_edges_02: Optional[np.ndarray] = None,  # 0.2 粒度分桶边界（诊断用）
        bin_edges_04: Optional[np.ndarray] = None,  # 0.4 粒度分桶边界（诊断用）
    ):
        self.encoder = encoder  # 保存编码器，后续 decode bin index 用

        # 对齐 notebook 默认分桶（主要是 California Housing 的 y 范围）。
        # 记忆点：width_bins 表示“区间宽度的分档”，不是 y 的分档。
        # 例子：width=0.7 会被分到 [0.4, 0.8) 桶，用来统计该桶的覆盖率。
        self.width_bins = width_bins or [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0]
        # 0.2 粒度分桶：用于粗粒度命中率 bin_acc_0.2
        self.bin_edges_02 = bin_edges_02 if bin_edges_02 is not None else np.arange(0, 5.21, 0.2)
        # 0.4 粒度分桶：用于粗粒度命中率 bin_acc_0.4
        self.bin_edges_04 = bin_edges_04 if bin_edges_04 is not None else np.arange(0, 5.21, 0.4)

    def leaf_probs_from_step_probs(self, step_probs: torch.Tensor) -> torch.Tensor:
        """
        step_probs: (B, depth, n_bins)  # 每步对所有叶子桶的“软决策概率”
        return    : (B, n_bins)         # 合成后的叶子分布（按叶子归一化）
        """
        # 关键逻辑：每一步的独立概率相乘 -> 叶子桶的联合概率
        # 直觉：叶子桶对应一条“决策路径”，路径上每步的概率相乘表示“到达该叶子”的联合概率。
        leaf_unnorm_probs = torch.prod(step_probs, dim=1)  # (B, n_bins) 每个叶子桶的未归一化概率
        # 归一化：保证每个样本的叶子分布求和为 1
        leaf_probs = leaf_unnorm_probs / (torch.sum(leaf_unnorm_probs, dim=1, keepdim=True) + 1e-9)
        return leaf_probs  # (B, n_bins)

    def interval_from_leaf_probs(self, leaf_probs: torch.Tensor, *, confidence: float) -> IntervalResult:
        # 计算 CDF，再按分位点取上下界（与 notebook 行为对齐）。
        # 注意：这里没有“区间累加错误”，CDF 是标准的累计概率。
        # 额外提醒：叶子“单个 bin 的宽度”是固定的，但 [L, U] 可以跨多个 bin，
        # 因此“区间宽度”是可变的，不是一个常数。
        cdf = torch.cumsum(leaf_probs, dim=1)  # (B, n_bins) 累积概率
        alpha = 1.0 - float(confidence)  # 置信度的“尾部总量”
        lower_q = alpha / 2.0  # 下分位点（左尾）
        upper_q = 1.0 - (alpha / 2.0)  # 上分位点（右尾）

        # 取第一个满足 cdf >= q 的 index（与 notebook 的 argmax 逻辑一致）
        lower_indices = torch.argmax((cdf >= lower_q).int(), dim=1)  # (B,) 下界桶索引
        upper_indices = torch.argmax((cdf >= upper_q).int(), dim=1)  # (B,) 上界桶索引

        # 把“桶索引”映射回“桶中心对应的连续值”
        # 记忆点：这里 decode 的是“桶中心值”，不是边界值。
        bin_values = torch.tensor(
            [self.encoder.decode_bin_index(i) for i in range(leaf_probs.shape[1])],
            dtype=torch.float32,
            device=leaf_probs.device,
        )
        L = bin_values[lower_indices]  # (B,) 下界连续值
        U = bin_values[upper_indices]  # (B,) 上界连续值
        return IntervalResult(L=L, U=U)  # 返回区间结构体

    def compute_metrics(
        self,
        model_probs: torch.Tensor,  # (B, depth, n_bins) 模型每步的叶子概率
        y_true: torch.Tensor,  # (B,) 真实连续标签
        *,
        confidence: float = 0.90,  # 目标覆盖率，例如 0.90
        return_extras: bool = True,  # 是否返回诊断指标
    ) -> Dict:
        """
        计算统一指标。
        设计逻辑：先把 step_probs 合成叶子分布，再得到点预测与区间。
        """
        # 形状检查：必须是 (B, depth, n_bins)
        if model_probs.ndim != 3:
            raise ValueError(f"model_probs must be 3D (B, depth, n_bins), got shape={tuple(model_probs.shape)}")
        # y_true 必须是一维 (B,)
        if y_true.ndim != 1:
            y_true = y_true.view(-1)
        # batch 对齐检查：样本数必须一致
        if model_probs.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"batch mismatch: model_probs.shape[0]={model_probs.shape[0]} vs y_true.shape[0]={y_true.shape[0]}"
            )

        device = model_probs.device  # 与模型输出保持同设备
        y_true = y_true.to(device)  # 把真实值移到同设备

        # 1) step_probs -> leaf_probs（每一步概率相乘得到叶子分布）
        leaf_probs = self.leaf_probs_from_step_probs(model_probs)  # (B, n_bins)

        # 2) 点预测：对桶中心的期望（期望值 = 概率 * 取值）
        n_bins = leaf_probs.shape[1]  # 叶子桶数 = 2^depth
        bin_values = torch.tensor(
            [self.encoder.decode_bin_index(i) for i in range(n_bins)],
            dtype=torch.float32,
            device=device,
        )
        y_pred_point = torch.sum(leaf_probs * bin_values, dim=1)  # (B,) 点预测值
        mae = torch.mean(torch.abs(y_pred_point - y_true)).item()  # 平均绝对误差
        rmse = torch.sqrt(torch.mean((y_pred_point - y_true) ** 2)).item()  # 均方根误差

        # 3) 区间：对 leaf_probs 做 CDF，用分位点取 [L, U]
        interval = self.interval_from_leaf_probs(leaf_probs, confidence=confidence)  # 计算区间
        L_pred, U_pred = interval.L, interval.U  # 拆出上下界

        # 4) PICP/MPIW：
        #    - PICP：真实值落在区间内的比例（可靠性）
        #    - MPIW：区间平均宽度（尖锐度）
        y_true_np = y_true.detach().cpu().numpy()  # 转 numpy 以便统计
        L_pred_np = L_pred.detach().cpu().numpy()  # 区间下界 numpy
        U_pred_np = U_pred.detach().cpu().numpy()  # 区间上界 numpy
        y_pred_point_np = y_pred_point.detach().cpu().numpy()  # 点预测 numpy

        covered = (y_true_np >= L_pred_np) & (y_true_np <= U_pred_np)  # 是否被区间覆盖
        picp = float(np.mean(covered))  # 覆盖率 = 覆盖样本占比
        widths = U_pred_np - L_pred_np  # 区间宽度
        mpiw = float(np.mean(widths))  # 平均宽度

        out: Dict = {
            "MAE": float(mae),  # 点预测误差（越小越好）
            "RMSE": float(rmse),  # 点预测误差（更惩罚大误差）
            "confidence": float(confidence),  # 目标覆盖率（期望 PICP 接近它）
            "PICP": picp,  # 实际覆盖率
            "MPIW": mpiw,  # 区间平均宽度
        }

        if not return_extras:
            return out  # 若不需要诊断指标，直接返回核心指标

        # A) 按区间宽度分桶的覆盖率（诊断：窄区间是否更容易漏真值）
        # 记忆点：width_bins 是“区间宽度”的分桶，不是 y 的分桶。
        # 例如宽度=0.7 会被归到 [0.4, 0.8) 这一档。
        fixed_bins = self.width_bins  # 宽度分桶边界
        bin_indices = np.digitize(widths, fixed_bins) - 1  # 把宽度映射到桶索引
        width_stratified_picp: Dict[str, str] = {}  # 记录各宽度区间的覆盖率
        for i in range(len(fixed_bins) - 1):
            mask = bin_indices == i  # 选中“宽度在第 i 桶”的样本
            count = int(np.sum(mask))  # 该桶样本数
            if count > 0:
                local_cov = float(np.mean(covered[mask]))  # 该桶覆盖率
                range_str = f"Width [{fixed_bins[i]:.1f}, {fixed_bins[i + 1]:.1f})"
                width_stratified_picp[range_str] = f"{local_cov:.4f} (n={count})"

        # B) 分桶命中率（固定 bin 边界，用粗粒度检验点预测一致性）
        true_bins_02 = np.digitize(y_true_np, self.bin_edges_02)  # 真实值落在哪个 0.2 桶
        pred_bins_02 = np.digitize(y_pred_point_np, self.bin_edges_02)  # 预测值落在哪个 0.2 桶
        bin_acc_02 = float(np.mean(true_bins_02 == pred_bins_02))  # 0.2 粒度命中率

        true_bins_04 = np.digitize(y_true_np, self.bin_edges_04)  # 真实值落在哪个 0.4 桶
        pred_bins_04 = np.digitize(y_pred_point_np, self.bin_edges_04)  # 预测值落在哪个 0.4 桶
        bin_acc_04 = float(np.mean(true_bins_04 == pred_bins_04))  # 0.4 粒度命中率

        out["width_stratified_PICP"] = width_stratified_picp  # 宽度分桶覆盖率
        out["bin_acc_0.2"] = bin_acc_02  # 0.2 粒度命中率
        out["bin_acc_0.4"] = bin_acc_04  # 0.4 粒度命中率
        return out  # 返回完整指标字典
