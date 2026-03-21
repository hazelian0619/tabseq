#!/usr/bin/env python3
"""
Sweep peak-merge alpha on a fixed leaf_probs export.

核心逻辑（控制变量）:
- leaf_probs 固定（已由选定推理模式生成，如 beam K=16）
- 只改变区间提取方式（alpha 参数）

输出文件:
- summary.csv: alpha -> (PICP, MPIW, avg_mass, avg_bins) 整体指标
- width.csv: alpha x width_bin -> (coverage, n) 按宽度分桶的条件覆盖率
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def _interval_peak_merge(
    leaf_probs: np.ndarray, bin_edges: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    从每个样本的 bin 概率分布中提取预测区间 [L, U]

    输入:
    - leaf_probs: (N, n_bins) 每个样本每个 bin 的概率
    - bin_edges: (n_bins+1,) bin 的边界值（如 [0, 0.1, 0.2, ..., 10.0]）
    - alpha: 峰值合并阈值比例 (0,1]，越小区间越宽

    算法步骤:
    1. 找概率最高峰 (peak_idx)
    2. thresh = peak_prob * alpha（扩展阈值）
    3. 从峰左右连续扩展，直到概率 < thresh
    4. [L,U] = 扩展范围的 bin_edges 边界

    输出:
    - L, U: (N,) 预测区间的左右端点
    - mass: (N,) 区间内概率总和（质量，用于调试）
    - bins: (N,) 区间包含的 bin 数量
    """

    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    N, n_bins = leaf_probs.shape
    if bin_edges.shape[0] != n_bins + 1:
        raise ValueError(f"bin_edges length must be n_bins+1, got {bin_edges.shape[0]} vs {n_bins+1}")

    # 为每个样本预分配输出数组
    L = np.empty((N,), dtype=np.float32)
    U = np.empty((N,), dtype=np.float32)
    mass = np.empty((N,), dtype=np.float32)
    bins = np.empty((N,), dtype=np.int32)

    for i in range(N):
        probs = leaf_probs[i]# 当前样本的概率向量 (n_bins,)

        # 步骤1: 找最高峰
        peak_idx = int(np.argmax(probs))# 峰值索引
        peak_val = float(probs[peak_idx])# 峰值概率
        thresh = peak_val * float(alpha)# 步骤2: 计算扩展阈值

         # 步骤3: 从峰向左扩展（包含峰本身）
        left = peak_idx
        # 向右扩展
        right = peak_idx
        while left - 1 >= 0 and float(probs[left - 1]) >= thresh:
            left -= 1
        while right + 1 < n_bins and float(probs[right + 1]) >= thresh:
            right += 1

         # 步骤4: 输出区间边界、质量、bin数
        L[i] = float(bin_edges[left])# 左端：left bin 的左边界
        U[i] = float(bin_edges[right + 1])# 右端：right bin 的右边界
        mass[i] = float(np.sum(probs[left : right + 1]))# 区间质量
        bins[i] = int(right - left + 1)# 区间 bin 数量

    return L, U, mass, bins


def _width_stratified_picp(widths: np.ndarray, covered: np.ndarray, width_bins: List[float]) -> List[Dict[str, object]]:
    """
    按区间宽度分桶，计算每个桶的条件覆盖率

    输入:
    - widths: (N,) 每个区间的宽度 U-L
    - covered: (N,) 0/1 数组，y_true 是否在 [L,U] 内
    - width_bins: 如 [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0]

    算法:
    1. np.digitize 把每个 width 分配到桶索引
    2. 对每个桶：mask 选样本 → coverage = mean(covered[mask])

    输出: List of dicts，每行一个桶的统计
    """

    fixed_bins = list(map(float, width_bins))
    # digitize 返回 (1,2,3...)，-1 转为 0-based 桶索引

    idx = np.digitize(widths, fixed_bins) - 1# 把每个 width 映射到桶索引
    rows: List[Dict[str, object]] = []
    # 遍历每个桶区间 [fixed_bins[i], fixed_bins[i+1])

    for i in range(len(fixed_bins) - 1):
        mask = idx == i# 当前桶的样本掩码
        n = int(np.sum(mask))
        if n <= 0: # 桶内样本数
            continue
        cov = float(np.mean(covered[mask]))
        rows.append(
            {
                "width_lo": float(fixed_bins[i]),# 桶左边界
                "width_hi": float(fixed_bins[i + 1]),# 桶右边界
                "coverage": cov,# 条件覆盖率
                "n": n,# 样本数
            }
        )
    return rows


def main() -> None:
     # 命令行参数解析
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-probs-npz", required=True, help="npz exported by eval/eval_prefix_search (leaf_probs_val_*.npz)")
    ap.add_argument("--alphas", nargs="+", type=float, required=True, help="alpha list, e.g. 0.5 0.33 0.25 0.2")
    ap.add_argument("--y-source", choices=["clipped", "raw"], default="clipped")
    ap.add_argument(
        "--width-bins",
        nargs="+",
        type=float,
        default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0],  # 默认桶边界
        help="width bins for width_stratified_PICP",
    )
    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--out-width-csv", required=True)
    args = ap.parse_args()

    # 步骤1: 加载数据
    data = np.load(args.leaf_probs_npz, allow_pickle=True)
    leaf_probs = data["leaf_probs"].astype(np.float32) # (N, n_bins) 概率矩阵
    bin_edges = data["bin_edges"].astype(np.float32)# bin 边界
    y_true = data["y_clipped"].astype(np.float32) if args.y_source == "clipped" else data["y_raw"].astype(np.float32)

    # 强制归一化（防御性，每行概率和=1）
    leaf_probs = leaf_probs / (np.sum(leaf_probs, axis=1, keepdims=True) + 1e-9)

    # 加载元信息（可选，用于记录）
    meta_raw = data["meta"] if "meta" in data.files else None
    meta: Dict[str, object] = {}
    if meta_raw is not None:
        try:
            meta = json.loads(meta_raw.item())
        except Exception:
            meta = {"raw": str(meta_raw)}

    # 步骤2: 准备输出行列表
    summary_rows: List[Dict[str, object]] = []
    width_rows: List[Dict[str, object]] = []

    # 步骤3: 核心循环 - 每个 alpha 生成一次区间并评估
    for alpha in args.alphas:
        # 生成预测区间 [L,U]
        L, U, mass, bins = _interval_peak_merge(leaf_probs, bin_edges, float(alpha))
        
        # 计算 covered 掩码：y_true 是否落在 [L,U] 内
        covered = (y_true >= L) & (y_true <= U)# (N,) bool → 可直接 mean()
        widths = U - L# (N,) 区间宽度

         # 整体指标
        picp = float(np.mean(covered))# PICP = 覆盖率
        mpiw = float(np.mean(widths))# MPIW = 平均区间宽度
        summary_rows.append(
            {
                "alpha": float(alpha),
                "PICP": picp,
                "MPIW": mpiw,
                "interval_mass_mean": float(np.mean(mass)),# 平均概率质量
                "interval_bins_mean": float(np.mean(bins)),# 平均区间 bin 数
                "y_source": str(args.y_source),
                "leaf_probs_npz": str(args.leaf_probs_npz),
                "mode": str(meta.get("mode", "")),
                "beam_size": meta.get("beam_size", ""),
                "temperature": meta.get("temperature", ""),
            }
        )

        # 分桶指标：每个宽度桶的条件覆盖率
        for r in _width_stratified_picp(widths, covered, width_bins=list(args.width_bins)):
            width_rows.append(
                {
                    "alpha": float(alpha),
                    "width_lo": r["width_lo"],
                    "width_hi": r["width_hi"],
                    "coverage": r["coverage"],
                    "n": r["n"],
                }
            )

    os.makedirs(os.path.dirname(args.out_summary_csv) or ".", exist_ok=True)
    with open(args.out_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print("saved:", args.out_summary_csv)

    os.makedirs(os.path.dirname(args.out_width_csv) or ".", exist_ok=True)
    with open(args.out_width_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(width_rows[0].keys()))
        w.writeheader()
        w.writerows(width_rows)
    print("saved:", args.out_width_csv)


if __name__ == "__main__":
    main()

