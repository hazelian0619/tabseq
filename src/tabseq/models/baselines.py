"""
用途：提供“非 TabSeq-Trace”的回归基线模型（Baseline Zoo）。

设计意义：
- 用一组常见的表格回归模型（MLP / 分位数 MLP / FT-Transformer）作为对照组，便于衡量 TabSeq-Trace 的收益。
- 依赖尽量少：尽量复用本项目的 `FeatureTokenizer`，避免引入额外的表格深度学习框架。

使用场景：
- `scripts/baseline_suite.py` / `scripts/train.py` 等在跑 baseline 对比时会 import 这里的模型。

注意：
- 这些基线“直接回归 y 或分位数”，不实现 TabSeq-Trace 的 `y_seq/y_mht` 编码与解码。
- 因此它们主要用于点预测指标（MAE/RMSE）或区间指标（PICP/MPIW）的对照，而不是决策轨迹本身。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from tabseq.models.feature_tokenizer import FeatureTokenizer


def _build_mlp_trunk(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> tuple[nn.Sequential, int]:
    """
    构建 MLP 的“主干网络”（trunk，不含最终输出层）。

    - 输入：`x` 形状通常为 `[B, input_dim]`
    - 输出：一个 `nn.Sequential`（线性层 + ReLU + 可选 Dropout）以及 trunk 的最后维度 `out_dim`

    这样做的好处是：点预测（1 维输出）和分位数预测（Q 维输出）可以复用同一套 trunk。
    """
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for dim in hidden_dims:
        layers.append(nn.Linear(in_dim, int(dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(dim)
    return nn.Sequential(*layers), in_dim


class MLPRegressor(nn.Module):
    """
    MLP 点预测基线：直接输出一个标量回归值 `y_hat`。

    - 输入：`x` `[B, D]`
    - 输出：`y_hat` `[B]`
    """

    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 64), dropout: float = 0.1):
        super().__init__()
        trunk, out_dim = _build_mlp_trunk(input_dim, tuple(hidden_dims), float(dropout))
        self.net = nn.Sequential(trunk, nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QuantileMLP(nn.Module):
    """
    分位数回归的 MLP 基线：一次预测多个分位数（例如 0.1/0.5/0.9）。

    - 输入：`x` `[B, D]`
    - 输出：`q_hat` `[B, Q]`，Q = len(quantiles)，列顺序与 `quantiles` 一一对应

    常见用法：用 (q_low, q_high) 构造预测区间，从而计算 PICP/MPIW 等区间质量指标。
    """

    def __init__(
        self,
        input_dim: int,
        quantiles: Sequence[float],
        hidden_dims: Iterable[int] = (64, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        trunk, out_dim = _build_mlp_trunk(input_dim, tuple(hidden_dims), float(dropout))
        self.trunk = trunk
        self.head = nn.Linear(out_dim, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class FTTransformerRegressor(nn.Module):
    """
    FT-Transformer 风格的点预测基线（“特征 token + Transformer 编码 + CLS 池化”）。

    层间关系（从输入到输出）：
    - 数值/类别特征 -> `FeatureTokenizer`：把每个特征变成一个 token 向量
    - token 序列 -> `TransformerEncoder`：做特征间交互建模
    - 取 CLS token 表示 -> MLP/线性 head -> 标量回归值

    形状约定（常见）：
    - `x_num`: `[B, n_num_features]`
    - `x_cat`: `[B, n_cat_features]`（可选，没有类别特征则为 None）
    - tokenizer 输出 token 序列: `[B, T, d_model]`（T = 特征 token 数）
    """

    def __init__(
        self,
        n_num_features: int,
        *,
        cat_cardinalities: Optional[Sequence[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dims: Iterable[int] = (64,),
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            n_num_features=int(n_num_features),
            cat_cardinalities=list(cat_cardinalities or []),
            d_model=int(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=4 * int(d_model),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))

        head_dims = tuple(head_hidden_dims)
        if head_dims:
            trunk, out_dim = _build_mlp_trunk(int(d_model), head_dims, float(dropout))
            self.head = nn.Sequential(trunk, nn.Linear(out_dim, 1))
        else:
            self.head = nn.Linear(int(d_model), 1)

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        if tokens.shape[1] == 0:
            # 没有任何特征 token 时，Transformer/CLS 都失去意义；这里直接报错更早暴露数据管道问题。
            raise ValueError("FTTransformerRegressor 至少需要一个特征 token（请检查数值/类别特征是否为空）")
        # 在序列头部拼一个可学习的 CLS token，让模型把“全局信息”聚合到这一位置。
        cls = self.cls_token.expand(tokens.shape[0], 1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        enc = self.encoder(seq)
        pooled = enc[:, 0, :]
        return self.head(pooled).squeeze(-1)


# 向后兼容：旧配置/旧 checkpoint 里可能还在用这个类名。
class TabularTransformerRegressor(FTTransformerRegressor):
    """历史别名：兼容旧配置/旧 checkpoint 中的类名。"""
    pass
