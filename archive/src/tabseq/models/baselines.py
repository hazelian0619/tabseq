"""
基线模型集合（M3 对比用）。
注意：这些模型是“对比组”，不是 TabSeq 主线模型。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from tabseq.models.feature_tokenizer import FeatureTokenizer

def _build_mlp_features(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> tuple[nn.Sequential, int]:
    # 构建一个简单 MLP：Linear + ReLU (+ Dropout)
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for dim in hidden_dims:
        layers.append(nn.Linear(in_dim, int(dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = int(dim)
    return nn.Sequential(*layers), in_dim


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 64), dropout: float = 0.1):
        super().__init__()
        # 普通回归 MLP：直接输出点预测 y_hat
        trunk, out_dim = _build_mlp_features(input_dim, tuple(hidden_dims), float(dropout))
        self.net = nn.Sequential(trunk, nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QuantileMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        quantiles: Sequence[float],
        hidden_dims: Iterable[int] = (64, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        # 分位数 MLP：同时输出多个分位点（区间上下界）
        self.quantiles = tuple(float(q) for q in quantiles)
        trunk, out_dim = _build_mlp_features(input_dim, tuple(hidden_dims), float(dropout))
        self.trunk = trunk
        self.head = nn.Linear(out_dim, len(self.quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return self.head(h)


class FTTransformerRegressor(nn.Module):
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
        # 1) FeatureTokenizer：把每一列特征变成 token
        self.tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=list(cat_cardinalities or []),
            d_model=d_model,
        )
        # 2) TransformerEncoder：让特征 token 之间互相注意（特征交互）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # 3) CLS token：作为全局汇聚向量（FT-Transformer 标准做法）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        head_dims = tuple(head_hidden_dims)
        if head_dims:
            # 4) 输出头：MLP + Linear，输出点预测
            trunk, out_dim = _build_mlp_features(d_model, head_dims, float(dropout))
            self.head = nn.Sequential(trunk, nn.Linear(out_dim, 1))
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tokens: (B, n_tokens, d_model)
        tokens = self.tokenizer(x_num, x_cat)
        if tokens.shape[1] == 0:
            raise ValueError("TabularTransformerRegressor requires at least one feature token")
        # seq: (B, 1 + n_tokens, d_model)，第 0 位是 CLS
        cls = self.cls_token.expand(tokens.shape[0], 1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        # enc: (B, 1 + n_tokens, d_model)，取 CLS 位置作为全局表征
        enc = self.encoder(seq)
        pooled = enc[:, 0, :]
        return self.head(pooled).squeeze(-1)


# Backward-compatible alias for old checkpoints/configs.
class TabularTransformerRegressor(FTTransformerRegressor):
    pass
