from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from tabseq.models.feature_tokenizer import FeatureTokenizer


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        *,
        cat_cardinalities: Optional[Sequence[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            n_num_features=int(n_num_features),
            cat_cardinalities=cat_cardinalities,
            d_model=int(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=4 * int(d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(tokens.shape[0], 1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        return self.norm(self.encoder(seq))
