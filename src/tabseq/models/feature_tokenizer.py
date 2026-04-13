from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Turn tabular columns into a token sequence.

    Numeric feature i becomes:
      x_i * W_i + b_i
    Categorical feature j becomes:
      Embedding[offset_j + x_j]
    """

    def __init__(self, n_num_features: int, cat_cardinalities: Optional[Sequence[int]] = None, d_model: int = 64):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = tuple(int(v) for v in (cat_cardinalities or ()))
        self.d_model = int(d_model)

        if self.n_num_features > 0:
            self.num_weight = nn.Parameter(torch.empty(self.n_num_features, self.d_model))
            self.num_bias = nn.Parameter(torch.zeros(self.n_num_features, self.d_model))
            nn.init.normal_(self.num_weight, mean=0.0, std=0.02)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        if self.cat_cardinalities:
            total_categories = int(sum(self.cat_cardinalities))
            offsets = np.cumsum([0, *self.cat_cardinalities[:-1]], dtype=np.int64)
            self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
            self.cat_embedding = nn.Embedding(total_categories, self.d_model)
            nn.init.normal_(self.cat_embedding.weight, mean=0.0, std=0.02)
        else:
            self.register_buffer("cat_offsets", torch.zeros(0, dtype=torch.long), persistent=False)
            self.cat_embedding = None

    def _batch_size(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> int:
        if x_num is not None:
            return int(x_num.shape[0])
        if x_cat is not None:
            return int(x_cat.shape[0])
        raise ValueError("at least one of x_num or x_cat must be provided")

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = self._batch_size(x_num, x_cat)
        device = x_num.device if x_num is not None else x_cat.device  # type: ignore[union-attr]

        tokens = []

        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required when n_num_features > 0")
            if x_num.ndim != 2 or x_num.shape[1] != self.n_num_features:
                raise ValueError(
                    f"x_num must have shape [B, {self.n_num_features}], got {tuple(x_num.shape)}"
                )
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        if self.cat_embedding is not None:
            if x_cat is None:
                raise ValueError("x_cat is required when cat_cardinalities is not empty")
            if x_cat.ndim != 2 or x_cat.shape[1] != len(self.cat_cardinalities):
                raise ValueError(
                    f"x_cat must have shape [B, {len(self.cat_cardinalities)}], got {tuple(x_cat.shape)}"
                )
            x_cat = x_cat.long().clamp_min(0)
            cat_tokens = self.cat_embedding(x_cat + self.cat_offsets.unsqueeze(0))
            tokens.append(cat_tokens)

        if not tokens:
            return torch.empty((batch_size, 0, self.d_model), dtype=torch.float32, device=device)
        return torch.cat(tokens, dim=1)
