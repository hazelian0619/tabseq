from typing import List, Optional

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Tokenize tabular features into per-feature tokens.
    - Numeric feature i: token = x_i * w_i + b_i
    - Categorical feature j: embedding lookup
    """

    def __init__(
        self,
        n_num_features: int,
        *,
        cat_cardinalities: Optional[List[int]] = None,
        d_model: int = 64,
    ):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.n_cat_features = len(self.cat_cardinalities)
        self.d_model = int(d_model)

        if self.n_num_features > 0:
            self.num_weight = nn.Parameter(torch.empty(self.n_num_features, self.d_model))
            self.num_bias = nn.Parameter(torch.zeros(self.n_num_features, self.d_model))
            nn.init.xavier_uniform_(self.num_weight)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embs = nn.ModuleList(
            [nn.Embedding(cardinality, self.d_model) for cardinality in self.cat_cardinalities]
        )

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + self.n_cat_features

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []

        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required when n_num_features > 0")
            if x_num.shape[1] != self.n_num_features:
                raise ValueError(
                    f"x_num has {x_num.shape[1]} features, expected {self.n_num_features}"
                )
            x_num = x_num.to(self.num_weight.dtype)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            tokens.append(num_tokens)

        if self.n_cat_features > 0:
            if x_cat is None:
                raise ValueError("x_cat is required when cat_cardinalities is provided")
            if x_cat.shape[1] != self.n_cat_features:
                raise ValueError(
                    f"x_cat has {x_cat.shape[1]} features, expected {self.n_cat_features}"
                )
            x_cat = x_cat.long()
            cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
            tokens.append(torch.stack(cat_tokens, dim=1))

        if not tokens:
            if x_num is None and x_cat is None:
                raise ValueError("x_num or x_cat must be provided")
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            device = x_num.device if x_num is not None else x_cat.device
            dtype = self.num_weight.dtype if self.num_weight is not None else torch.float32
            return torch.empty((batch_size, 0, self.d_model), device=device, dtype=dtype)

        return torch.cat(tokens, dim=1)
