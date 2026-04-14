from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"ReGLU expects an even last dimension, got {tuple(x.shape)}")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class _NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.weight = nn.Parameter(torch.empty(self.n_features, self.d_model))
        self.bias = nn.Parameter(torch.empty(self.n_features, self.d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.d_model ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if x_num.ndim != 2 or x_num.shape[1] != self.n_features:
            raise ValueError(f"x_num must have shape [B, {self.n_features}], got {tuple(x_num.shape)}")
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, cardinalities: Sequence[int], d_model: int) -> None:
        super().__init__()
        self.cardinalities = tuple(int(x) for x in cardinalities)
        self.d_model = int(d_model)
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, self.d_model) for cardinality in self.cardinalities])
        self.bias = nn.Parameter(torch.empty(len(self.cardinalities), self.d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.d_model ** -0.5
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.ndim != 2 or x_cat.shape[1] != len(self.embeddings):
            raise ValueError(f"x_cat must have shape [B, {len(self.embeddings)}], got {tuple(x_cat.shape)}")
        x_cat = x_cat.long().clamp_min(0)
        tokens = torch.stack([embedding(x_cat[:, idx]) for idx, embedding in enumerate(self.embeddings)], dim=1)
        return tokens + self.bias.unsqueeze(0)


class _FTFeatureTokenizer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: Optional[Sequence[int]], d_model: int) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = tuple(int(x) for x in (cat_cardinalities or ()))
        self.d_model = int(d_model)

        self.num_tokenizer = (
            _NumericalFeatureTokenizer(self.n_num_features, self.d_model) if self.n_num_features > 0 else None
        )
        self.cat_tokenizer = (
            _CategoricalFeatureTokenizer(self.cat_cardinalities, self.d_model) if self.cat_cardinalities else None
        )

    @staticmethod
    def _batch_size(x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> int:
        if x_num is not None:
            return int(x_num.shape[0])
        if x_cat is not None:
            return int(x_cat.shape[0])
        raise ValueError("at least one of x_num or x_cat must be provided")

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._batch_size(x_num, x_cat)
        device = x_num.device if x_num is not None else x_cat.device  # type: ignore[union-attr]
        tokens = []

        if self.num_tokenizer is not None:
            if x_num is None:
                raise ValueError("x_num is required when n_num_features > 0")
            tokens.append(self.num_tokenizer(x_num.float()))

        if self.cat_tokenizer is not None:
            if x_cat is None:
                raise ValueError("x_cat is required when cat_cardinalities is not empty")
            tokens.append(self.cat_tokenizer(x_cat))

        if not tokens:
            return torch.empty((batch_size, 0, self.d_model), dtype=torch.float32, device=device)
        return torch.cat(tokens, dim=1)


class _CLSToken(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(d_model)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[0] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.view(1, 1, -1).expand(batch_size, 1, -1)


class _FTTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        ffn_d_hidden_multiplier: float,
        skip_attention_norm: bool,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        d_hidden = int(round(d_model * float(ffn_d_hidden_multiplier)))
        if d_hidden <= 0:
            raise ValueError(f"ffn hidden size must be positive, got {d_hidden}")

        self.attention_norm = None if skip_attention_norm else nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(n_heads),
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.attention_residual_dropout = nn.Dropout(float(residual_dropout))

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_linear1 = nn.Linear(d_model, 2 * d_hidden)
        self.ffn_activation = _ReGLU()
        self.ffn_dropout = nn.Dropout(float(ffn_dropout))
        self.ffn_linear2 = nn.Linear(d_hidden, d_model)
        self.ffn_residual_dropout = nn.Dropout(float(residual_dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = x if self.attention_norm is None else self.attention_norm(x)
        attention_output, _ = self.attention(attention_input, attention_input, attention_input, need_weights=False)
        x = x + self.attention_residual_dropout(attention_output)

        ffn_output = self.ffn_linear1(self.ffn_norm(x))
        ffn_output = self.ffn_activation(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_linear2(ffn_output)
        x = x + self.ffn_residual_dropout(ffn_output)
        return x


class FTTransformerSeqEncoder(nn.Module):
    """
    FT-Transformer-style tabular encoder that returns token-sequence memory.

    The backbone follows the FT-Transformer design more closely than the old
    encoder: feature tokenization + CLS token + PreNorm blocks + ReGLU FFN.
    Unlike the original regressor head, this module keeps the full token
    sequence so the TabSeq decoder can cross-attend to it.
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
        attention_dropout: Optional[float] = None,
        ffn_dropout: Optional[float] = None,
        residual_dropout: float = 0.0,
        ffn_d_hidden_multiplier: float = 4.0 / 3.0,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        n_heads = int(n_heads)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={d_model}, n_heads={n_heads}")

        self.tokenizer = _FTFeatureTokenizer(
            n_num_features=int(n_num_features),
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
        )
        self.cls_token = _CLSToken(d_model)
        self.blocks = nn.ModuleList(
            [
                _FTTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attention_dropout=float(dropout if attention_dropout is None else attention_dropout),
                    ffn_dropout=float(dropout if ffn_dropout is None else ffn_dropout),
                    residual_dropout=float(residual_dropout),
                    ffn_d_hidden_multiplier=float(ffn_d_hidden_multiplier),
                    skip_attention_norm=(layer_idx == 0),
                )
                for layer_idx in range(int(n_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token(int(tokens.shape[0]))
        x = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
