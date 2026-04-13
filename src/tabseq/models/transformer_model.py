from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn

from tabseq.models.ft_encoder import FTTransformerEncoder as VanillaTransformerEncoder
from tabseq.models.ft_transformer_encoder import FTTransformerSeqEncoder


class TransformerTabSeqModel(nn.Module):
    """
    Minimal Transformer-based TabSeq model.

    Input:
      - x_num: [B, D_num]
      - x_cat: [B, D_cat] or None
      - dec_input: [B, depth] with values {0, 1, SOS}

    Output:
      - mht_logits: [B, depth, n_bins]
      - bit_logits: [B, depth]
      - leaf_logits: [B, n_bins]
    """

    def __init__(
        self,
        n_num_features: int,
        depth: int,
        n_bins: int,
        *,
        cat_cardinalities: Optional[Sequence[int]] = None,
        encoder_type: str = "vanilla",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        sos_token: int = 2,
    ):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.depth = int(depth)
        self.n_bins = int(n_bins)
        self.sos_token = int(sos_token)
        self.d_model = int(d_model)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "vanilla":
            self.tabular_encoder = VanillaTransformerEncoder(
                n_num_features=self.n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_model=self.d_model,
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                dropout=float(dropout),
            )
        elif self.encoder_type == "ft_transformer":
            self.tabular_encoder = FTTransformerSeqEncoder(
                n_num_features=self.n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_model=self.d_model,
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                dropout=float(dropout),
            )
        else:
            raise ValueError(f"unknown encoder_type={self.encoder_type!r}, expected 'vanilla' or 'ft_transformer'")
        self.dec_embedding = nn.Embedding(self.sos_token + 1, self.d_model)
        self.dec_pos_embedding = nn.Parameter(torch.zeros(1, self.depth, self.d_model))
        self.dropout = nn.Dropout(float(dropout))

        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(n_layers))
        self.mht_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_bins),
        )
        self.bit_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )
        self.leaf_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_bins),
        )

        nn.init.normal_(self.dec_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dec_pos_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((length, length), dtype=torch.float32, device=device)
        upper = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
        return mask.masked_fill(upper, float("-inf"))

    def _unpack_inputs(
        self,
        batch: Optional[Mapping[str, torch.Tensor]],
        *,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        dec_input: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if batch is not None:
            x_num = batch.get("x_num") if x_num is None else x_num
            x_cat = batch.get("x_cat") if x_cat is None else x_cat
            dec_input = batch.get("dec_input") if dec_input is None else dec_input

        if x_num is None and x_cat is None:
            raise ValueError("at least one of x_num or x_cat must be provided")

        if x_num is None:
            batch_size = int(x_cat.shape[0])  # type: ignore[union-attr]
            device = x_cat.device  # type: ignore[union-attr]
            x_num = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        else:
            x_num = x_num.float()

        if x_cat is not None and x_cat.ndim == 2 and x_cat.shape[1] == 0:
            x_cat = None

        if dec_input is None:
            dec_input = torch.full(
                (x_num.shape[0], self.depth),
                fill_value=self.sos_token,
                dtype=torch.long,
                device=x_num.device,
            )
        else:
            dec_input = dec_input.long()

        if dec_input.ndim != 2 or dec_input.shape[1] != self.depth:
            raise ValueError(f"dec_input must have shape [B, {self.depth}], got {tuple(dec_input.shape)}")

        return x_num, x_cat, dec_input

    def forward(
        self,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
        *,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x_num, x_cat, dec_input = self._unpack_inputs(batch, x_num=x_num, x_cat=x_cat, dec_input=dec_input)
        memory = self.tabular_encoder(x_num, x_cat)

        tgt = self.dec_embedding(dec_input) + self.dec_pos_embedding[:, : dec_input.shape[1], :]
        tgt = self.dropout(tgt + memory[:, :1, :])
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=self._causal_mask(dec_input.shape[1], dec_input.device),
        )
        mht_logits = self.mht_head(decoded)
        bit_logits = self.bit_head(decoded).squeeze(-1)
        leaf_logits = self.leaf_head(memory[:, 0, :])
        return {
            "mht_logits": mht_logits,
            "bit_logits": bit_logits,
            "leaf_logits": leaf_logits,
        }
