from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tabseq.models.ft_encoder import FTEncoder

class DacaDecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor], gate: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        B, T, D = tgt.shape
        S = memory.shape[1]
        memory_gated = memory.unsqueeze(1) * gate.unsqueeze(0).unsqueeze(2)
        tgt_flat = tgt.reshape(B * T, 1, D)
        memory_key = memory_gated.reshape(B * T, S, D)
        memory_val = memory.unsqueeze(1).expand(B, T, S, D).reshape(B * T, S, D)
        cross_out, _ = self.cross_attn(tgt_flat, memory_key, memory_val)
        cross_out = cross_out.reshape(B, T, D)
        tgt = self.norm2(tgt + self.dropout2(cross_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))
        return tgt

class DacaDecoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DacaDecoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor], gate: torch.Tensor) -> torch.Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(out, memory=memory, tgt_mask=tgt_mask, gate=gate)
        return out

class TransformerTabSeqModel(nn.Module):

    def __init__(self, n_num_features: int, depth: int, n_bins: int, d_model: int=64, n_heads: int=4, n_layers: int=2, dropout: float=0.1, cat_cardinalities: Optional[List[int]]=None):
        super().__init__()
        self.depth = int(depth)
        self.n_bins = int(n_bins)
        self.tabular_encoder = FTEncoder(n_num_features=n_num_features, cat_cardinalities=cat_cardinalities, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.dec_emb = nn.Embedding(3, d_model)
        self.dec_pos_emb = nn.Embedding(self.depth, d_model)
        self.decoder = DacaDecoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(d_model, n_bins)
        tgt_mask = torch.triu(torch.full((self.depth, self.depth), float('-inf')), diagonal=1)
        self.register_buffer('_tgt_mask', tgt_mask, persistent=False)
        self.empty_ctx = nn.Parameter(torch.zeros(1, 1, d_model))
        self.daca_gate_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.confidence_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def compute_alpha_instance(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        ctx_summary = ctx_tokens.mean(dim=1)
        return torch.sigmoid(self.confidence_head(ctx_summary)).squeeze(-1)

    def forward(self, batch: dict, return_context: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_num = batch['x_num']
        dec_input = batch['dec_input']
        x_cat = batch.get('x_cat')
        ctx_tokens = self.tabular_encoder(x_num, x_cat)
        if ctx_tokens.shape[1] == 0:
            ctx_tokens = self.empty_ctx.expand(x_num.shape[0], 1, -1)
        seq_len = dec_input.shape[1]
        if seq_len > self.depth:
            raise ValueError(f'dec_input length {seq_len} exceeds depth {self.depth}')
        tok = self.dec_emb(dec_input)
        pos = torch.arange(seq_len, device=dec_input.device)
        pos_emb = self.dec_pos_emb(pos)
        tok = tok + pos_emb.unsqueeze(0)
        gate = torch.sigmoid(self.daca_gate_mlp(pos_emb))
        tgt_mask = self._tgt_mask[:seq_len, :seq_len].to(dtype=tok.dtype, device=tok.device)
        h_steps = self.decoder(tgt=tok, memory=ctx_tokens, tgt_mask=tgt_mask, gate=gate)
        logits = self.out(h_steps)
        if return_context:
            return (logits, ctx_tokens)
        return logits
