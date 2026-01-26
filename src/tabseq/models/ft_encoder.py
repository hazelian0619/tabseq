from typing import List, Optional
import torch
import torch.nn as nn
from tabseq.models.feature_tokenizer import FeatureTokenizer

class FTEncoder(nn.Module):

    def __init__(self, n_num_features: int, *, cat_cardinalities: Optional[List[int]]=None, d_model: int=64, n_heads: int=4, n_layers: int=2, dropout: float=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features=n_num_features, cat_cardinalities=cat_cardinalities, d_model=d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    @property
    def n_tokens(self) -> int:
        return self.tokenizer.n_tokens

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        if tokens.shape[1] == 0:
            return tokens
        return self.encoder(tokens)
