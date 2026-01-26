"""
FTEncoder（主线模型的表格编码器）。
作用：把表格特征编码成“token 序列”，供 DACA 解码器读取。
注意：这里只是“编码器”，不是完整的 FT‑Transformer baseline（没有 CLS pooling）。

逻辑链路：
  x_num/x_cat -> FeatureTokenizer -> tokens -> TransformerEncoder -> ctx_tokens
"""

from typing import List, Optional  # 类型标注
import torch  # 张量计算
import torch.nn as nn  # PyTorch 模型组件
from tabseq.models.feature_tokenizer import FeatureTokenizer  # 特征分词器（列→token）


class FTEncoder(nn.Module):
    """
    FT-Transformer 风格的编码器（不含 CLS）。
    设计目的：用“特征 token + TransformerEncoder”建模特征交互。
    关键配置：PreNorm + GELU（与 FT 论文风格保持一致）。
    """
    def __init__(
        self,
        n_num_features: int,
        *,
        cat_cardinalities: Optional[List[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1) FeatureTokenizer：把每一列特征变成一个 token（维度=d_model）
        self.tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
        )

        # 2) TransformerEncoder：让不同 token 之间互相注意（特征交互）
        #    - GELU：FT 风格激活
        #    - PreNorm：更稳定（norm_first=True）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,  # FFN 隐层大小（常用 4x）
            dropout=dropout,
            activation="gelu",            # FT 风格激活
            batch_first=True,
            norm_first=True,              # PreNorm（更稳定）
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
    
    @property
    def n_tokens(self) -> int:
        # token 数 = 数值特征数 + 类别特征数
        return self.tokenizer.n_tokens
    
    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        # 1) 列特征 -> token 序列
        tokens = self.tokenizer(x_num, x_cat)
        if tokens.shape[1] == 0:
            # 没有任何特征时返回空序列
            return tokens
        # 2) token 序列 -> 编码后的上下文 tokens
        return self.encoder(tokens)
