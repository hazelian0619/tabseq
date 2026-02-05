from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增：用于 LayerNorm


class FeatureTokenizer(nn.Module):
    """
    Tokenize tabular features into per-feature tokens (工业级优化版).
    - Numeric: x_i * w_i + b_i + LayerNorm (数值稳定)
    - Categorical: embedding lookup + OOV 保护
    """
    def __init__(
        self,
        n_num_features: int,
        *,
        cat_cardinalities: Optional[List[int]] = None,
        d_model: int = 64,
        num_bias_init_std: float = 1e-4,  # 新增：bias 随机初始化
    ):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.n_cat_features = len(self.cat_cardinalities)
        self.d_model = int(d_model)

        # 1️⃣ 数值特征线性变换（核心）
        if self.n_num_features > 0:
            self.num_weight = nn.Parameter(torch.empty(self.n_num_features, self.d_model))
            self.num_bias = nn.Parameter(
                torch.normal(0, num_bias_init_std, size=(self.n_num_features, self.d_model))
            )  # ✅ 优化3：随机初始化，打破对称性
            nn.init.xavier_uniform_(self.num_weight)
            
            # ✅ 优化1：LayerNorm 数值稳定层（Critical）
            self.num_ln = nn.LayerNorm(d_model)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)
            self.num_ln = None

        # 2️⃣ 类别特征 Embedding
        self.cat_embs = nn.ModuleList([
            nn.Embedding(cardinality + 1, self.d_model)  # ✅ 优化2：+1 预留 <UNK>
            for cardinality in self.cat_cardinalities
        ])

    @property
    def n_tokens(self) -> int:
        return self.n_num_features + self.n_cat_features

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []

        # 处理数值特征
        if self.n_num_features > 0:
            if x_num is None:
                raise ValueError("x_num is required when n_num_features > 0")
            if x_num.shape[1] != self.n_num_features:
                raise ValueError(f"x_num has {x_num.shape[1]} features, expected {self.n_num_features}")
            
            x_num = x_num.to(self.num_weight.dtype)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            num_tokens = self.num_ln(num_tokens)  # ✅ LayerNorm 稳定输出
            tokens.append(num_tokens)

        # 处理类别特征（✅ 工业级 OOV 保护）
        if self.n_cat_features > 0:
            if x_cat is None:
                raise ValueError("x_cat is required when cat_cardinalities is provided")
            if x_cat.shape[1] != self.n_cat_features:
                raise ValueError(f"x_cat has {x_cat.shape[1]} features, expected {self.n_cat_features}")
            
            x_cat = x_cat.long()
            cat_tokens = []
            for i, emb in enumerate(self.cat_embs):
                cardinality = self.cat_cardinalities[i]  # 原始 cardinality
                # ✅ OOV 保护：clamp 到 [0, cardinality]，cardinality 映射到 <UNK>
                input_ids = x_cat[:, i].clamp(0, cardinality)
                cat_tokens.append(emb(input_ids))
            tokens.append(torch.stack(cat_tokens, dim=1))

        # 空输入处理
        if not tokens:
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            device = x_num.device if x_num is not None else x_cat.device
            dtype = self.num_weight.dtype if self.num_weight is not None else torch.float32
            return torch.empty((batch_size, 0, self.d_model), device=device, dtype=dtype)

        return torch.cat(tokens, dim=1)
