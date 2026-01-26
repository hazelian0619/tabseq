"""
TabSeq 主模型（DACA + ACM 对齐 PDF 版本）。

核心逻辑（主线记忆）：
1) x_num/x_cat -> FTEncoder -> ctx_tokens（表格上下文）
2) dec_input -> DACA Decoder -> logits (B, depth, n_bins)
3) 训练时用 y_mht 做多热监督；评估时 logits -> 概率 -> 指标

注意：
- 这里输出的是 logits（原始打分），不是 MAE/RMSE/PICP/MPIW 这些指标；
- 指标计算在 eval.py / metrics 里完成。
"""

from typing import List, Optional, Tuple, Union  # 类型标注（增强可读性）

import torch  # 张量与计算
import torch.nn as nn  # PyTorch 模型组件

from tabseq.models.ft_encoder import FTEncoder  # 表格编码器（输出 token 序列）


class DacaDecoderLayer(nn.Module):
    """
    DACA 解码层：
    - 自回归 self-attention（因果掩码）
    - cross-attention 读取 ctx_tokens
    - gate 只作用于 Key（与 PDF 的 K_t = X_ctx ⊙ G_t 对齐）
    """

    def __init__(
        self,
        d_model: int,  # token 维度
        n_heads: int,  # 多头注意力头数
        dropout: float,  # dropout 比例
    ):
        super().__init__()  # 初始化父类
        self.self_attn = nn.MultiheadAttention(  # 自注意力层
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(  # 交叉注意力层
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # 前馈网络（FFN）：两层 Linear + 激活
        self.linear1 = nn.Linear(d_model, 4 * d_model)  # 扩展维度
        self.linear2 = nn.Linear(4 * d_model, d_model)  # 回到 d_model
        self.dropout = nn.Dropout(dropout)  # FFN dropout
        self.dropout1 = nn.Dropout(dropout)  # self-attn 后 dropout
        self.dropout2 = nn.Dropout(dropout)  # cross-attn 后 dropout
        self.dropout3 = nn.Dropout(dropout)  # FFN 后 dropout
        self.norm1 = nn.LayerNorm(d_model)  # self-attn 残差归一化
        self.norm2 = nn.LayerNorm(d_model)  # cross-attn 残差归一化
        self.norm3 = nn.LayerNorm(d_model)  # FFN 残差归一化
        self.activation = nn.ReLU()  # 非线性激活

    def forward(
        self,
        tgt: torch.Tensor,  # (B, T, D) 解码端 token
        memory: torch.Tensor,  # (B, S, D) 上下文 token
        tgt_mask: Optional[torch.Tensor],  # (T, T) 因果 mask
        gate: torch.Tensor,  # (T, D) 每一步的门控向量
    ) -> torch.Tensor:
        # 1) Self-attention（因果）：t 只能看见 <=t 的 token
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))  # 残差 + 归一化

        # 2) DACA cross-attention（按步门控 Key）
        # gate: (T, D)，memory: (B, S, D)
        # 目标：每个解码步 t 都有自己的门控 G_t，只作用在 Key 上
        B, T, D = tgt.shape  # B=batch, T=序列长度, D=维度
        S = memory.shape[1]  # S=上下文 token 数量
        memory_gated = memory.unsqueeze(1) * gate.unsqueeze(0).unsqueeze(2)  # (B, T, S, D)
        tgt_flat = tgt.reshape(B * T, 1, D)  # 把每个步展开成单独 query
        memory_key = memory_gated.reshape(B * T, S, D)  # 每步对应一份 gated key
        # Value 不做 gate（严格对齐 PDF 的 K_t 公式）
        memory_val = memory.unsqueeze(1).expand(B, T, S, D).reshape(B * T, S, D)
        cross_out, _ = self.cross_attn(tgt_flat, memory_key, memory_val)  # cross-attn
        cross_out = cross_out.reshape(B, T, D)  # reshape 回 (B, T, D)
        tgt = self.norm2(tgt + self.dropout2(cross_out))  # 残差 + 归一化

        # 3) Feed-forward（逐 token 非线性变换）
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))  # 残差 + 归一化
        return tgt  # 返回更新后的解码 token


class DacaDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()  # 初始化父类
        # 堆叠多个 DacaDecoderLayer
        self.layers = nn.ModuleList(
            [DacaDecoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,  # (B, T, D) 解码端 token
        memory: torch.Tensor,  # (B, S, D) 上下文 token
        tgt_mask: Optional[torch.Tensor],  # (T, T) 因果 mask
        gate: torch.Tensor,  # (T, D) 门控向量
    ) -> torch.Tensor:
        out = tgt  # 初始解码状态
        for layer in self.layers:  # 逐层更新
            out = layer(out, memory=memory, tgt_mask=tgt_mask, gate=gate)
        return out  # 返回最终解码结果


class TransformerTabSeqModel(nn.Module):
    """
    主线模型（DACA + ACM）：
    - 表格侧：FeatureTokenizer + FTEncoder 得到 ctx_tokens
    - 序列侧：dec_input 自回归解码（带因果掩码）
    - 输出：每一步对所有叶子桶的 logits

    输入字段：
      - x_num: (B, F) 数值特征
      - x_cat: (B, C) 类别特征（可为空）
      - dec_input: (B, depth) 训练前缀（[SOS] + y_seq[:-1]）

    输出：
      - logits: (B, depth, n_bins)
      - 可选 ctx_tokens（用于 ACM）
    """

    def __init__(
        self,
        n_num_features: int,  # 数值特征列数
        depth: int,  # 决策序列长度
        n_bins: int,  # 桶数量 = 2^depth
        d_model: int = 64,  # token 维度
        n_heads: int = 4,  # 注意力头数
        n_layers: int = 2,  # 编码/解码层数
        dropout: float = 0.1,  # dropout 比例
        cat_cardinalities: Optional[List[int]] = None,  # 类别特征基数
    ):
        super().__init__()  # 初始化父类
        self.depth = int(depth)  # 保存深度
        self.n_bins = int(n_bins)  # 保存桶数

        # 1) FT encoder -> context tokens (token-level, no CLS pooling)
        self.tabular_encoder = FTEncoder(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # 2) dec_input token embedding: 0/1/SOS(=2), vocab size=3
        self.dec_emb = nn.Embedding(3, d_model)  # 输入 token embedding
        self.dec_pos_emb = nn.Embedding(self.depth, d_model)  # 位置编码

        # 3) DACA decoder：对 ctx_tokens 做按步门控
        self.decoder = DacaDecoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)

        # 4) 输出头：每一步 -> n_bins 个 logits
        self.out = nn.Linear(d_model, n_bins)

        # 因果 mask：防止 t 看到 t+1 之后的 token
        tgt_mask = torch.triu(torch.full((self.depth, self.depth), float("-inf")), diagonal=1)
        self.register_buffer("_tgt_mask", tgt_mask, persistent=False)  # 不保存到 checkpoint

        # 如果没有任何特征 token，用一个可学习的空上下文代替
        self.empty_ctx = nn.Parameter(torch.zeros(1, 1, d_model))
        # DACA 门控：G_t = sigmoid(MLP(E_pos(t)))
        self.daca_gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # ACM 的 alpha_instance 预测头（训练阶段使用）
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def compute_alpha_instance(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        # 用上下文 token 的平均池化作为样本表征
        ctx_summary = ctx_tokens.mean(dim=1)
        # 输出范围 (0,1) 的样本置信权重
        return torch.sigmoid(self.confidence_head(ctx_summary)).squeeze(-1)

    def forward(self, batch: dict, return_context: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_num = batch["x_num"]          # (B, F) 数值特征
        dec_input = batch["dec_input"]  # (B, depth) 解码前缀

        x_cat = batch.get("x_cat")  # (B, C) 类别特征（可为空）
        # 1) 表格编码：x_num/x_cat -> ctx_tokens
        ctx_tokens = self.tabular_encoder(x_num, x_cat)  # (B, n_ctx, d_model)
        if ctx_tokens.shape[1] == 0:  # 若没有特征 token
            ctx_tokens = self.empty_ctx.expand(x_num.shape[0], 1, -1)  # 用空上下文替代

        # 2) 序列编码：dec_input -> token + 位置编码
        seq_len = dec_input.shape[1]  # 当前序列长度
        if seq_len > self.depth:  # 防御：长度超过深度
            raise ValueError(f"dec_input length {seq_len} exceeds depth {self.depth}")

        tok = self.dec_emb(dec_input)  # (B, depth, d_model) token embedding
        pos = torch.arange(seq_len, device=dec_input.device)  # 位置索引
        pos_emb = self.dec_pos_emb(pos)  # 位置 embedding
        tok = tok + pos_emb.unsqueeze(0)  # 加到 token 上

        # 3) DACA gate: G_t = sigmoid(MLP(E_pos(t)))
        # NOTE: paper mentions early/late depth intuition (e.g., t<=3 / t>=8) but does not hard-code thresholds.
        gate = torch.sigmoid(self.daca_gate_mlp(pos_emb))  # (T, d_model)

        # 4) 解码：自回归 self-attention + cross-attention(ctx_tokens)
        tgt_mask = self._tgt_mask[:seq_len, :seq_len].to(dtype=tok.dtype, device=tok.device)
        h_steps = self.decoder(tgt=tok, memory=ctx_tokens, tgt_mask=tgt_mask, gate=gate)  # (B, depth, d_model)

        # 5) 输出：每一步对 n_bins 的 logits
        logits = self.out(h_steps)  # (B, depth, n_bins)
        if return_context:  # 若需要 ACM 的 ctx_tokens
            return logits, ctx_tokens
        return logits  # 默认只返回 logits
