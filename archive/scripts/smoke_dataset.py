"""
数据链路烟测：验证“数据 → 编码 → Dataset 封装”的衔接是否正确。
设计意义：在训练前用最小输入检查字段/形状是否对齐，提前发现编码或数据封装错误。
使用场景：任何改动 TraceLabelEncoder 或 TabSeqDataset 后先跑一次，避免训练白跑。
注意：这里只打印样本字段与形状，不涉及训练指标（MAE/RMSE/PICP/MPIW）。

记忆口诀（帮助快速记住）：
- “y 先变序列，序列喂模型；模型出分布，分布算指标”
- “训练看 y_mht，评估看 y_raw；dec_input 是训练前缀”

逻辑关系（记忆用）：
1) TraceLabelEncoder：把连续 y 编成 y_seq（二进制序列）与 y_mht（多热集合）。
2) TabSeqDataset：把 y_seq 变成 dec_input（teacher forcing 前缀），并把 y_mht
   作为监督目标；同时保留 y_raw 供评估使用。
"""

import numpy as np  # 用于构造小规模的虚拟输入

from tabseq.labels.trace_encoder import TraceLabelEncoder  # 将连续 y 编码为 y_seq/y_mht
from tabseq.data.tabseq_dataset import TabSeqDataset  # Dataset 封装（含 dec_input/y_mht）

# 1) 标签编码器：把连续 y 划分到固定范围的桶里
#    depth=6 -> 2^6=64 个桶
enc = TraceLabelEncoder(0, 5, depth=6)

# 2) 构造小样本输入（3 条样本）
#    - 3：样本数量很小，方便快速查看输出字段与形状（不是训练规模）
#    - 4：数值特征维度的占位值，只为验证“数值特征能进 Dataset”
#    - 2：类别特征维度的占位值，只为验证“类别特征通路存在”
#    这些维度并非最优超参，只是用来做“形状对齐”的最小示例
X_num = np.random.randn(3, 4).astype(np.float32)
X_cat = np.zeros((3, 2), dtype=np.int64)

# 3) 连续标签 y（回归目标）
#    - 这里用 [1.0, 2.0, 3.0] 只是方便人工理解的示例值
#    - 它们必须落在编码器的范围 [0, 5] 内，否则会被 clip
#    - 目的不是“学到规律”，而是验证编码与字段生成是否正确
y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# 4) Dataset 封装：内部会生成 y_seq/y_mht，并构造 dec_input
ds = TabSeqDataset(X_num=X_num, X_cat=X_cat, y=y, encoder=enc)
# 取第 1 条样本，检查字段与形状是否正确
#    - 这里用索引 0 只是拿第一条样本做检查
#    - 换成 ds[1]/ds[2] 也应当返回相同结构
item = ds[0]

# keys 含义（这些字段后续会被模型/评估使用）：
# - x_num: 数值特征（模型输入）
# - x_cat: 类别特征（模型输入，可为空）
# - dec_input: 训练时喂给模型的前缀（[SOS] + y_seq[:-1]）
# - y_seq: 叶子桶索引的二进制序列（长度=depth，用来构造 dec_input）
# - y_mht: 多热集合（每一步“仍可能的叶子集合”，训练监督目标）
# - y_raw: 原始连续 y（评估 MAE/RMSE/PICP/MPIW 时用）
print("keys:", item.keys())

# shapes 含义（用于核对编码是否对齐）：
# - x_num.shape: 数值特征维度（这里是 4）
# - x_cat.shape: 类别特征维度（这里是 2，占位）
# - dec_input.shape: 序列长度（=depth，这里是 6）
# - y_mht.shape: (depth, n_bins)，n_bins=2^depth（这里是 6x64）
print(
    "shapes:",
    item["x_num"].shape,
    item["x_cat"].shape,
    item["dec_input"].shape,
    item["y_mht"].shape,
)

# y_raw: 原始回归标签值（用于和编码后的 y_seq/y_mht 对照）
print("y_raw:", float(item["y_raw"]))
