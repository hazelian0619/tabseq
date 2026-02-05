#!/usr/bin/env python3
"""
TabSeqDataset 完整测试脚本
运行：python scripts/test_tabseq_dataset.py
"""

import numpy as np
import torch
from tabseq.data.datasets import load_california_housing_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder

print("=" * 80)
print("🧪 TabSeqDataset 完整测试")
print("=" * 80)

# Step 1: 加载数据（上游）
print("\n1️⃣ 加载 California Housing 数据集...")
split = load_california_housing_split(random_state=42, test_size=0.2)
print(f"   ✓ X_train: {split.X_train.shape}")
print(f"   ✓ y_train: {split.y_train.shape}")
print(f"   ✓ y_train range: [{split.y_train.min():.3f}, {split.y_train.max():.3f}]")

# Step 2: 创建编码器
print("\n2️⃣ 创建 TraceLabelEncoder...")
encoder = TraceLabelEncoder(depth=4, v_min=0.0, v_max=5.0)
print(f"   ✓ n_bins = {encoder.n_bins}")
print(f"   ✓ bin_width = {encoder.bin_width:.4f}")

# Step 3: 测试单个样本编码
print("\n3️⃣ 测试单个样本编码...")
test_y = 3.14
seq, leaf_idx = encoder.encode(test_y)
mht = encoder.encode_multi_hot(leaf_idx)
print(f"   ✓ y={test_y} → seq={seq}, leaf_idx={leaf_idx}")
print(f"   ✓ mht.shape = {mht.shape}")
print(f"   ✓ mht[0].sum() = {mht[0].sum()}")  # 第一步应该是 8

# Step 4: 创建数据集
print("\n4️⃣ 创建 TabSeqDataset...")
dataset = TabSeqDataset(
    X_num=split.X_train[:1000],  # 只取前1000个样本测试
    X_cat=None,
    y=split.y_train[:1000],
    encoder=encoder,
    is_train=True,
    sos_token=2
)
print(f"   ✓ 数据集大小: {len(dataset)}")
print(f"   ✓ y_multi_hots 预计算: {'是' if dataset.y_multi_hots is not None else '否'}")

# Step 5: 测试 __getitem__
print("\n5️⃣ 测试 __getitem__(0)...")
sample = dataset[0]
for key, value in sample.items():
    shape = value.shape if hasattr(value, 'shape') else 'scalar'
    print(f"   ✓ {key:12}: {shape}")

# 详细检查 dec_input 和 y_seq 的右移关系
print(f"   ✓ dec_input = {sample['dec_input']}")
print(f"   ✓ y_seq     = {sample['y_seq']}")
print(f"   ✓ dec_input[1:] == y_seq[:-1]: {torch.equal(sample['dec_input'][1:], sample['y_seq'][:-1])}")

# Step 6: 测试 DataLoader
print("\n6️⃣ 测试 DataLoader...")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
batch = next(iter(dataloader))
for key, value in batch.items():
    shape = value.shape if hasattr(value, 'shape') else 'scalar'
    print(f"   ✓ batch['{key}']: {shape}")

print(f"   ✓ y_mht 维度检查: batch['y_mht'].shape = {batch['y_mht'].shape}")
print(f"   ✓ y_mht 第一步和: {batch['y_mht'][:, 0, :].sum(dim=1)[:3]}")  # 应该是全8

# Step 7: 验证多热标签逻辑
print("\n7️⃣ 验证多热标签...")
leaf_idx_from_sample = int(dataset.y_leaf_idx[sample_idx].item())
mht_precomputed = dataset.y_multi_hots[sample_idx] if dataset.y_multi_hots is not None else None
mht_on_the_fly = torch.from_numpy(encoder.encode_multi_hot(leaf_idx_from_sample))

print(f"   ✓ 样本索引: {sample_idx}")
print(f"   ✓ leaf_idx = {leaf_idx_from_sample}")
print(f"   ✓ 预计算 mht[{sample_idx}][0].sum(): {mht_precomputed[0].sum() if mht_precomputed is not None else 'N/A'}")
print(f"   ✓ 实时计算 mht.sum(): {mht_on_the_fly.sum()}")
if mht_precomputed is not None:
    print(f"   ✓ 预计算 vs 实时一致: {(mht_precomputed[0] == mht_on_the_fly).all()}")

print("\n" + "=" * 80)
print("✅ 测试完成！如果上面所有 ✓ 都显示正确，数据流就没问题了")
print("=" * 80)
