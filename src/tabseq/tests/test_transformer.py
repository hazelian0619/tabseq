import numpy as np
import os
import sys

import torch
from torch.utils.data import DataLoader

# 1. 确保能找到项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

# 2. 标准导入
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.data.datasets import load_california_housing_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder

def test_transformer_model():
    print("=" * 60)
    print("🚀 TransformerTabSeqModel 调试测试")
    print("=" * 60)
    
    # 1. 加载数据
    print("正在加载数据...")
    split = load_california_housing_split(random_state=42, test_size=0.2)
    X_train = split.X_train
    y_train = split.y_train
    encoder = TraceLabelEncoder(
        depth=4,
        v_min=float(y_train.min()),
        v_max=float(y_train.max()),
    )
    train_dataset = TabSeqDataset(
        X_num=X_train,
        X_cat=np.zeros((len(y_train), 0), dtype=np.int64),
        y=y_train,
        encoder=encoder,
        is_train=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    batch = next(iter(train_loader))
    print(f"✓ 数据加载成功: x_num={batch['x_num'].shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 2. 模型创建
    model = TransformerTabSeqModel(
        n_num_features=8,
        depth=4,
        n_bins=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ).to(device)
    print("✓ 模型创建成功")
    
    # 3. 前向传播测试
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    print("\n--- 测试1：前向传播 (Forward) ---")
    logits = model(batch) 
    print(f"  ✓ logits.shape: {logits.shape}")  # 期望: (8, 4, 16)
    print(f"  ✓ 无 NaN: {not torch.isnan(logits).any()}")
    
    # 4. 损失测试
    print("\n--- 测试2：损失计算 (Compute Loss) ---")
    logits, ctx_tokens = model(batch, return_context=True)  # 返回两个值
    loss = model.compute_loss(
        logits=logits, 
        y_mht=batch['y_mht'], 
        ctx_tokens=ctx_tokens  # 传入 ctx_tokens
    )
    print(f"  ✓ loss: {loss.item():.4f}")
    
    # 5. 反向传播测试
    print("\n--- 测试3：反向传播 (Backward) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  ✓ 梯度更新成功！")
    
    print("\n" + "=" * 60)
    print("🎉 TransformerTabSeqModel 核心逻辑调通！")
    print("=" * 60)

if __name__ == "__main__":
    test_transformer_model()
