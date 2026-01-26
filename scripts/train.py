"""
主线训练入口（TabSeq-DACA + ACM）。
用途：把“数据→编码→模型→训练”串起来，生成可复现的 checkpoint 与训练日志。
设计意义：没有训练产物就无法评估/对比，因此必须先跑通这个闭环。
输出产物：
  - outputs/run_*/checkpoint.pt：模型参数 + 训练配置
  - outputs/run_*/config.json：编码范围/深度/超参（评估必须复用）
  - outputs/run_*/metrics_train.json：训练端最小指标（loss/alpha 统计）
  - outputs/run_*/git.txt：代码版本（复现追踪）
"""

import argparse  # 命令行参数
import json  # 写 config/metrics 的 JSON 文件
import os  # 读取环境变量与创建输出目录
from datetime import datetime  # 生成唯一 run_id

import numpy as np  # 处理 numpy 数组数据
import torch  # 训练所需的张量与优化器
from torch.utils.data import DataLoader  # 批量加载 Dataset

try:
    import swanlab  # 可选的实验记录平台
except Exception:
    swanlab = None  # 未安装时直接跳过

from tabseq.data.datasets import load_california_housing_split  # 数据加载与切分
from tabseq.labels.trace_encoder import TraceLabelEncoder  # y→y_seq/y_mht 编码
from tabseq.data.tabseq_dataset import TabSeqDataset  # Dataset 封装（含 dec_input/y_mht）
from tabseq.models.transformer_model import TransformerTabSeqModel  # 主模型（DACA+ACM）
from tabseq.utils.git import get_git_hash  # 记录 git commit
from tabseq.utils.seed import set_seed  # 固定随机种子


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-confidence-masking", action="store_true")
    ap.add_argument("--alpha-depth-mode", type=str, default=None)
    ap.add_argument("--alpha-min", type=float, default=None)
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    # 1) 固定一个最小可跑设置（先跑通主线，不追最强效果）
    #    这一步决定“编码深度、桶数量、ACM 参数”，后续都会写入 config
    seed = int(args.seed) if args.seed is not None else int(os.environ.get("TABSEQ_SEED", "0"))  # 训练随机种子，可复现
    depth = int(args.depth) if args.depth is not None else 6  # 二叉决策序列长度
    n_bins = 2 ** depth  # 叶子桶数：2^depth（决定每一步的类别数）

    # ACM: 训练阶段的置信度掩码（对齐 PDF 2.3.2）
    # 设计意图：对“负样本桶”的惩罚做减弱，保留不确定性
    use_confidence_masking = not args.no_confidence_masking  # 是否开启 ACM
    alpha_depth_mode = args.alpha_depth_mode or "linear"  # 深度权重调度方式
    alpha_min = float(args.alpha_min) if args.alpha_min is not None else 0.2  # 深度权重最小值
    alpha_max = float(args.alpha_max) if args.alpha_max is not None else 0.8  # 深度权重最大值

    if not (0.0 <= alpha_min <= alpha_max <= 1.0):  # 合法性校验
        raise ValueError("alpha_min/alpha_max must satisfy 0<=alpha_min<=alpha_max<=1")

    # 训练规模与超参（最小可跑）
    # 注意：这些数值不是最终性能最优，只是保证“能跑通”
    n_train = 256  # 预留字段：当前未裁剪训练集
    batch_size = int(args.batch_size) if args.batch_size is not None else 32  # 每个 batch 的样本数
    epochs = int(args.epochs) if args.epochs is not None else 2  # 训练轮数（最小可跑）
    lr = float(args.lr) if args.lr is not None else 1e-3  # 学习率

    # 2) 输出目录（本地生成，不进 git）
    # 训练产物会被 eval/benchmark 读取，必须持久化
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # 以时间生成 run_id
    out_dir = args.out_dir or os.path.join("outputs", f"run_{run_id}")  # 输出目录
    os.makedirs(out_dir, exist_ok=True)  # 确保目录存在

    set_seed(seed)  # 固定随机种子（numpy/torch）

    run_cfg = {  # 记录本次运行的关键配置（用于实验追踪）
        "seed": seed,
        "depth": depth,
        "n_bins": n_bins,
        "n_train": n_train,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "use_confidence_masking": use_confidence_masking,
        "alpha_depth_mode": alpha_depth_mode,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
    }
    if swanlab is not None:  # 可选：记录实验到 swanlab
        try:
            swanlab.init(project="tabseq-trace", experiment=run_id, config=run_cfg)
        except TypeError:
            swanlab.init(project="tabseq-trace", config=run_cfg)

    # 3) 加载真实数据集（California Housing）
    # 这一步决定“输入特征维度”和“训练标签范围”
    split = load_california_housing_split(random_state=seed)  # 标准化 + 切分
    X_num = split.X_train  # 数值特征
    y_train = split.y_train  # 连续回归标签
    n_num_features = X_num.shape[1]  # 数值特征维度
    # 类别特征暂未接入，因此用空矩阵占位（保证接口统一）
    X_cat = np.zeros((len(y_train), 0), dtype=np.int64)

    # 用训练集的范围做 v_min/v_max（保证训练与评估编码一致）
    # 编码范围必须写入 config，否则 eval 无法复原一致的编码器
    v_min, v_max = float(y_train.min()), float(y_train.max())
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)  # y 编码器
    # Dataset 会把 y 编成 y_seq/y_mht，并构造 dec_input（teacher forcing）
    ds = TabSeqDataset(X_num=X_num, X_cat=X_cat, y=y_train, encoder=enc)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)  # DataLoader（打乱训练）

    # 4) 模型 + loss + 优化器
    model = TransformerTabSeqModel(n_num_features=n_num_features, depth=depth, n_bins=n_bins)  # 主模型
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    # BCEWithLogitsLoss：
    # - 适合 multi-hot 监督（每个桶都是 0/1）
    # - 输出逐元素 loss，便于 ACM 做加权
    # 注意：它不是 MAE/RMSE，那些在 eval.py 中计算
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    # 5) ACM 的 alpha_depth（随深度递增）
    if depth <= 1:  # 深度为 1 的边界情况
        alpha_depth = torch.full((depth,), alpha_max)  # 只有一个深度，直接取最大值
    elif alpha_depth_mode == "linear":  # 线性递增：浅层小，深层大
        steps = torch.linspace(0.0, 1.0, steps=depth)
        alpha_depth = alpha_min + (alpha_max - alpha_min) * steps
    else:
        raise ValueError(f"unknown alpha_depth_mode: {alpha_depth_mode}")
    alpha_depth_values = alpha_depth.tolist()  # 保存到 config 用于复现
    alpha_depth = alpha_depth.to(next(model.parameters()).device)  # 放到同一设备
    alpha_depth_view = alpha_depth.view(1, depth, 1)  # 方便广播到 (B, depth, n_bins)

    # 6) 训练（最小版本：只看 loss 能否下降、能否保存）
    model.train()  # 进入训练模式
    step = 0  # 记录全局 step
    alpha_sum = 0.0  # 统计 alpha_instance 均值
    alpha_count = 0  # 统计 alpha_instance 样本数
    alpha_instance_min = None  # 记录 alpha_instance 最小值
    alpha_instance_max = None  # 记录 alpha_instance 最大值

    # 训练循环的衔接关系（从字段到损失）：
    # - batch["x_num"/"x_cat"/"dec_input"] -> 模型前向 -> logits
    # - batch["y_mht"] -> 训练监督（多热集合）
    # - logits 与 y_mht 用 BCE 计算逐元素损失
    for epoch in range(epochs):  # 外层 epoch
        for batch in dl:  # 内层 batch
            # batch 来自 TabSeqDataset：
            # - dec_input 是 y_seq 的前缀（teacher forcing）
            # - y_mht 是当前前缀下“仍可能的叶子集合”
            if use_confidence_masking:  # 若开启 ACM
                logits, ctx_tokens = model(batch, return_context=True)  # 前向 + 取上下文
                alpha_instance = model.compute_alpha_instance(ctx_tokens)  # (B,) 样本权重
            else:
                logits = model(batch)  # 只取 logits
                alpha_instance = None  # 不计算 alpha_instance

            # target 来自编码器的 multi-hot：每一步对应一半叶子集合
            # 直觉：模型每一步不是“猜一个叶子”，而是“选择哪一半叶子还可能”
            target = batch["y_mht"]
            # loss_raw 的形状与 logits 相同：逐步、逐桶的损失
            # 这样才能在 ACM 中对“负样本桶”做加权
            loss_raw = loss_fn(logits, target)

            if use_confidence_masking:
                # ACM 权重：alpha = alpha_depth(t) * alpha_instance(x)
                # 正样本权重=1；负样本权重=1-alpha（降低“过度排除”惩罚）
                alpha = alpha_depth_view * alpha_instance.view(-1, 1, 1)
                alpha = alpha.clamp(0.0, 1.0)  # 防止负权重
                weight = torch.where(target > 0.5, torch.ones_like(loss_raw), 1.0 - alpha)
                loss = (loss_raw * weight).mean()  # 加权后取均值
                # 直觉：alpha 越大 -> 负样本惩罚越小 -> 模型更“保守”

                alpha_sum += float(alpha_instance.sum().item())  # 统计均值
                alpha_count += int(alpha_instance.numel())  # 统计样本数
                batch_min = float(alpha_instance.min().item())  # 当前 batch 最小值
                batch_max = float(alpha_instance.max().item())  # 当前 batch 最大值
                alpha_instance_min = batch_min if alpha_instance_min is None else min(alpha_instance_min, batch_min)
                alpha_instance_max = batch_max if alpha_instance_max is None else max(alpha_instance_max, batch_max)
            else:
                loss = loss_raw.mean()  # 不使用 ACM 的普通 BCE

            # 参数更新三步：清空梯度 -> 反向传播 -> 优化器更新
            # 这是从“损失”回到“模型参数”的唯一路径
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 日志间隔：只记录训练损失（不是评估指标）
            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")
                if swanlab is not None:  # 可选记录到 swanlab（训练 loss，不是评估指标）
                    log_data = {
                        "loss": float(loss.item()),
                        "epoch": int(epoch),
                        "step": int(step),
                    }
                    if use_confidence_masking:
                        log_data["alpha_instance_mean"] = float(alpha_instance.mean().item())
                        log_data["alpha_instance_min"] = float(alpha_instance.min().item())
                        log_data["alpha_instance_max"] = float(alpha_instance.max().item())
                    swanlab.log(log_data)
            step += 1  # step 递增

    # 7) 保存 checkpoint + 最小指标（供 eval/benchmark 复现）
    # 关键衔接：
    # - eval.py 会读 config 里的 v_min/v_max/depth/n_bins 来复原编码器
    # - 这些值一旦不一致，评估指标将不可比
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")  # 模型参数保存路径
    # config.json 的每个字段都会被 eval.py 使用或记录：
    # - v_min/v_max/depth/n_bins：用于复原 TraceLabelEncoder（编码一致性）
    # - n_num_features：用于重建模型输入层
    # - seed：用于复现实验
    # - use_confidence_masking/alpha_*：用于记录 ACM 的具体设置
    cfg = {
        "v_min": v_min,
        "v_max": v_max,
        "depth": depth,
        "n_bins": n_bins,
        "n_num_features": n_num_features,
        "seed": seed,
        "cat_cardinalities": [],
        "model": "tabseq_daca",
        "use_confidence_masking": use_confidence_masking,
        "alpha_depth_mode": alpha_depth_mode,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_depth_values": alpha_depth_values,
    }
    torch.save(  # 保存模型参数与配置
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)  # 写 config.json
    with open(os.path.join(out_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")  # 写入 git hash

    # 训练端最小指标（注意：不是验证集指标）
    # final_loss 只代表“最后一个 batch 的训练损失”，用于快速 sanity check
    # alpha_instance_* 用于判断 ACM 是否稳定（过大/过小可能导致训练不稳）
    metrics = {"final_loss": float(loss.item())}
    if use_confidence_masking and alpha_count > 0:  # 若启用 ACM，记录 alpha 统计
        metrics["alpha_instance_mean"] = float(alpha_sum / alpha_count)
        metrics["alpha_instance_min"] = float(alpha_instance_min)
        metrics["alpha_instance_max"] = float(alpha_instance_max)
    with open(os.path.join(out_dir, "metrics_train.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)  # 写 metrics_train.json

    print("saved:", ckpt_path)  # 训练完成提示


if __name__ == "__main__":
    main()  # 脚本入口
