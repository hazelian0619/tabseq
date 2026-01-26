"""
评估脚本（主线 M1：train -> eval 的 eval 部分）

你可以把它当成一句话：
  读取训练保存的 checkpoint -> 在验证集上统一口径计算指标 -> 把结果写回 outputs/ 目录

运行方式（把路径换成你自己的）：
  python scripts/eval.py --ckpt outputs/run_YYYYmmdd_HHMMSS/
  # 或
  python scripts/eval.py --ckpt outputs/run_YYYYmmdd_HHMMSS/checkpoint.pt

指标含义（便于记忆）：
- MAE：平均绝对误差，越小越好（点预测）
- RMSE：均方根误差，越小越好（对大误差更敏感）
- PICP：区间覆盖率，越接近 confidence 越好（可靠性）
- MPIW：区间平均宽度，越小越好（尖锐度）

评估口径说明：
- teacher_forcing：训练口径，喂真实前缀（上限/调试）
- greedy：推理口径，只给 SOS，自回归生成前缀（真实部署）

流程衔接（从输入到输出）：
1) 读取 checkpoint/config → 复原编码器参数（v_min/v_max/depth）
2) 构造验证集 Dataset → 产生 dec_input/y_mht/y_raw
3) 模型前向 → step_probs（每步对叶子桶的概率）
4) step_probs → leaf_probs → 点预测/区间 → MAE/RMSE/PICP/MPIW
5) 写回 metrics_val*.json 供 benchmark 汇总
"""

import argparse  # 解析命令行参数
import json  # 保存评估结果
import os  # 路径处理

import numpy as np  # 数组与类型转换
import torch  # 张量与模型推理
from torch.utils.data import DataLoader

from tabseq.data.datasets import load_california_housing_split  # 数据加载与切分
from tabseq.data.tabseq_dataset import TabSeqDataset  # Dataset 封装
from tabseq.labels.trace_encoder import TraceLabelEncoder  # y 编码器
from tabseq.metrics.holographic import ExtendedHolographicMetric  # 统一指标
from tabseq.models.transformer_model import TransformerTabSeqModel  # 主模型
from tabseq.utils.seed import set_seed  # 固定随机种子（复现）


def _resolve_ckpt_path(path: str) -> str:
    # 允许传 run 目录（更符合日常使用）
    if os.path.isdir(path):  # 如果传的是目录
        ckpt_path = os.path.join(path, "checkpoint.pt")  # 默认找目录下的 checkpoint.pt
        if not os.path.isfile(ckpt_path):  # 目录里没有模型文件就报错
            raise FileNotFoundError(f"checkpoint.pt not found in dir: {path}")
        return ckpt_path  # 返回完整 checkpoint 路径
    return path  # 传的是文件路径就直接返回


def _infer_model_class(state_dict: dict):
    # 通过 state_dict 的参数名推断模型类型（兼容旧 checkpoint）
    if any(key.startswith("decoder.") for key in state_dict.keys()):  # 新结构带 decoder
        return TransformerTabSeqModel  # 推断为 TransformerTabSeqModel
    if any(key.startswith("tabular_encoder.") for key in state_dict.keys()):  # TabularEncoder 前缀
        return TransformerTabSeqModel
    if any(key.startswith("encoder.") for key in state_dict.keys()):  # 旧结构可能带 encoder 前缀
        return TransformerTabSeqModel


def _greedy_step_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    depth: int,
    n_bins: int,
    temperature: float,
    sos_token: int = 2,
) -> torch.Tensor:
    """
    真正推理（greedy，自回归）：
    - 只给 [SOS] 起步
    - 每一步用模型输出决定“这一层选左(0)还是右(1)”
    - 再把这个 bit 填进 dec_input，继续下一步

    返回：
      step_probs: (B, depth, n_bins)
    """
    temperature = float(temperature)  # 温度缩放：T>1 更保守，T<1 更自信
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B = x_num.shape[0]
    device = x_num.device

    # dec_input 的语义与训练一致：
    # dec_input = [SOS] + [b0, b1, ..., b_{D-2}]
    dec_input = torch.zeros((B, depth), dtype=torch.long, device=device)
    dec_input[:, 0] = sos_token

    step_probs_out = torch.empty((B, depth, n_bins), dtype=torch.float32, device=device)

    # 每个样本都有自己的 [start,end) 区间（因为 greedy 可能走不同路径）
    start = [0 for _ in range(B)]
    end = [n_bins for _ in range(B)]

    for t in range(depth):  # 按步自回归生成（t=0..depth-1）
        logits = model({"x_num": x_num, "dec_input": dec_input})
        probs_t = torch.sigmoid(logits[:, t, :] / temperature)
        # 只保留当前区间 [start, end) 内的概率，其他位置置 0

        mask = torch.zeros_like(probs_t)
        for b in range(B):
            mask[b, start[b]:end[b]] = 1.0  # 只保留[s,e]
        probs_t = probs_t * mask  # 应用掩码     
      
        step_probs_out[:, t, :] = probs_t # 记录当前步的叶子概率

        # 决策 bit_t：
        # 比较当前区间内 “左半叶子集合” vs “右半叶子集合” 的平均概率
        if t < depth - 1:  # 最后一步不需要再产生下一个 bit
            bits = torch.empty((B,), dtype=torch.long, device=device)  # 存每个样本的 bit_t
            for b in range(B):  # 对每个样本分别做区间决策
                s = start[b]  # 当前样本的区间起点
                e = end[b]  # 当前样本的区间终点
                mid = (s + e) // 2  # 当前区间中点
                left = probs_t[b, s:mid].mean().item()   # 左半集合平均概率
                right = probs_t[b, mid:e].mean().item()  # 右半集合平均概率
                bit = 1 if right > left else 0  # 选择概率更大的那一半
                bits[b] = bit  # 记录该样本的决策
                if bit == 0:  # 选择左半
                    end[b] = mid  # 右边界收缩到 mid
                else:  # 选择右半
                    start[b] = mid  # 左边界推进到 mid

            dec_input[:, t + 1] = bits  # 把 bit_t 写回前缀，供下一步使用

    return step_probs_out


def main():
    ap = argparse.ArgumentParser()  # 命令行参数解析器
    ap.add_argument("--ckpt", required=True, help="例如：outputs/run_xxx/ 或 outputs/run_xxx/checkpoint.pt")  # 模型路径
    ap.add_argument("--batch-size", type=int, default=256)  # 评估 batch 大小
    ap.add_argument("--random-state", type=int, default=0, help="要和 train.py 一致（默认 0）")  # 数据划分种子
    ap.add_argument("--confidence", type=float, default=0.90, help="区间置信度，例如 0.90 表示 90%% 预测区间")  # PICP 目标
    ap.add_argument("--temperature", type=float, default=1.0, help="温度T：sigmoid(logits/T)，T>1 更保守")  # 温度缩放
    ap.add_argument(  # 模式选择
        "--mode",
        type=str,
        default="teacher_forcing",
        choices=["teacher_forcing", "greedy", "both"],
        help="teacher_forcing=用真实dec_input；greedy=真实推理只给SOS；both=两者都跑",
    )
    args = ap.parse_args()  # 解析命令行参数

    # 1) 读取 checkpoint（里面应该保存：模型参数 + 训练时用的关键配置）
    ckpt_path = _resolve_ckpt_path(args.ckpt)  # 解析 checkpoint 路径
    ckpt = torch.load(ckpt_path, map_location="cpu")  # 读取模型权重
    cfg = ckpt["config"]  # 读取训练时写入的配置

    # config 的关键字段用于“复原编码器与模型形状”
    depth = int(cfg["depth"])  # 复原决策深度
    n_bins = int(cfg["n_bins"])  # 复原桶数量
    n_num_features = int(cfg["n_num_features"])  # 复原数值特征维度
    v_min = float(cfg["v_min"])  # 复原编码范围下界
    v_max = float(cfg["v_max"])  # 复原编码范围上界

    set_seed(args.random_state)  # 固定随机种子，保证数据划分一致

    # 2) 准备验证集（California Housing）
    #    注意：我们只在验证集上算指标；标准化统计量来自训练集。
    split = load_california_housing_split(random_state=args.random_state)  # 加载数据并标准化
    X_val, y_val = split.X_val, split.y_val  # 取验证集

    # 3) 构造 Dataset / DataLoader
    #    这里 y 会被 TraceLabelEncoder 转成：
    #    - y_seq（0/1 序列）
    #    - y_mht（multi-hot 监督目标）
    #    - y_raw（原始连续 y，用来算 MAE/RMSE）
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)  # 创建编码器（与训练一致）
    ds = TabSeqDataset(  # 构造 Dataset
        X_num=X_val,  # 数值特征
        X_cat=np.zeros((len(y_val), 0), dtype=np.int64),  # 先不做类别特征，留空
        y=y_val,  # 连续标签
        encoder=enc,  # 编码器（生成 y_seq/y_mht/dec_input）
        is_train=False,  # 评估模式（不影响字段结构）
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)  # DataLoader（不打乱）

    # 4) 载入模型并进入 eval 模式
    #    这里的模型形状必须与训练时一致（n_num_features/depth/n_bins）
    model_cls = _infer_model_class(ckpt["model_state_dict"])  # 推断模型类型
    if model_cls is TransformerTabSeqModel:  # 主模型
        model = model_cls(  # 按训练配置复原
            n_num_features=n_num_features,
            depth=depth,
            n_bins=n_bins,
            cat_cardinalities=cfg.get("cat_cardinalities"),
        )
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)  # 兼容旧模型
    model.load_state_dict(ckpt["model_state_dict"])  # 加载权重
    model.eval()  # 进入评估模式（关闭 dropout）

    # 5) 统一口径的指标计算（对齐 notebook 的 ExtendedHolographicMetric）
    #    这个 metric 内部会：step_probs -> leaf_probs -> y_hat/[L,U] -> 指标
    metric_calc = ExtendedHolographicMetric(enc)  # 初始化指标计算器

    def run_mode(mode: str):
        # 收集所有 batch 的 step_probs 与 y_true
        step_probs_all = []
        y_true_all = []

        with torch.no_grad():
            for batch in dl:
                if mode == "teacher_forcing":
                    # teacher_forcing：用真实 dec_input 前缀
                    logits = model(batch)
                    step_probs = torch.sigmoid(logits / float(args.temperature))
                elif mode == "greedy":
                    # greedy：从 [SOS] 开始自回归生成前缀
                    step_probs = _greedy_step_probs(
                        model=model,
                        x_num=batch["x_num"],
                        depth=depth,
                        n_bins=n_bins,
                        temperature=float(args.temperature),
                        sos_token=2,
                    )
                else:
                    raise ValueError(f"unknown mode: {mode}")

                # step_probs: (B, depth, n_bins)
                # y_raw: (B,) 用于最终指标
                step_probs_all.append(step_probs.cpu())
                y_true_all.append(batch["y_raw"].cpu())

        # 统一指标计算（内部会算 MAE/RMSE/PICP/MPIW）
        metrics = metric_calc.compute_metrics(
            model_probs=torch.cat(step_probs_all, dim=0),
            y_true=torch.cat(y_true_all, dim=0),
            confidence=float(args.confidence),
            return_extras=True,
        )
        # 记录评估口径与温度，方便对比
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = mode
        metrics["temperature"] = float(args.temperature)
        return metrics

    out_dir = os.path.dirname(ckpt_path)  # 输出与 checkpoint 同目录

    if args.mode in ("teacher_forcing", "both"):  # 需要 teacher_forcing
        m_tf = run_mode("teacher_forcing")  # 计算指标
        print("teacher_forcing:", m_tf)  # 打印指标
        with open(os.path.join(out_dir, "metrics_val_teacher_forcing.json"), "w", encoding="utf-8") as f:
            json.dump(m_tf, f, ensure_ascii=False, indent=2)  # 写入文件

    if args.mode in ("greedy", "both"):  # 需要 greedy
        m_g = run_mode("greedy")  # 计算指标
        print("greedy:", m_g)  # 打印指标
        with open(os.path.join(out_dir, "metrics_val_greedy.json"), "w", encoding="utf-8") as f:
            json.dump(m_g, f, ensure_ascii=False, indent=2)  # 写入文件

    if args.mode == "both":  # 同时保存两种口径
        both = {"teacher_forcing": m_tf, "greedy": m_g}  # 合并结果
        out_path = os.path.join(out_dir, "metrics_val.json")  # 综合输出
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(both, f, ensure_ascii=False, indent=2)  # 写入文件
        print("saved:", out_path)  # 打印保存路径
    else:
        out_path = os.path.join(out_dir, f"metrics_val_{args.mode}.json")  # 单一口径输出
        print("saved:", out_path)  # 打印保存路径


if __name__ == "__main__":
    main()
