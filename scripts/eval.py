"""
评估脚本（主线 M1：train -> eval 的 eval 部分）

你可以把它当成一句话：
  读取训练保存的 checkpoint -> 在验证集上统一口径计算指标 -> 把结果写回 outputs/ 目录

运行方式（把路径换成你自己的）：
  python scripts/eval.py --ckpt outputs/<dataset>/run_YYYYmmdd_HHMMSS/
  # 或
  python scripts/eval.py --ckpt outputs/<dataset>/run_YYYYmmdd_HHMMSS/checkpoint.pt

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
from typing import Optional, Tuple
from datetime import datetime

import numpy as np  # 数组与类型转换
import torch  # 张量与模型推理
from torch.utils.data import DataLoader

from tabseq.data.datasets import load_dataset_split  # 数据加载与切分
from tabseq.data.tabseq_dataset import TabSeqDataset  # Dataset 封装
from tabseq.labels.trace_encoder import TraceLabelEncoder  # y 编码器
from tabseq.metrics.holographic import ExtendedHolographicMetric  # 统一指标
from tabseq.models.transformer_model import TransformerTabSeqModel  # 主模型
from tabseq.utils.config import choose, load_config, resolve_section, write_json
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


def _maybe_subsample(
    leaf_probs: np.ndarray,
    y_raw: np.ndarray,
    y_clipped: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or max_samples >= len(y_raw):
        return leaf_probs, y_raw, y_clipped
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_raw), size=max_samples, replace=False)
    return leaf_probs[idx], y_raw[idx], y_clipped[idx]


def _fmt_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _safe_path(out_dir: str, filename: str) -> str:
    path = os.path.join(out_dir, filename)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(filename)
    idx = 2
    while True:
        candidate = os.path.join(out_dir, f"{base}_v{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _greedy_step_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    depth: int,
    n_bins: int,
    temperature: float,
    mask_outside: Optional[float] = None,
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
    if mask_outside is not None:
        mask_outside = float(mask_outside)
        if not (0.0 <= mask_outside <= 1.0):
            raise ValueError("mask_outside must be in [0, 1]")

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

    for t in range(depth):
        logits = model({"x_num": x_num, "x_cat": x_cat, "dec_input": dec_input})
        probs_t = torch.sigmoid(logits[:, t, :] / temperature)
        if mask_outside is not None and mask_outside < 1.0:
            mask = torch.full_like(probs_t, fill_value=mask_outside)
            for b in range(B):
                mask[b, start[b]:end[b]] = 1.0
            probs_t = probs_t * mask
        
        # ✅ 记录当前概率（可选 soft mask）
        step_probs_out[:, t, :] = probs_t
        
        # 决策用 mask
        if t < depth - 1:
            bits = torch.empty((B,), dtype=torch.long, device=device)
            for b in range(B):
                s, e = start[b], end[b]
                mid = (s + e) // 2
                # 🔧 改成 sum（助手说的小优化）
                left = probs_t[b, s:mid].sum().item()
                right = probs_t[b, mid:e].sum().item()
                bit = 1 if right > left else 0
                bits[b] = bit
                
                if bit == 0:
                    end[b] = mid
                else:
                    start[b] = mid
            
            dec_input[:, t + 1] = bits
    
    return step_probs_out



def main():
    ap = argparse.ArgumentParser()  # 命令行参数解析器
    ap.add_argument("--ckpt", required=True, help="例如：outputs/<dataset>/run_xxx/ 或 outputs/<dataset>/run_xxx/checkpoint.pt")  # 模型路径
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")  # 评估配置
    ap.add_argument("--batch-size", type=int, default=None)  # 评估 batch 大小
    ap.add_argument("--dataset", type=str, default=None, help="override dataset name")
    ap.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="数据划分种子（默认使用 checkpoint 里的 seed）",
    )
    ap.add_argument("--confidence", type=float, default=None, help="区间置信度，例如 0.90 表示 90%% 预测区间")  # PICP 目标
    ap.add_argument("--temperature", type=float, default=None, help="温度T：sigmoid(logits/T)，T>1 更保守")  # 温度缩放
    ap.add_argument(  # 模式选择
        "--mode",
        type=str,
        default=None,
        choices=["teacher_forcing", "greedy", "both"],
        help="teacher_forcing=用真实dec_input；greedy=真实推理只给SOS；both=两者都跑",
    )
    ap.add_argument(
        "--mask-outside",
        type=float,
        default=None,
        help="greedy 时对当前不可达区间的 soft mask 系数（0~1，1 表示不 mask）",
    )
    ap.add_argument("--export-leaf-probs", action="store_true", help="export leaf_probs/y_true/bin_edges to npz")
    ap.add_argument("--export-samples", type=int, default=0, help="subsample count for export (0 means all)")
    args = ap.parse_args()  # 解析命令行参数

    defaults = {
        "batch_size": 256,
        "confidence": 0.90,
        "temperature": 1.0,
        "mode": "teacher_forcing",
        "mask_outside": 1.0,
        "export_leaf_probs": False,
        "export_samples": 0,
    }
    config = load_config(args.config)
    eval_cfg = resolve_section(config, "eval")
    args.batch_size = choose(args.batch_size, eval_cfg.get("batch_size"), defaults["batch_size"])
    args.confidence = choose(args.confidence, eval_cfg.get("confidence"), defaults["confidence"])
    args.temperature = choose(args.temperature, eval_cfg.get("temperature"), defaults["temperature"])
    args.mode = choose(args.mode, eval_cfg.get("mode"), defaults["mode"])
    args.mask_outside = choose(args.mask_outside, eval_cfg.get("mask_outside"), defaults["mask_outside"])
    args.export_leaf_probs = choose(
        args.export_leaf_probs, eval_cfg.get("export_leaf_probs"), defaults["export_leaf_probs"]
    )
    args.export_samples = choose(
        args.export_samples, eval_cfg.get("export_samples"), defaults["export_samples"]
    )
    args.random_state = choose(args.random_state, eval_cfg.get("random_state"), None)
    args.dataset = choose(args.dataset, eval_cfg.get("dataset"), None)
    if args.mode not in {"teacher_forcing", "greedy", "both"}:
        raise ValueError(f"invalid mode: {args.mode}")

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

    random_state = int(args.random_state) if args.random_state is not None else int(cfg.get("seed", 0))
    set_seed(random_state)  # 固定随机种子，保证数据划分一致

    dataset = args.dataset or cfg.get("dataset", "california_housing")
    if args.dataset is not None and cfg.get("dataset") and args.dataset != cfg.get("dataset"):
        raise ValueError(f"dataset mismatch: args.dataset={args.dataset} vs ckpt={cfg.get('dataset')}")

    # 2) 准备验证集
    #    注意：我们只在验证集上算指标；标准化统计量来自训练集。
    split = load_dataset_split(dataset, random_state=random_state)  # 加载数据并标准化
    X_val, y_val = split.X_val, split.y_val  # 取验证集
    X_cat_val = split.X_cat_val
    cat_cardinalities = cfg.get("cat_cardinalities") or []
    n_cat_features = int(X_cat_val.shape[1]) if X_cat_val is not None else 0
    if n_cat_features == 0 and cat_cardinalities:
        raise ValueError("checkpoint expects categorical features but eval data has none")
    if n_cat_features > 0:
        if not cat_cardinalities:
            raise ValueError("eval data has categorical features but checkpoint has no cat_cardinalities")
        if len(cat_cardinalities) != n_cat_features:
            raise ValueError(
                f"len(cat_cardinalities)={len(cat_cardinalities)} must match X_cat_val.shape[1]={n_cat_features}"
            )

    # 3) 构造 Dataset / DataLoader
    #    这里 y 会被 TraceLabelEncoder 转成：
    #    - y_seq（0/1 序列）
    #    - y_mht（multi-hot 监督目标）
    #    - y_raw（原始连续 y，可能超出编码范围）
    #    - y_clipped（裁剪到 [v_min, v_max] 的 y，用于指标）
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)  # 创建编码器（与训练一致）
    ds = TabSeqDataset(  # 构造 Dataset
        X_num=X_val,  # 数值特征
        X_cat=X_cat_val if X_cat_val is not None else np.zeros((len(y_val), 0), dtype=np.int64),
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
            cat_cardinalities=cat_cardinalities,
        )
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)  # 兼容旧模型
    model.load_state_dict(ckpt["model_state_dict"])  # 加载权重
    model.eval()  # 进入评估模式（关闭 dropout）

    # 5) 统一口径的指标计算（对齐 notebook 的 ExtendedHolographicMetric）
    #    这个 metric 内部会：step_probs -> leaf_probs -> y_hat/[L,U] -> 指标
    metric_calc = ExtendedHolographicMetric(enc)  # 初始化指标计算器

    out_dir = os.path.dirname(ckpt_path)  # 输出与 checkpoint 同目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common_tag = f"T{_fmt_float(float(args.temperature))}_C{_fmt_float(float(args.confidence))}"

    def run_mode(mode: str):
        # 收集所有 batch 的 step_probs 与 y_true
        step_probs_all = []
        y_true_all = []
        y_raw_all = []
        y_clipped_all = []
        oob_count = 0
        total_count = 0

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
                        x_cat=batch.get("x_cat"),
                        depth=depth,
                        n_bins=n_bins,
                        temperature=float(args.temperature),
                        mask_outside=float(args.mask_outside),
                        sos_token=2,
                    )
                else:
                    raise ValueError(f"unknown mode: {mode}")

                # step_probs: (B, depth, n_bins)
                # y_clipped: (B,) 用于最终指标（与编码范围一致）
                step_probs_all.append(step_probs.cpu())
                y_true_all.append(batch["y_clipped"].cpu())
                y_raw_all.append(batch["y_raw"].cpu())
                y_clipped_all.append(batch["y_clipped"].cpu())
                if "y_raw" in batch:
                    y_raw = batch["y_raw"]
                    oob = (y_raw < v_min) | (y_raw > v_max)
                    oob_count += int(oob.sum().item())
                    total_count += int(y_raw.numel())

        # 统一指标计算（内部会算 MAE/RMSE/PICP/MPIW）
        model_probs = torch.cat(step_probs_all, dim=0)
        y_true = torch.cat(y_true_all, dim=0)
        metrics = metric_calc.compute_metrics(
            model_probs=model_probs,
            y_true=y_true,
            confidence=float(args.confidence),
            return_extras=True,
        )
        # 记录评估口径与温度，方便对比
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = mode
        metrics["temperature"] = float(args.temperature)
        if total_count > 0:
            metrics["oob_rate"] = float(oob_count / total_count)

        if args.export_leaf_probs:
            leaf_probs = metric_calc.leaf_probs_from_step_probs(model_probs).cpu().numpy()
            y_raw_np = torch.cat(y_raw_all, dim=0).numpy()
            y_clipped_np = torch.cat(y_clipped_all, dim=0).numpy()
            leaf_probs, y_raw_np, y_clipped_np = _maybe_subsample(
                leaf_probs, y_raw_np, y_clipped_np, int(args.export_samples), seed=int(random_state)
            )
            bin_edges = np.linspace(v_min, v_max, n_bins + 1, dtype=np.float32)
            bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1:] - bin_edges[:-1])
            meta = {
                "ckpt": ckpt_path,
                "split": "val",
                "mode": mode,
                "temperature": float(args.temperature),
                "depth": int(depth),
                "n_bins": int(n_bins),
                "v_min": float(v_min),
                "v_max": float(v_max),
                "random_state": int(random_state),
            }
            export_samples = "all" if int(args.export_samples) <= 0 else str(int(args.export_samples))
            tag_parts = [mode, common_tag, f"mask{_fmt_float(float(args.mask_outside))}"]
            if export_samples:
                tag_parts.append(f"N{export_samples}")
            tag_parts.append(timestamp)
            tag = "_".join(tag_parts)
            export_path = _safe_path(out_dir, f"leaf_probs_val_{tag}.npz")
            np.savez_compressed(
                export_path,
                leaf_probs=leaf_probs.astype(np.float32),
                y_raw=y_raw_np.astype(np.float32),
                y_clipped=y_clipped_np.astype(np.float32),
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                meta=json.dumps(meta, ensure_ascii=False),
            )
            print("saved:", export_path)
        return metrics

    write_json(
        _safe_path(
            out_dir,
            f"eval_config_{common_tag}_mask{_fmt_float(float(args.mask_outside))}_{timestamp}.json",
        ),
        {
            "config_path": args.config,
            "batch_size": int(args.batch_size),
            "random_state": int(random_state),
            "confidence": float(args.confidence),
            "temperature": float(args.temperature),
            "mode": str(args.mode),
            "mask_outside": float(args.mask_outside),
            "export_leaf_probs": bool(args.export_leaf_probs),
            "export_samples": int(args.export_samples),
        },
    )

    if args.mode in ("teacher_forcing", "both"):  # 需要 teacher_forcing
        m_tf = run_mode("teacher_forcing")  # 计算指标
        print("teacher_forcing:", m_tf)  # 打印指标
        tf_tag = f"teacher_forcing_{common_tag}_{timestamp}"
        tf_path = _safe_path(out_dir, f"metrics_val_{tf_tag}.json")
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(m_tf, f, ensure_ascii=False, indent=2)  # 写入文件
        print("saved:", tf_path)

    if args.mode in ("greedy", "both"):  # 需要 greedy
        m_g = run_mode("greedy")  # 计算指标
        print("greedy:", m_g)  # 打印指标
        g_tag = f"greedy_{common_tag}_mask{_fmt_float(float(args.mask_outside))}_{timestamp}"
        g_path = _safe_path(out_dir, f"metrics_val_{g_tag}.json")
        with open(g_path, "w", encoding="utf-8") as f:
            json.dump(m_g, f, ensure_ascii=False, indent=2)  # 写入文件
        print("saved:", g_path)

    if args.mode == "both":  # 同时保存两种口径
        both = {"teacher_forcing": m_tf, "greedy": m_g}  # 合并结果
        out_path = _safe_path(out_dir, f"metrics_val_both_{common_tag}_{timestamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(both, f, ensure_ascii=False, indent=2)  # 写入文件
        print("saved:", out_path)  # 打印保存路径


if __name__ == "__main__":
    main()
