"""
评估脚本：对训练得到的 checkpoint 在验证集（val split）上统一口径计算指标。

一句话记忆：
  ckpt -> 复原(encoder/model/shape) -> val 推理 -> 指标(MAE/RMSE/PICP/MPIW) -> 写回 ckpt 同目录

常用命令：
  python scripts/eval.py --ckpt outputs/<dataset>/run_YYYYmmdd_HHMMSS/
  python scripts/eval.py --ckpt outputs/<dataset>/run_YYYYmmdd_HHMMSS/checkpoint.pt

两种评估口径（重点）：
  - teacher_forcing：喂真实 `dec_input`（训练口径；通常更像上限/调试）
  - greedy：只给 [SOS] 自回归生成前缀（推理口径；更像真实部署）
  - both：两者都跑，并把结果分别保存

核心流程（从输入到输出，按这个背）：
  1) 读 checkpoint/config：depth/n_bins/v_min/v_max/...（复原形状与编码范围）
  2) load_dataset_split(..., random_state)：构造 val split
  3) TraceLabelEncoder + TabSeqDataset：得到 batch 字段（x_num/x_cat/dec_input/y_clipped/y_raw）
  4) 模型前向：得到 step_probs (B, depth, n_bins)
  5) ExtendedHolographicMetric：step_probs -> leaf_probs -> 点预测/区间 -> MAE/RMSE/PICP/MPIW
  6) 保存 `metrics_val_*.json` + `eval_config_*.json`（可选导出 `leaf_probs_val_*.npz` 用于分析）
"""

import argparse
import json
import os
from typing import Optional, Tuple
from datetime import datetime
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tabseq.data.datasets import load_dataset_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.utils.config import choose, load_config, resolve_section, write_json
from tabseq.utils.seed import set_seed


def _resolve_ckpt_path(path: str) -> str:
    """允许传 run 目录或 checkpoint 文件路径。"""
    if os.path.isdir(path):
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint.pt not found in dir: {path}")
        return ckpt_path
    return path


def _infer_model_class(state_dict: dict):
    """
    从 checkpoint 的 state_dict key 前缀推断模型类型（兼容旧 checkpoint）。

    目前主线模型是 `TransformerTabSeqModel`；历史版本可能用过不同的模块命名前缀，
    所以这里做一个“最小可用”的 pattern-match。
    """
    if any(key.startswith("decoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel
    if any(key.startswith("tabular_encoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel
    if any(key.startswith("encoder.") for key in state_dict.keys()):
        return TransformerTabSeqModel
    raise ValueError("unable to infer model class from checkpoint state_dict keys")


def _maybe_subsample(
    leaf_probs: np.ndarray,
    y_raw: np.ndarray,
    y_clipped: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """导出 `.npz` 时可选抽样，避免文件过大（不影响 json 指标计算）。"""
    if max_samples <= 0 or max_samples >= len(y_raw):
        return leaf_probs, y_raw, y_clipped
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_raw), size=max_samples, replace=False)
    return leaf_probs[idx], y_raw[idx], y_clipped[idx]


def _fmt_float(value: float) -> str:
    """把 float 格式化成适合文件名的 tag（避免 '.' / '-'）。"""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _parse_float_list(text: Optional[str]) -> Optional[list[float]]:
    if text is None:
        return None
    raw = [chunk.strip() for chunk in str(text).split(",")]
    items = [float(chunk) for chunk in raw if chunk]
    return items or None


def _scale_bins(
    *,
    v_min: float,
    v_max: float,
    width_bins: list[float],
    bin_step_02: float,
    bin_step_04: float,
    ref_range: float,
) -> tuple[list[float], float, float]:
    span = float(v_max) - float(v_min)
    if span <= 0 or ref_range <= 0:
        return width_bins, bin_step_02, bin_step_04
    scale = span / float(ref_range)
    return (
        [float(b) * scale for b in width_bins],
        float(bin_step_02) * scale,
        float(bin_step_04) * scale,
    )


def _safe_path(out_dir: str, filename: str) -> str:
    """避免覆盖已有文件：若冲突则自动追加 `_vN` 后缀。"""
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


DEFAULT_WIDTH_BINS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0]
# California Housing v_max - v_min (from config.json) as the reference range.
DEFAULT_REF_RANGE = 5.000010013580322 - 0.1499900072813034


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
    greedy 推理（自回归解码 step_probs）。

    解码协议（很好背）：
      - 每个样本从 `dec_input[:, 0] = [SOS]` 起步
      - 在第 t 层，用当前 `dec_input` 前向得到 `probs_t`（对所有 leaf bin 的软概率）
      - 决策下一位 bit：在“当前可达区间 [start,end)”内，把区间一分为二，
        比较左半 vs 右半的总概率质量（sum），选更大的一侧作为下一步方向
      - 把这个 bit 写回 `dec_input`，进入下一层

    mask_outside（可选，greedy 用）：
      - 若设置为 < 1.0：对每个样本当前不可达区间的 bin 乘上 soft mask，
        让 greedy 的“路径约束”更明确；=1.0 表示不 mask

    返回：
      step_probs: (B, depth, n_bins)  # 每层对所有 leaf bin 的概率
    """
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if mask_outside is not None:
        mask_outside = float(mask_outside)
        if not (0.0 <= mask_outside <= 1.0):
            raise ValueError("mask_outside must be in [0, 1]")

    B = x_num.shape[0]
    device = x_num.device

    # 与训练一致：dec_input = [SOS] + [b0, b1, ..., b_{D-2}]
    dec_input = torch.zeros((B, depth), dtype=torch.long, device=device)
    dec_input[:, 0] = sos_token

    step_probs_out = torch.empty((B, depth, n_bins), dtype=torch.float32, device=device)

    # 每个样本维护自己的可达区间 [start, end)（greedy 走不同路径会导致区间不同）
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
        
        # 记录当前层的 step_probs（注意：这里可能已经应用了 soft mask）
        step_probs_out[:, t, :] = probs_t
        
        if t < depth - 1:
            bits = torch.empty((B,), dtype=torch.long, device=device)
            for b in range(B):
                s, e = start[b], end[b]
                mid = (s + e) // 2
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        required=True,
        help="模型路径：run 目录或 checkpoint.pt 文件路径",
    )
    ap.add_argument("--config", type=str, default=None, help="评估配置文件（YAML/JSON），可覆盖默认参数")
    ap.add_argument("--batch-size", type=int, default=None, help="评估 batch size（默认 256）")
    ap.add_argument("--dataset", type=str, default=None, help="覆盖 checkpoint 里的 dataset 名称（一般不需要）")
    ap.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="数据划分随机种子（默认使用 checkpoint 里的 seed）",
    )
    ap.add_argument("--confidence", type=float, default=None, help="区间置信度，例如 0.90 表示 90% 预测区间")
    ap.add_argument("--temperature", type=float, default=None, help="温度 T：sigmoid(logits/T)，T>1 更平滑")
    ap.add_argument(
        "--interval-method",
        type=str,
        default=None,
        choices=["cdf", "peak_merge"],
        help="区间提取方法：cdf=等尾分位点；peak_merge=波峰合并（不回退）",
    )
    ap.add_argument(
        "--peak-merge-alpha",
        type=float,
        default=None,
        help="peak-merge 阈值比例 alpha（仅 interval-method=peak_merge 时生效）",
    )
    ap.add_argument(
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
    ap.add_argument(
        "--width-bins",
        type=str,
        default=None,
        help="width_stratified_PICP 的分桶边界（逗号分隔，例：0,0.4,0.8,1.2,1.6,2.0,100）",
    )
    ap.add_argument("--bin-step-02", type=float, default=None, help="bin_acc_0.2 的分桶步长")
    ap.add_argument("--bin-step-04", type=float, default=None, help="bin_acc_0.4 的分桶步长")
    ap.add_argument(
        "--scale-diagnostics",
        action="store_true",
        help="按 y 范围相对 ref_range 缩放 width_bins 与 bin_step（适配大尺度数据集）",
    )
    ap.add_argument(
        "--ref-range",
        type=float,
        default=None,
        help="诊断分桶缩放参考范围（默认使用 California Housing 范围）",
    )
    ap.add_argument("--export-leaf-probs", action="store_true", help="导出 leaf_probs 等到 .npz（用于分析/画图）")
    ap.add_argument("--export-samples", type=int, default=0, help="导出时最多保留 N 个样本（0=全部）")
    args = ap.parse_args()

    # 0) 合并配置（优先级：命令行 > config.eval > defaults）
    defaults = {
        "batch_size": 256,
        "confidence": 0.90,
        "temperature": 1.0,
        "interval_method": "cdf",
        "peak_merge_alpha": 0.33,
        "mode": "teacher_forcing",
        "mask_outside": 1.0,
        "width_bins": DEFAULT_WIDTH_BINS,
        "bin_step_02": 0.2,
        "bin_step_04": 0.4,
        "scale_diagnostics": False,
        "ref_range": DEFAULT_REF_RANGE,
        "export_leaf_probs": False,
        "export_samples": 0,
    }
    config = load_config(args.config)
    eval_cfg = resolve_section(config, "eval")
    args.batch_size = choose(args.batch_size, eval_cfg.get("batch_size"), defaults["batch_size"])
    args.confidence = choose(args.confidence, eval_cfg.get("confidence"), defaults["confidence"])
    args.temperature = choose(args.temperature, eval_cfg.get("temperature"), defaults["temperature"])
    args.interval_method = choose(args.interval_method, eval_cfg.get("interval_method"), defaults["interval_method"])
    args.peak_merge_alpha = choose(
        args.peak_merge_alpha, eval_cfg.get("peak_merge_alpha"), defaults["peak_merge_alpha"]
    )
    args.mode = choose(args.mode, eval_cfg.get("mode"), defaults["mode"])
    args.mask_outside = choose(args.mask_outside, eval_cfg.get("mask_outside"), defaults["mask_outside"])
    args.width_bins = choose(args.width_bins, eval_cfg.get("width_bins"), defaults["width_bins"])
    args.bin_step_02 = choose(args.bin_step_02, eval_cfg.get("bin_step_02"), defaults["bin_step_02"])
    args.bin_step_04 = choose(args.bin_step_04, eval_cfg.get("bin_step_04"), defaults["bin_step_04"])
    args.scale_diagnostics = choose(
        args.scale_diagnostics, eval_cfg.get("scale_diagnostics"), defaults["scale_diagnostics"]
    )
    args.ref_range = choose(args.ref_range, eval_cfg.get("ref_range"), defaults["ref_range"])
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

    # 1) 读取 checkpoint/config：用于复原模型形状与编码范围
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    depth = int(cfg["depth"])
    n_bins = int(cfg["n_bins"])
    n_num_features = int(cfg["n_num_features"])
    v_min = float(cfg["v_min"])
    v_max = float(cfg["v_max"])

    width_bins = args.width_bins
    if isinstance(width_bins, str):
        width_bins = _parse_float_list(width_bins)
    elif isinstance(width_bins, (list, tuple)):
        width_bins = [float(x) for x in width_bins]
    if not width_bins:
        width_bins = list(DEFAULT_WIDTH_BINS)

    bin_step_02 = float(args.bin_step_02) if args.bin_step_02 is not None else None
    bin_step_04 = float(args.bin_step_04) if args.bin_step_04 is not None else None
    if bin_step_02 is not None and bin_step_02 <= 0:
        bin_step_02 = None
    if bin_step_04 is not None and bin_step_04 <= 0:
        bin_step_04 = None

    if bool(args.scale_diagnostics):
        width_bins, bin_step_02, bin_step_04 = _scale_bins(
            v_min=v_min,
            v_max=v_max,
            width_bins=width_bins,
            bin_step_02=bin_step_02 or 0.0,
            bin_step_04=bin_step_04 or 0.0,
            ref_range=float(args.ref_range),
        )
        if bin_step_02 == 0.0:
            bin_step_02 = None
        if bin_step_04 == 0.0:
            bin_step_04 = None

    bin_edges_02 = None
    bin_edges_04 = None
    if bin_step_02 is not None:
        bin_edges_02 = np.arange(v_min, v_max + bin_step_02, bin_step_02)
    if bin_step_04 is not None:
        bin_edges_04 = np.arange(v_min, v_max + bin_step_04, bin_step_04)

    random_state = int(args.random_state) if args.random_state is not None else int(cfg.get("seed", 0))
    set_seed(random_state)

    dataset = args.dataset or cfg.get("dataset", "diamonds")
    if args.dataset is not None and cfg.get("dataset") and args.dataset != cfg.get("dataset"):
        raise ValueError(f"dataset mismatch: args.dataset={args.dataset} vs ckpt={cfg.get('dataset')}")

    # 2) 准备验证集 split（注意：标准化统计量来自训练集；由 loader 内部统一处理）
    split = load_dataset_split(dataset, random_state=random_state)
    X_val, y_val = split.X_val, split.y_val
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

    # 3) 构造 Dataset/DataLoader
    #    TabSeqDataset 会把连续 y 编码成 trace 监督（关键字段）：
    #      - dec_input : [SOS] + prefix bits（teacher_forcing 用）
    #      - y_mht     : 每步 multi-hot 监督
    #      - y_raw     : 原始 y（可能超出 [v_min, v_max]）
    #      - y_clipped : 裁剪到 [v_min, v_max] 的 y（指标用这个对齐编码范围）
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)
    ds = TabSeqDataset(
        X_num=X_val,
        X_cat=X_cat_val if X_cat_val is not None else np.zeros((len(y_val), 0), dtype=np.int64),
        y=y_val,
        encoder=enc,
        is_train=False,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 4) 复原模型并切到 eval()（形状必须与训练一致）
    model_cls = _infer_model_class(ckpt["model_state_dict"])
    if model_cls is TransformerTabSeqModel:
        model = model_cls(
            n_num_features=n_num_features,
            depth=depth,
            n_bins=n_bins,
            cat_cardinalities=cat_cardinalities,
            d_model=int(cfg.get("d_model", 64)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5) 指标计算器（统一口径）：step_probs -> leaf_probs -> 点预测/区间 -> MAE/RMSE/PICP/MPIW
    metric_calc = ExtendedHolographicMetric(
        enc,
        width_bins=width_bins,
        bin_edges_02=bin_edges_02,
        bin_edges_04=bin_edges_04,
    )

    out_dir = os.path.dirname(ckpt_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common_tag = f"T{_fmt_float(float(args.temperature))}_C{_fmt_float(float(args.confidence))}"

    def run_mode(mode: str):
        """跑一种 eval mode，返回 metrics dict（必要时导出 npz）。"""
        step_probs_all = []
        y_true_all = []
        y_raw_all = []
        y_clipped_all = []
        oob_count = 0
        total_count = 0

        with torch.no_grad():
            for batch in dl:
                if mode == "teacher_forcing":
                    # 训练口径：直接使用 batch 里的 dec_input 前缀
                    logits = model(batch)
                    step_probs = torch.sigmoid(logits / float(args.temperature))
                elif mode == "greedy":
                    # 推理口径：只给 [SOS]，自回归生成前缀
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

                step_probs_all.append(step_probs.cpu())
                y_true_all.append(batch["y_clipped"].cpu())
                y_raw_all.append(batch["y_raw"].cpu())
                y_clipped_all.append(batch["y_clipped"].cpu())
                if "y_raw" in batch:
                    y_raw = batch["y_raw"]
                    oob = (y_raw < v_min) | (y_raw > v_max)
                    oob_count += int(oob.sum().item())
                    total_count += int(y_raw.numel())

        # 统一口径算指标（在整个 val 上统计）。
        model_probs = torch.cat(step_probs_all, dim=0)
        y_true = torch.cat(y_true_all, dim=0)
        metrics = metric_calc.compute_metrics(
            model_probs=model_probs,
            y_true=y_true,
            confidence=float(args.confidence),
            interval_method=str(args.interval_method),
            peak_merge_alpha=float(args.peak_merge_alpha),
            return_extras=True,
        )
        metrics["model"] = cfg.get("model", "tabseq")
        metrics["mode"] = mode
        metrics["temperature"] = float(args.temperature)
        if total_count > 0:
            metrics["oob_rate"] = float(oob_count / total_count)

        if args.export_leaf_probs:
            # 导出：leaf_probs + y_raw/y_clipped + bin_edges（便于画图/做分析）
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
                "interval_method": str(args.interval_method),
                "peak_merge_alpha": float(args.peak_merge_alpha),
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

    # 写一份“本次 eval 的最终参数”，方便复现实验/对齐产物。
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
            "interval_method": str(args.interval_method),
            "peak_merge_alpha": float(args.peak_merge_alpha),
            "mode": str(args.mode),
            "mask_outside": float(args.mask_outside),
            "width_bins": [float(v) for v in width_bins],
            "bin_step_02": float(bin_step_02) if bin_step_02 is not None else None,
            "bin_step_04": float(bin_step_04) if bin_step_04 is not None else None,
            "scale_diagnostics": bool(args.scale_diagnostics),
            "ref_range": float(args.ref_range),
            "export_leaf_probs": bool(args.export_leaf_probs),
            "export_samples": int(args.export_samples),
        },
    )

    interval_tag = (
        f"pm_a{_fmt_float(float(args.peak_merge_alpha))}" if str(args.interval_method) == "peak_merge" else "cdf"
    )

    if args.mode in ("teacher_forcing", "both"):
        m_tf = run_mode("teacher_forcing")
        print("teacher_forcing:", m_tf)
        tf_tag = f"teacher_forcing_{common_tag}_{interval_tag}_{timestamp}"
        tf_path = _safe_path(out_dir, f"metrics_val_{tf_tag}.json")
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(m_tf, f, ensure_ascii=False, indent=2)
        print("saved:", tf_path)

    if args.mode in ("greedy", "both"):
        m_g = run_mode("greedy")
        print("greedy:", m_g)
        g_tag = f"greedy_{common_tag}_{interval_tag}_mask{_fmt_float(float(args.mask_outside))}_{timestamp}"
        g_path = _safe_path(out_dir, f"metrics_val_{g_tag}.json")
        with open(g_path, "w", encoding="utf-8") as f:
            json.dump(m_g, f, ensure_ascii=False, indent=2)
        print("saved:", g_path)

    if args.mode == "both":
        both = {"teacher_forcing": m_tf, "greedy": m_g}
        out_path = _safe_path(out_dir, f"metrics_val_both_{common_tag}_{timestamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(both, f, ensure_ascii=False, indent=2)
        print("saved:", out_path)


if __name__ == "__main__":
    main()
