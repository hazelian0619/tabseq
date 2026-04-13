from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tabseq.data.datasets import load_dataset_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.inference import (
    calibrate_confidence_from_leaf_probs,
    collect_beam_outputs,
    constrained_interval_rank,
    interval_metric_rank,
    move_batch_to_device,
    normalize_confidence_grid,
)
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.utils.config import write_json
from tabseq.utils.git import get_git_hash
from tabseq.utils.seed import set_seed


DEFAULT_WIDTH_BINS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0)
BIT_LOSS_WEIGHT = 1.0
MHT_LOSS_WEIGHT = 0.5
LEAF_LOSS_WEIGHT = 0.25
CONSISTENCY_LOSS_WEIGHT = 0.0


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "diamonds"
    seed: int = 0
    val_size: float = 0.2
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    depth: int = 8
    binning_strategy: str = "uniform"
    encoder_type: str = "ft_transformer"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    confidence: float = 0.9
    confidence_grid: tuple[float, ...] = ()
    temperature: float = 1.0
    interval_method: str = "relaxed_mass"
    peak_merge_alpha: float = 0.33
    mask_outside: float = 0.0
    beam_size: int = 8
    leaf_prior_weight: float = 0.0
    selection_warmup_epochs: int = 10
    tolerance_bins: int = 1
    out_root: str = "outputs"
    run_id: Optional[str] = None
    device: Optional[str] = None
    num_workers: int = 0
    bin_step_02: float = 0.2
    bin_step_04: float = 0.4
    width_bins: tuple[float, ...] = DEFAULT_WIDTH_BINS
    model: str = "tabseq_transformer"


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(str(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_run_dir(out_root: str, dataset: str, run_id: Optional[str]) -> tuple[str, str]:
    rid = str(run_id) if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(str(out_root), str(dataset), f"run_{rid}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, rid


def _leaf_branch_right_mass(leaf_logits: torch.Tensor, *, depth: int) -> torch.Tensor:
    leaf_probs = torch.softmax(leaf_logits, dim=1)
    n_bins = int(leaf_probs.shape[1])
    shifts = torch.arange(int(depth) - 1, -1, -1, dtype=torch.long, device=leaf_logits.device)
    leaf_indices = torch.arange(n_bins, dtype=torch.long, device=leaf_logits.device).unsqueeze(1)
    bit_table = ((leaf_indices >> shifts) & 1).float()
    return leaf_probs @ bit_table


def _run_epoch(
    *,
    model: TransformerTabSeqModel,
    dl: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    total_loss = 0.0
    total_loss_bit = 0.0
    total_loss_mht = 0.0
    total_loss_leaf = 0.0
    total_loss_consistency = 0.0
    total_count = 0
    train_mode = optimizer is not None
    model.train(train_mode)

    for batch in dl:
        batch_dev = move_batch_to_device(batch, device)
        outputs = model(batch_dev)
        if isinstance(outputs, dict):
            mht_logits = outputs["mht_logits"]
            bit_logits = outputs["bit_logits"]
            leaf_logits = outputs["leaf_logits"]

            loss_bit = F.binary_cross_entropy_with_logits(bit_logits, batch_dev["y_seq"].float())
            loss_mht = F.binary_cross_entropy_with_logits(mht_logits, batch_dev["y_mht"])
            loss_leaf = F.cross_entropy(leaf_logits, batch_dev["y_leaf_idx"])
            loss_consistency = F.mse_loss(
                torch.sigmoid(bit_logits),
                _leaf_branch_right_mass(leaf_logits, depth=int(bit_logits.shape[1])),
            )
            loss = (
                (BIT_LOSS_WEIGHT * loss_bit)
                + (MHT_LOSS_WEIGHT * loss_mht)
                + (LEAF_LOSS_WEIGHT * loss_leaf)
                + (CONSISTENCY_LOSS_WEIGHT * loss_consistency)
            )
        else:
            loss_bit = torch.zeros((), device=device)
            loss = F.binary_cross_entropy_with_logits(outputs, batch_dev["y_mht"])
            loss_mht = loss
            loss_leaf = torch.zeros((), device=device)
            loss_consistency = torch.zeros((), device=device)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = int(batch_dev["x_num"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_loss_bit += float(loss_bit.item()) * batch_size
        total_loss_mht += float(loss_mht.item()) * batch_size
        total_loss_leaf += float(loss_leaf.item()) * batch_size
        total_loss_consistency += float(loss_consistency.item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise ValueError("empty dataloader")
    return {
        "loss": total_loss / total_count,
        "loss_bit": total_loss_bit / total_count,
        "loss_mht": total_loss_mht / total_count,
        "loss_leaf": total_loss_leaf / total_count,
        "loss_consistency": total_loss_consistency / total_count,
    }


def _is_better_checkpoint(
    candidate: Dict[str, Any],
    best: Optional[Dict[str, Any]],
    *,
    confidence: float,
    tolerance_bins: int,
    candidate_val_loss: float,
    best_val_loss: float,
) -> bool:
    if best is None:
        return True
    candidate_rank = constrained_interval_rank(
        candidate,
        target_confidence=float(confidence),
        tolerance_bins=int(tolerance_bins),
    )
    best_rank = constrained_interval_rank(
        best,
        target_confidence=float(confidence),
        tolerance_bins=int(tolerance_bins),
    )
    if candidate_rank != best_rank:
        return candidate_rank > best_rank
    if float(candidate_val_loss) != float(best_val_loss):
        return float(candidate_val_loss) < float(best_val_loss)
    candidate_rank = interval_metric_rank(candidate, confidence=float(confidence), tolerance_bins=int(tolerance_bins))
    best_rank = interval_metric_rank(best, confidence=float(confidence), tolerance_bins=int(tolerance_bins))
    return candidate_rank > best_rank


def _write_git_txt(run_dir: str) -> None:
    with open(os.path.join(run_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(get_git_hash() + "\n")


def train_tabseq_model(config: TrainConfig) -> str:
    set_seed(int(config.seed))
    device = _resolve_device(config.device)
    run_dir, resolved_run_id = _make_run_dir(config.out_root, config.dataset, config.run_id)

    split = load_dataset_split(config.dataset, random_state=int(config.seed), val_size=float(config.val_size))
    encoder = TraceLabelEncoder.from_targets(
        split.y_train,
        depth=int(config.depth),
        strategy=str(config.binning_strategy),
    )
    v_min = float(encoder.v_min)
    v_max = float(encoder.v_max)

    train_ds = TabSeqDataset(
        X_num=split.X_train,
        X_cat=split.X_cat_train,
        y=split.y_train,
        encoder=encoder,
        is_train=True,
        precompute_mht=True,
    )
    val_ds = TabSeqDataset(
        X_num=split.X_val,
        X_cat=split.X_cat_val,
        y=split.y_val,
        encoder=encoder,
        is_train=False,
        precompute_mht=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
    )

    model = TransformerTabSeqModel(
        n_num_features=int(split.X_train.shape[1]),
        depth=int(config.depth),
        n_bins=2 ** int(config.depth),
        cat_cardinalities=split.cat_cardinalities,
        encoder_type=str(config.encoder_type),
        d_model=int(config.d_model),
        n_heads=int(config.n_heads),
        n_layers=int(config.n_layers),
        dropout=float(config.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

    metric_calc = ExtendedHolographicMetric(
        encoder,
        width_bins=list(config.width_bins),
        bin_edges_02=(
            None
            if float(config.bin_step_02) <= 0
            else np.arange(v_min, v_max + float(config.bin_step_02), float(config.bin_step_02), dtype=np.float32)
        ),
        bin_edges_04=(
            None
            if float(config.bin_step_04) <= 0
            else np.arange(v_min, v_max + float(config.bin_step_04), float(config.bin_step_04), dtype=np.float32)
        ),
    )

    history = []
    best_val_loss = float("inf")
    best_val_metrics: Optional[Dict[str, Any]] = None
    best_epoch = 0
    selection_start_epoch = max(1, min(int(config.selection_warmup_epochs), int(config.epochs)))
    confidence_grid = normalize_confidence_grid(config.confidence_grid, target_confidence=float(config.confidence))

    cfg: Dict[str, Any] = {
        "dataset": config.dataset,
        "model": config.model,
        "run_id": resolved_run_id,
        "seed": int(config.seed),
        "val_size": float(config.val_size),
        "batch_size": int(config.batch_size),
        "epochs": int(config.epochs),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "depth": int(config.depth),
        "n_bins": 2 ** int(config.depth),
        "binning_strategy": str(config.binning_strategy),
        "bin_edges": [float(x) for x in encoder.get_all_bin_edges().tolist()],
        "encoder_type": str(config.encoder_type),
        "d_model": int(config.d_model),
        "n_heads": int(config.n_heads),
        "n_layers": int(config.n_layers),
        "dropout": float(config.dropout),
        "confidence": float(config.confidence),
        "confidence_grid": [float(x) for x in confidence_grid],
        "temperature": float(config.temperature),
        "interval_method": str(config.interval_method),
        "peak_merge_alpha": float(config.peak_merge_alpha),
        "mask_outside": float(config.mask_outside),
        "beam_size": int(config.beam_size),
        "leaf_prior_weight": float(config.leaf_prior_weight),
        "selection_warmup_epochs": int(config.selection_warmup_epochs),
        "bit_loss_weight": float(BIT_LOSS_WEIGHT),
        "mht_loss_weight": float(MHT_LOSS_WEIGHT),
        "leaf_loss_weight": float(LEAF_LOSS_WEIGHT),
        "consistency_loss_weight": float(CONSISTENCY_LOSS_WEIGHT),
        "tolerance_bins": int(config.tolerance_bins),
        "v_min": v_min,
        "v_max": v_max,
        "n_num_features": int(split.X_train.shape[1]),
        "cat_cardinalities": list(split.cat_cardinalities),
        "device": str(device),
        "width_bins": list(config.width_bins),
        "bin_step_02": float(config.bin_step_02),
        "bin_step_04": float(config.bin_step_04),
    }

    for epoch in range(1, int(config.epochs) + 1):
        train_loss = _run_epoch(model=model, dl=train_dl, device=device, optimizer=optimizer)
        val_loss = _run_epoch(model=model, dl=val_dl, device=device, optimizer=None)
        val_outputs = collect_beam_outputs(
            model=model,
            dl=val_dl,
            device=device,
            depth=int(config.depth),
            n_bins=2 ** int(config.depth),
            temperature=float(config.temperature),
            beam_size=int(config.beam_size),
            leaf_prior_weight=float(config.leaf_prior_weight),
            mask_outside=float(config.mask_outside),
        )
        raw_val_metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
            leaf_probs=val_outputs["leaf_probs"],
            y_true=val_outputs["y_raw"],
            y_leaf_idx=val_outputs["y_leaf_idx"],
            confidence=float(config.confidence),
            interval_method=str(config.interval_method),
            peak_merge_alpha=float(config.peak_merge_alpha),
            tolerance_bins=int(config.tolerance_bins),
        )
        val_metrics = calibrate_confidence_from_leaf_probs(
            metric_calc=metric_calc,
            leaf_probs=val_outputs["leaf_probs"],
            y_true=val_outputs["y_raw"],
            y_leaf_idx=val_outputs["y_leaf_idx"],
            target_confidence=float(config.confidence),
            confidence_grid=confidence_grid,
            interval_method=str(config.interval_method),
            peak_merge_alpha=float(config.peak_merge_alpha),
            tolerance_bins=int(config.tolerance_bins),
        )
        val_metrics.update(
            {
                "mode": "beam",
                "beam_size": int(config.beam_size),
                "leaf_prior_weight": float(config.leaf_prior_weight),
                "temperature": float(config.temperature),
                "interval_method": str(config.interval_method),
                "mask_outside": float(config.mask_outside),
                "raw_confidence": float(config.confidence),
                "raw_avg_coverage": float(raw_val_metrics["avg_coverage"]),
                "raw_avg_length": float(raw_val_metrics["avg_length"]),
            }
        )
        if str(config.interval_method) == "peak_merge":
            val_metrics["peak_merge_alpha"] = float(config.peak_merge_alpha)

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss["loss"]),
            "train_loss_bit": float(train_loss["loss_bit"]),
            "train_loss_mht": float(train_loss["loss_mht"]),
            "train_loss_leaf": float(train_loss["loss_leaf"]),
            "train_loss_consistency": float(train_loss["loss_consistency"]),
            "val_loss": float(val_loss["loss"]),
            "val_loss_bit": float(val_loss["loss_bit"]),
            "val_loss_mht": float(val_loss["loss_mht"]),
            "val_loss_leaf": float(val_loss["loss_leaf"]),
            "val_loss_consistency": float(val_loss["loss_consistency"]),
            "val_bin_acc": float(raw_val_metrics["bin_acc"]),
            f"val_tol_bin_acc@{int(config.tolerance_bins)}": float(
                raw_val_metrics[f"tol_bin_acc@{int(config.tolerance_bins)}"]
            ),
            "val_avg_coverage": float(raw_val_metrics["avg_coverage"]),
            "val_avg_length": float(raw_val_metrics["avg_length"]),
            "val_calibrated_confidence": float(val_metrics["calibrated_confidence"]),
            "val_calibrated_coverage": float(val_metrics["avg_coverage"]),
            "val_calibrated_length": float(val_metrics["avg_length"]),
        }
        history.append(record)
        print(
            "[TRAIN] "
            f"epoch={epoch}/{int(config.epochs)} "
            f"train_loss={record['train_loss']:.6f} "
            f"(bit={record['train_loss_bit']:.6f}, mht={record['train_loss_mht']:.6f}, leaf={record['train_loss_leaf']:.6f}, cons={record['train_loss_consistency']:.6f}) "
            f"val_loss={record['val_loss']:.6f} "
            f"(bit={record['val_loss_bit']:.6f}, mht={record['val_loss_mht']:.6f}, leaf={record['val_loss_leaf']:.6f}, cons={record['val_loss_consistency']:.6f}) "
            f"val_bin_acc={record['val_bin_acc']:.4f} "
            f"val_cov={record['val_avg_coverage']:.4f} "
            f"val_len={record['val_avg_length']:.4f} "
            f"cal_conf={record['val_calibrated_confidence']:.3f} "
            f"cal_cov={record['val_calibrated_coverage']:.4f} "
            f"cal_len={record['val_calibrated_length']:.4f} "
            f"val_mode=beam{int(config.beam_size)} "
            f"leaf_prior={float(config.leaf_prior_weight):.2f}",
            flush=True,
        )

        if epoch >= selection_start_epoch and _is_better_checkpoint(
            val_metrics,
            best_val_metrics,
            confidence=float(config.confidence),
            tolerance_bins=int(config.tolerance_bins),
            candidate_val_loss=float(val_loss["loss"]),
            best_val_loss=float(best_val_loss),
        ):
            best_val_loss = float(val_loss["loss"])
            best_val_metrics = dict(val_metrics)
            best_epoch = int(epoch)
            ckpt_cfg = dict(cfg)
            ckpt_cfg["calibrated_confidence"] = float(val_metrics["calibrated_confidence"])
            ckpt = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": float(best_val_loss),
                "best_val_metrics": best_val_metrics,
                "config": ckpt_cfg,
            }
            torch.save(ckpt, os.path.join(run_dir, "checkpoint.pt"))

    if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
        raise RuntimeError("training finished without writing checkpoint.pt")

    best_ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_confidence = float(best_ckpt.get("config", {}).get("calibrated_confidence", config.confidence))
    best_outputs = collect_beam_outputs(
        model=model,
        dl=val_dl,
        device=device,
        depth=int(config.depth),
        n_bins=2 ** int(config.depth),
        temperature=float(config.temperature),
        beam_size=int(config.beam_size),
        leaf_prior_weight=float(config.leaf_prior_weight),
        mask_outside=float(config.mask_outside),
    )
    raw_best_metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
        leaf_probs=best_outputs["leaf_probs"],
        y_true=best_outputs["y_raw"],
        y_leaf_idx=best_outputs["y_leaf_idx"],
        confidence=float(config.confidence),
        interval_method=str(config.interval_method),
        peak_merge_alpha=float(config.peak_merge_alpha),
        tolerance_bins=int(config.tolerance_bins),
    )
    best_metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
        leaf_probs=best_outputs["leaf_probs"],
        y_true=best_outputs["y_raw"],
        y_leaf_idx=best_outputs["y_leaf_idx"],
        confidence=float(best_confidence),
        interval_method=str(config.interval_method),
        peak_merge_alpha=float(config.peak_merge_alpha),
        tolerance_bins=int(config.tolerance_bins),
    )
    best_metrics.update(
        {
            "dataset": config.dataset,
            "model": config.model,
            "run_id": resolved_run_id,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "confidence": float(best_confidence),
            "target_confidence": float(config.confidence),
            "calibrated_confidence": float(best_confidence),
            "confidence_grid": [float(x) for x in confidence_grid],
            "raw_confidence": float(config.confidence),
            "raw_avg_coverage": float(raw_best_metrics["avg_coverage"]),
            "raw_avg_length": float(raw_best_metrics["avg_length"]),
            "mode": "beam",
            "beam_size": int(config.beam_size),
            "leaf_prior_weight": float(config.leaf_prior_weight),
            "temperature": float(config.temperature),
            "interval_method": str(config.interval_method),
            "mask_outside": float(config.mask_outside),
        }
    )
    if str(config.interval_method) == "peak_merge":
        best_metrics["peak_merge_alpha"] = float(config.peak_merge_alpha)

    write_json(os.path.join(run_dir, "config.json"), cfg)
    write_json(
        os.path.join(run_dir, "eval_config.json"),
        {
            "dataset": config.dataset,
            "random_state": int(config.seed),
            "batch_size": int(config.batch_size),
            "confidence": float(best_confidence),
            "target_confidence": float(config.confidence),
            "confidence_grid": [float(x) for x in confidence_grid],
            "encoder_type": str(config.encoder_type),
            "temperature": float(config.temperature),
            "interval_method": str(config.interval_method),
            "peak_merge_alpha": float(config.peak_merge_alpha),
            "mode": "beam",
            "beam_size": int(config.beam_size),
            "leaf_prior_weight": float(config.leaf_prior_weight),
            "selection_warmup_epochs": int(config.selection_warmup_epochs),
            "mask_outside": float(config.mask_outside),
            "tolerance_bins": int(config.tolerance_bins),
            "width_bins": list(config.width_bins),
            "bin_step_02": float(config.bin_step_02),
            "bin_step_04": float(config.bin_step_04),
        },
    )
    write_json(os.path.join(run_dir, "history.json"), {"history": history})
    write_json(os.path.join(run_dir, "metrics_val_beam.json"), best_metrics)
    write_json(
        os.path.join(run_dir, "train_summary.json"),
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_bin_acc": float(best_metrics["bin_acc"]),
            f"best_tol_bin_acc@{int(config.tolerance_bins)}": float(best_metrics[f"tol_bin_acc@{int(config.tolerance_bins)}"]),
            "best_avg_coverage": float(best_metrics["avg_coverage"]),
            "best_avg_length": float(best_metrics["avg_length"]),
            "best_mode": "beam",
            "beam_size": int(config.beam_size),
            "binning_strategy": str(config.binning_strategy),
            "target_confidence": float(config.confidence),
            "calibrated_confidence": float(best_confidence),
            "confidence_grid": [float(x) for x in confidence_grid],
            "raw_best_avg_coverage": float(raw_best_metrics["avg_coverage"]),
            "raw_best_avg_length": float(raw_best_metrics["avg_length"]),
            "leaf_prior_weight": float(config.leaf_prior_weight),
            "selection_warmup_epochs": int(config.selection_warmup_epochs),
            "device": str(device),
            "config": asdict(config),
        },
    )
    _write_git_txt(run_dir)

    return run_dir
