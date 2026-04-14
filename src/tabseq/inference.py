from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader


def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def default_confidence_grid(target_confidence: float) -> tuple[float, ...]:
    target_confidence = float(target_confidence)
    offsets = tuple(round(x, 2) for x in (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))
    values = sorted(
        {
            round(min(0.99, max(0.50, target_confidence + delta)), 6)
            for delta in offsets
        }
    )
    return tuple(values)


def normalize_confidence_grid(
    confidence_grid: Optional[Sequence[float]],
    *,
    target_confidence: float,
) -> tuple[float, ...]:
    values = confidence_grid if confidence_grid else default_confidence_grid(float(target_confidence))
    normalized = []
    seen = set()
    for value in values:
        conf = round(float(value), 6)
        if not (0.0 < conf < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {conf}")
        if conf in seen:
            continue
        seen.add(conf)
        normalized.append(conf)
    if not normalized:
        raise ValueError("confidence_grid must contain at least one valid confidence")
    return tuple(sorted(normalized))


def _unpack_model_outputs(outputs: Any) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(outputs, Mapping):
        return outputs["mht_logits"], outputs.get("bit_logits"), outputs.get("leaf_logits")
    return outputs, None, None


def _validate_temperature(temperature: float) -> float:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return temperature


def _validate_mask_outside(mask_outside: float) -> float:
    mask_outside = float(mask_outside)
    if not (0.0 <= mask_outside <= 1.0):
        raise ValueError("mask_outside must be in [0, 1]")
    return mask_outside


def _validate_leaf_prior_weight(leaf_prior_weight: float) -> float:
    leaf_prior_weight = float(leaf_prior_weight)
    if leaf_prior_weight < 0:
        raise ValueError("leaf_prior_weight must be >= 0")
    return leaf_prior_weight


def _apply_range_mask(
    probs_t: torch.Tensor,
    *,
    start: list[int],
    end: list[int],
    mask_outside: float,
) -> torch.Tensor:
    if mask_outside >= 1.0:
        return probs_t

    masked = torch.full_like(probs_t, fill_value=float(mask_outside))
    for i, (s, e) in enumerate(zip(start, end)):
        masked[i, s:e] = 1.0
    return probs_t * masked


def _branch_probs_from_state(
    *,
    probs_t: Optional[torch.Tensor],
    bit_prob_right: Optional[float],
    start: int,
    end: int,
    leaf_prob_left: Optional[float] = None,
    leaf_prob_right: Optional[float] = None,
    leaf_prior_weight: float = 0.0,
) -> tuple[float, float]:
    eps = 1e-9
    mid = (int(start) + int(end)) // 2

    left_mht = None
    right_mht = None
    if probs_t is not None:
        left_slice = probs_t[int(start) : mid]
        right_slice = probs_t[mid : int(end)]
        left_mht = float(left_slice.mean().item()) if left_slice.numel() > 0 else 0.0
        right_mht = float(right_slice.mean().item()) if right_slice.numel() > 0 else 0.0

    if bit_prob_right is None:
        if left_mht is None or right_mht is None:
            return 0.5, 0.5
        total = left_mht + right_mht
        if total <= eps:
            return 0.5, 0.5
        return left_mht / total, right_mht / total

    left_bit = max(1.0 - float(bit_prob_right), eps)
    right_bit = max(float(bit_prob_right), eps)
    if left_mht is None or right_mht is None:
        total = left_bit + right_bit
        return left_bit / total, right_bit / total

    left_score = left_bit * max(left_mht, eps)
    right_score = right_bit * max(right_mht, eps)
    total = left_score + right_score
    if total <= eps:
        total = left_bit + right_bit
        return left_bit / total, right_bit / total
    left_model = left_score / total
    right_model = right_score / total

    if leaf_prob_left is None or leaf_prob_right is None or float(leaf_prior_weight) <= 0.0:
        return left_model, right_model

    left_score = max(left_model, eps) * (max(float(leaf_prob_left), eps) ** float(leaf_prior_weight))
    right_score = max(right_model, eps) * (max(float(leaf_prob_right), eps) ** float(leaf_prior_weight))
    total = left_score + right_score
    if total <= eps:
        return left_model, right_model
    return left_score / total, right_score / total


@torch.no_grad()
def greedy_step_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    *,
    depth: int,
    n_bins: int,
    temperature: float,
    mask_outside: float,
    sos_token: int = 2,
) -> torch.Tensor:
    temperature = _validate_temperature(temperature)
    mask_outside = _validate_mask_outside(mask_outside)

    batch_size = int(x_num.shape[0])
    device = x_num.device

    dec_input = torch.zeros((batch_size, int(depth)), dtype=torch.long, device=device)
    dec_input[:, 0] = int(sos_token)
    step_probs_out = torch.empty((batch_size, int(depth), int(n_bins)), dtype=torch.float32, device=device)

    start = [0 for _ in range(batch_size)]
    end = [int(n_bins) for _ in range(batch_size)]

    for t in range(int(depth)):
        outputs = model({"x_num": x_num, "x_cat": x_cat, "dec_input": dec_input})
        mht_logits, bit_logits, _ = _unpack_model_outputs(outputs)
        probs_t = torch.sigmoid(mht_logits[:, t, :] / temperature)
        probs_t = _apply_range_mask(probs_t, start=start, end=end, mask_outside=mask_outside)

        step_probs_out[:, t, :] = probs_t

        if t < int(depth) - 1:
            bits = torch.empty((batch_size,), dtype=torch.long, device=device)
            if bit_logits is not None:
                bit_probs_t = torch.sigmoid(bit_logits[:, t] / temperature)
                bits.copy_((bit_probs_t >= 0.5).long())
                for b in range(batch_size):
                    s, e = start[b], end[b]
                    mid = (s + e) // 2
                    if int(bits[b].item()) == 0:
                        end[b] = mid
                    else:
                        start[b] = mid
            else:
                for b in range(batch_size):
                    s, e = start[b], end[b]
                    mid = (s + e) // 2
                    left_mass = float(probs_t[b, s:mid].sum().item())
                    right_mass = float(probs_t[b, mid:e].sum().item())
                    bit = 1 if right_mass > left_mass else 0
                    bits[b] = bit
                    if bit == 0:
                        end[b] = mid
                    else:
                        start[b] = mid
            dec_input[:, t + 1] = bits

    return step_probs_out


@torch.no_grad()
def beam_leaf_probs(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: Optional[torch.Tensor],
    *,
    depth: int,
    n_bins: int,
    temperature: float,
    beam_size: int,
    leaf_prior_weight: float,
    mask_outside: float,
    sos_token: int = 2,
) -> torch.Tensor:
    temperature = _validate_temperature(temperature)
    mask_outside = _validate_mask_outside(mask_outside)
    leaf_prior_weight = _validate_leaf_prior_weight(leaf_prior_weight)
    beam_size = int(beam_size)
    if beam_size <= 0:
        raise ValueError("beam_size must be > 0")

    batch_size = int(x_num.shape[0])
    device = x_num.device
    eps = 1e-9

    init_dec_input = torch.full((batch_size, int(depth)), fill_value=int(sos_token), dtype=torch.long, device=device)
    init_outputs = model({"x_num": x_num, "x_cat": x_cat, "dec_input": init_dec_input})
    _, _, init_leaf_logits = _unpack_model_outputs(init_outputs)
    leaf_prior = None
    if init_leaf_logits is not None and float(leaf_prior_weight) > 0.0:
        leaf_prior = torch.softmax(init_leaf_logits, dim=1)

    states: list[list[dict[str, Any]]] = [
        [{"bits": (), "start": 0, "end": int(n_bins), "logprob": 0.0}] for _ in range(batch_size)
    ]

    for t in range(int(depth)):
        prefix_bits: list[tuple[int, ...]] = []
        prefix_start: list[int] = []
        prefix_end: list[int] = []
        prefix_logprob: list[float] = []
        prefix_sample: list[int] = []

        for sample_idx, sample_states in enumerate(states):
            for state in sample_states:
                prefix_bits.append(tuple(state["bits"]))
                prefix_start.append(int(state["start"]))
                prefix_end.append(int(state["end"]))
                prefix_logprob.append(float(state["logprob"]))
                prefix_sample.append(sample_idx)

        if not prefix_bits:
            raise RuntimeError("beam search produced no active states")

        num_prefixes = len(prefix_bits)
        dec_input = torch.zeros((num_prefixes, int(depth)), dtype=torch.long, device=device)
        dec_input[:, 0] = int(sos_token)
        if t > 0:
            bits_tensor = torch.tensor(prefix_bits, dtype=torch.long, device=device)
            dec_input[:, 1 : t + 1] = bits_tensor

        sample_idx_tensor = torch.tensor(prefix_sample, dtype=torch.long, device=device)
        x_rep = x_num[sample_idx_tensor]
        x_cat_rep = x_cat[sample_idx_tensor] if x_cat is not None else None

        outputs = model({"x_num": x_rep, "x_cat": x_cat_rep, "dec_input": dec_input})
        mht_logits, bit_logits, _ = _unpack_model_outputs(outputs)
        probs_t = torch.sigmoid(mht_logits[:, t, :] / temperature)
        probs_t = _apply_range_mask(probs_t, start=prefix_start, end=prefix_end, mask_outside=mask_outside)

        if bit_logits is not None:
            bit_probs_right = torch.sigmoid(bit_logits[:, t] / temperature)
        else:
            bit_probs_right = None

        next_states: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        for i in range(num_prefixes):
            s = prefix_start[i]
            e = prefix_end[i]
            mid = (s + e) // 2
            right_bit_prob = None if bit_probs_right is None else float(bit_probs_right[i].item())
            sample_idx = prefix_sample[i]
            leaf_left = None
            leaf_right = None
            if leaf_prior is not None:
                leaf_span = leaf_prior[sample_idx, s:e]
                leaf_mass = float(leaf_span.sum().item())
                if leaf_mass > eps:
                    left_mass = float(leaf_prior[sample_idx, s:mid].sum().item()) / leaf_mass
                    right_mass = float(leaf_prior[sample_idx, mid:e].sum().item()) / leaf_mass
                    leaf_left = left_mass
                    leaf_right = right_mass
            left_prob, right_prob = _branch_probs_from_state(
                probs_t=probs_t[i],
                bit_prob_right=right_bit_prob,
                start=s,
                end=e,
                leaf_prob_left=leaf_left,
                leaf_prob_right=leaf_right,
                leaf_prior_weight=float(leaf_prior_weight),
            )
            logprob = prefix_logprob[i]
            bits = prefix_bits[i]

            next_states[sample_idx].append(
                {
                    "bits": bits + (0,),
                    "start": s,
                    "end": mid,
                    "logprob": logprob + math.log(max(left_prob, eps)),
                }
            )
            next_states[sample_idx].append(
                {
                    "bits": bits + (1,),
                    "start": mid,
                    "end": e,
                    "logprob": logprob + math.log(max(right_prob, eps)),
                }
            )

        states = []
        for sample_states in next_states:
            if not sample_states:
                raise RuntimeError("beam search pruning produced an empty state set")
            sample_states = sorted(sample_states, key=lambda item: float(item["logprob"]), reverse=True)
            states.append(sample_states[:beam_size])

    leaf_probs = torch.zeros((batch_size, int(n_bins)), dtype=torch.float32, device=device)
    for sample_idx, sample_states in enumerate(states):
        logprobs = torch.tensor([float(state["logprob"]) for state in sample_states], dtype=torch.float32, device=device)
        weights = torch.softmax(logprobs, dim=0)
        for state, weight in zip(sample_states, weights):
            leaf_idx = int(state["start"])
            leaf_probs[sample_idx, leaf_idx] += float(weight.item())

    return leaf_probs


@torch.no_grad()
def collect_greedy_outputs(
    *,
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    depth: int,
    n_bins: int,
    temperature: float,
    mask_outside: float,
    sos_token: int = 2,
) -> dict[str, torch.Tensor]:
    model.eval()

    step_probs_all = []
    y_raw_all = []
    y_leaf_idx_all = []

    for batch in dl:
        batch_dev = move_batch_to_device(batch, device)
        step_probs = greedy_step_probs(
            model=model,
            x_num=batch_dev["x_num"],
            x_cat=batch_dev.get("x_cat"),
            depth=int(depth),
            n_bins=int(n_bins),
            temperature=float(temperature),
            mask_outside=float(mask_outside),
            sos_token=int(sos_token),
        )
        step_probs_all.append(step_probs.cpu())
        y_raw_all.append(batch["y_raw"].cpu())
        y_leaf_idx_all.append(batch["y_leaf_idx"].cpu())

    if not step_probs_all:
        raise ValueError("empty dataloader")

    return {
        "model_probs": torch.cat(step_probs_all, dim=0),
        "y_raw": torch.cat(y_raw_all, dim=0),
        "y_leaf_idx": torch.cat(y_leaf_idx_all, dim=0),
    }


@torch.no_grad()
def collect_beam_outputs(
    *,
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    depth: int,
    n_bins: int,
    temperature: float,
    beam_size: int,
    leaf_prior_weight: float,
    mask_outside: float,
    sos_token: int = 2,
) -> dict[str, torch.Tensor]:
    model.eval()

    leaf_probs_all = []
    y_raw_all = []
    y_leaf_idx_all = []

    for batch in dl:
        batch_dev = move_batch_to_device(batch, device)
        leaf_probs = beam_leaf_probs(
            model=model,
            x_num=batch_dev["x_num"],
            x_cat=batch_dev.get("x_cat"),
            depth=int(depth),
            n_bins=int(n_bins),
            temperature=float(temperature),
            beam_size=int(beam_size),
            leaf_prior_weight=float(leaf_prior_weight),
            mask_outside=float(mask_outside),
            sos_token=int(sos_token),
        )
        leaf_probs_all.append(leaf_probs.cpu())
        y_raw_all.append(batch["y_raw"].cpu())
        y_leaf_idx_all.append(batch["y_leaf_idx"].cpu())

    if not leaf_probs_all:
        raise ValueError("empty dataloader")

    return {
        "leaf_probs": torch.cat(leaf_probs_all, dim=0),
        "y_raw": torch.cat(y_raw_all, dim=0),
        "y_leaf_idx": torch.cat(y_leaf_idx_all, dim=0),
    }


def compute_greedy_bin_interval_metrics(
    *,
    model: torch.nn.Module,
    dl: DataLoader,
    metric_calc: Any,
    device: torch.device,
    depth: int,
    n_bins: int,
    confidence: float,
    temperature: float,
    interval_method: str,
    peak_merge_alpha: float,
    mask_outside: float,
    tolerance_bins: int = 1,
    sos_token: int = 2,
) -> dict[str, Any]:
    outputs = collect_greedy_outputs(
        model=model,
        dl=dl,
        device=device,
        depth=int(depth),
        n_bins=int(n_bins),
        temperature=float(temperature),
        mask_outside=float(mask_outside),
        sos_token=int(sos_token),
    )
    metrics = metric_calc.compute_bin_interval_metrics(
        model_probs=outputs["model_probs"],
        y_true=outputs["y_raw"],
        y_leaf_idx=outputs["y_leaf_idx"],
        confidence=float(confidence),
        interval_method=str(interval_method),
        peak_merge_alpha=float(peak_merge_alpha),
        tolerance_bins=int(tolerance_bins),
    )
    metrics.update(
        {
            "mode": "greedy",
            "temperature": float(temperature),
            "interval_method": str(interval_method),
            "mask_outside": float(mask_outside),
        }
    )
    if str(interval_method) == "peak_merge":
        metrics["peak_merge_alpha"] = float(peak_merge_alpha)
    return metrics


def compute_beam_bin_interval_metrics(
    *,
    model: torch.nn.Module,
    dl: DataLoader,
    metric_calc: Any,
    device: torch.device,
    depth: int,
    n_bins: int,
    confidence: float,
    temperature: float,
    beam_size: int,
    leaf_prior_weight: float,
    interval_method: str,
    peak_merge_alpha: float,
    mask_outside: float,
    tolerance_bins: int = 1,
    sos_token: int = 2,
) -> dict[str, Any]:
    outputs = collect_beam_outputs(
        model=model,
        dl=dl,
        device=device,
        depth=int(depth),
        n_bins=int(n_bins),
        temperature=float(temperature),
        beam_size=int(beam_size),
        leaf_prior_weight=float(leaf_prior_weight),
        mask_outside=float(mask_outside),
        sos_token=int(sos_token),
    )
    metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
        leaf_probs=outputs["leaf_probs"],
        y_true=outputs["y_raw"],
        y_leaf_idx=outputs["y_leaf_idx"],
        confidence=float(confidence),
        interval_method=str(interval_method),
        peak_merge_alpha=float(peak_merge_alpha),
        tolerance_bins=int(tolerance_bins),
    )
    metrics.update(
        {
            "mode": "beam",
            "beam_size": int(beam_size),
            "leaf_prior_weight": float(leaf_prior_weight),
            "temperature": float(temperature),
            "interval_method": str(interval_method),
            "mask_outside": float(mask_outside),
        }
    )
    if str(interval_method) == "peak_merge":
        metrics["peak_merge_alpha"] = float(peak_merge_alpha)
    return metrics


def interval_metric_rank(
    metrics: Mapping[str, Any],
    *,
    confidence: float,
    tolerance_bins: int = 1,
) -> tuple[float, float, float, float]:
    tol_key = f"tol_bin_acc@{int(tolerance_bins)}"
    return (
        -abs(float(metrics["avg_coverage"]) - float(confidence)),
        -float(metrics["avg_length"]),
        float(metrics["bin_acc"]),
        float(metrics[tol_key]),
    )


def constrained_interval_rank(
    metrics: Mapping[str, Any],
    *,
    target_confidence: float,
    tolerance_bins: int = 1,
) -> tuple[float, float, float, float, float]:
    tol_key = f"tol_bin_acc@{int(tolerance_bins)}"
    coverage = float(metrics["avg_coverage"])
    length = float(metrics["avg_length"])
    bin_acc = float(metrics["bin_acc"])
    tol_bin_acc = float(metrics[tol_key])
    if coverage >= float(target_confidence):
        return (
            1.0,
            -length,
            bin_acc,
            tol_bin_acc,
            -(coverage - float(target_confidence)),
        )
    return (
        0.0,
        coverage,
        -length,
        bin_acc,
        tol_bin_acc,
    )


def calibrate_confidence_from_leaf_probs(
    *,
    metric_calc: Any,
    leaf_probs: torch.Tensor,
    y_true: torch.Tensor,
    y_leaf_idx: torch.Tensor,
    target_confidence: float,
    confidence_grid: Optional[Sequence[float]],
    interval_method: str,
    peak_merge_alpha: float,
    tolerance_bins: int = 1,
) -> dict[str, Any]:
    target_confidence = float(target_confidence)
    grid = normalize_confidence_grid(confidence_grid, target_confidence=target_confidence)
    best_metrics: Optional[dict[str, Any]] = None

    for confidence in grid:
        metrics = metric_calc.compute_bin_interval_metrics_from_leaf_probs(
            leaf_probs=leaf_probs,
            y_true=y_true,
            y_leaf_idx=y_leaf_idx,
            confidence=float(confidence),
            interval_method=str(interval_method),
            peak_merge_alpha=float(peak_merge_alpha),
            tolerance_bins=int(tolerance_bins),
        )
        metrics["confidence"] = float(confidence)
        if best_metrics is None:
            best_metrics = metrics
            continue
        if constrained_interval_rank(metrics, target_confidence=target_confidence, tolerance_bins=int(tolerance_bins)) > constrained_interval_rank(
            best_metrics,
            target_confidence=target_confidence,
            tolerance_bins=int(tolerance_bins),
        ):
            best_metrics = metrics

    assert best_metrics is not None
    calibrated = dict(best_metrics)
    calibrated["target_confidence"] = float(target_confidence)
    calibrated["calibrated_confidence"] = float(best_metrics["confidence"])
    calibrated["confidence_grid"] = [float(x) for x in grid]
    return calibrated


def greedy_metric_rank(
    metrics: Mapping[str, Any],
    *,
    confidence: float,
    tolerance_bins: int = 1,
) -> tuple[float, float, float, float]:
    return interval_metric_rank(metrics, confidence=float(confidence), tolerance_bins=int(tolerance_bins))


def beam_metric_rank(
    metrics: Mapping[str, Any],
    *,
    confidence: float,
    tolerance_bins: int = 1,
) -> tuple[float, float, float, float]:
    return interval_metric_rank(metrics, confidence=float(confidence), tolerance_bins=int(tolerance_bins))
