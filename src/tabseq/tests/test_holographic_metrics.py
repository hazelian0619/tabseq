import numpy as np
import torch

from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric


def test_compute_bin_interval_metrics_from_leaf_probs() -> None:
    enc = TraceLabelEncoder(v_min=0.0, v_max=8.0, depth=3)
    metric = ExtendedHolographicMetric(enc)

    leaf_probs = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor([5.7, 5.2], dtype=torch.float32)
    y_leaf_idx = torch.tensor([5, 5], dtype=torch.long)

    metrics = metric.compute_bin_interval_metrics_from_leaf_probs(
        leaf_probs,
        y_true,
        y_leaf_idx,
        confidence=0.9,
        interval_method="cdf",
        tolerance_bins=1,
    )

    assert metrics["bin_acc"] == 0.5
    assert metrics["tol_bin_acc@1"] == 1.0
    assert metrics["avg_coverage"] == 0.5
    assert metrics["avg_length"] == 1.0


def test_shortest_mass_interval_prefers_compact_high_mass_region() -> None:
    enc = TraceLabelEncoder(v_min=0.0, v_max=8.0, depth=3)
    metric = ExtendedHolographicMetric(enc)

    leaf_probs = np.array([[0.05, 0.05, 0.35, 0.35, 0.05, 0.15, 0.0, 0.0]], dtype=np.float32)
    bin_edges = np.arange(9, dtype=np.float32)

    lower, upper = metric._interval_bounds_from_leaf_probs_np(
        leaf_probs,
        bin_edges=bin_edges,
        confidence=0.7,
        interval_method="shortest_mass",
        peak_merge_alpha=0.33,
    )

    assert float(lower[0]) == 2.0
    assert float(upper[0]) == 4.0


def test_relaxed_mass_interval_falls_back_to_global_target_mass_when_local_spans_are_insufficient() -> None:
    enc = TraceLabelEncoder(v_min=0.0, v_max=8.0, depth=3)
    metric = ExtendedHolographicMetric(enc)

    leaf_probs = np.array([[0.32, 0.31, 0.02, 0.0, 0.0, 0.18, 0.17, 0.0]], dtype=np.float32)
    bin_edges = np.arange(9, dtype=np.float32)

    lower, upper = metric._interval_bounds_from_leaf_probs_np(
        leaf_probs,
        bin_edges=bin_edges,
        confidence=0.9,
        interval_method="relaxed_mass",
        peak_merge_alpha=0.33,
    )

    assert float(lower[0]) == 0.0
    assert float(upper[0]) == 7.0


def test_relaxed_mass_interval_can_merge_short_gap_between_neighboring_segments() -> None:
    enc = TraceLabelEncoder(v_min=0.0, v_max=8.0, depth=3)
    metric = ExtendedHolographicMetric(enc)

    leaf_probs = np.array([[0.27, 0.26, 0.02, 0.25, 0.20, 0.0, 0.0, 0.0]], dtype=np.float32)
    bin_edges = np.arange(9, dtype=np.float32)

    lower, upper = metric._interval_bounds_from_leaf_probs_np(
        leaf_probs,
        bin_edges=bin_edges,
        confidence=0.9,
        interval_method="relaxed_mass",
        peak_merge_alpha=0.33,
    )

    assert float(lower[0]) == 0.0
    assert float(upper[0]) == 5.0
