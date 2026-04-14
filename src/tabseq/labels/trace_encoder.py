from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

class TraceLabelEncoder:
    def __init__(
        self,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        depth: int = 10,
        *,
        bin_edges: Optional[np.ndarray] = None,
        binning_strategy: str = "uniform",
    ):
        self.depth = int(depth)
        self.n_bins = 2 ** int(depth)
        self.binning_strategy = str(binning_strategy)

        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=np.float32).reshape(-1)
            if edges.shape[0] != self.n_bins + 1:
                raise ValueError(f"bin_edges must have length {self.n_bins + 1}, got {edges.shape[0]}")
            if not np.all(np.isfinite(edges)):
                raise ValueError("bin_edges must be finite")
            if np.any(np.diff(edges) <= 0):
                raise ValueError("bin_edges must be strictly increasing")
            self.bin_edges = edges
        else:
            if v_min is None or v_max is None:
                raise ValueError("v_min/v_max are required when bin_edges is not provided")
            lo = float(v_min)
            hi = float(v_max)
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError("v_min/v_max must be finite")
            if hi <= lo:
                raise ValueError(f"v_max must be > v_min, got {hi} <= {lo}")
            self.bin_edges = np.linspace(lo, hi, self.n_bins + 1, dtype=np.float32)

        self.v_min = float(self.bin_edges[0])
        self.v_max = float(self.bin_edges[-1])
        self.bin_width = float(np.mean(np.diff(self.bin_edges)))
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @classmethod
    def from_targets(
        cls,
        y: np.ndarray,
        *,
        depth: int,
        strategy: str = "uniform",
    ) -> "TraceLabelEncoder":
        values = np.asarray(y, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("cannot build TraceLabelEncoder from empty targets")

        strategy = str(strategy)
        if strategy == "uniform":
            return cls(v_min=float(np.min(values)), v_max=float(np.max(values)), depth=int(depth), binning_strategy=strategy)
        if strategy == "quantile":
            return cls(
                depth=int(depth),
                bin_edges=cls._build_quantile_edges(values, n_bins=2 ** int(depth)),
                binning_strategy=strategy,
            )
        raise ValueError(f"unknown binning strategy: {strategy}")

    @staticmethod
    def _build_quantile_edges(values: np.ndarray, *, n_bins: int) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            raise ValueError("values must contain at least one finite target")

        q = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
        try:
            edges = np.quantile(flat, q, method="linear")
        except TypeError:
            edges = np.quantile(flat, q, interpolation="linear")

        if np.any(np.diff(edges) <= 0):
            unique = np.unique(flat)
            if unique.size >= 2:
                xp = np.linspace(0.0, 1.0, unique.size, dtype=np.float64)
                edges = np.interp(q, xp, unique)
            else:
                center = float(unique[0])
                delta = max(abs(center) * 1e-3, 1e-3)
                edges = np.linspace(center - delta, center + delta, int(n_bins) + 1, dtype=np.float64)

        if np.any(np.diff(edges) <= 0):
            raise ValueError("failed to construct strictly increasing quantile bin edges")
        return edges.astype(np.float32)

    def encode(self, y: float) -> Tuple[List[int], int]:
        y = float(np.clip(y, self.v_min, self.v_max))
        leaf_idx = int(np.searchsorted(self.bin_edges, y, side="right") - 1)
        leaf_idx = int(np.clip(leaf_idx, 0, self.n_bins - 1))
        binary_str = format(leaf_idx, f'0{self.depth}b')
        sequence = [int(bit) for bit in binary_str]
        return sequence, leaf_idx
    
    def encode_multi_hot(self, leaf_idx: int) -> np.ndarray:
        multi_hot = np.zeros((self.depth, self.n_bins), dtype=np.float32)
        start, end = 0, self.n_bins
        for t in range(self.depth):
            mid = (start + end) // 2
            if leaf_idx < mid:
                multi_hot[t, start:mid] = 1.0
                end = mid
            else:
                multi_hot[t, mid:end] = 1.0
                start = mid
        return multi_hot

    def decode_bin_index(self, bin_idx: int) -> float:
        return float(self.bin_centers[int(bin_idx)])

    def decode_sequence(self, sequence: List[int]) -> float:
        bin_idx = 0
        for bit in sequence:
            bin_idx = (bin_idx << 1) | bit
        return self.decode_bin_index(bin_idx)

    def get_bin_edges(self, bin_idx: int) -> Tuple[float, float]:
        idx = int(bin_idx)
        return float(self.bin_edges[idx]), float(self.bin_edges[idx + 1])

    def get_all_bin_edges(self) -> np.ndarray:
        return self.bin_edges.copy()
