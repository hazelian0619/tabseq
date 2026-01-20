from typing import List, Tuple
import numpy as np

class TraceLabelEncoder:
    def __init__(self, v_min: float, v_max: float, depth: int = 10):
        self.v_min = v_min
        self.v_max = v_max
        self.depth = depth
        self.n_bins = 2 ** depth 
        self.bin_width = (v_max - v_min) / self.n_bins

    def encode(self, y: float) -> Tuple[List[int], int]:
        y = np.clip(y, self.v_min, self.v_max)
        norm_y = (y - self.v_min) / (self.v_max - self.v_min)
        leaf_idx = int(min(np.floor(norm_y * self.n_bins), self.n_bins - 1))
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
        return self.v_min + (bin_idx + 0.5) * self.bin_width

    def decode_sequence(self, sequence: List[int]) -> float:
        bin_idx = 0
        for bit in sequence:
            bin_idx = (bin_idx << 1) | bit
        return self.decode_bin_index(bin_idx)

    def get_bin_edges(self, bin_idx: int) -> Tuple[float, float]:
        lower = self.v_min + bin_idx * self.bin_width
        upper = lower + self.bin_width
        return lower, upper