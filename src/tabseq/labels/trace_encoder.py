from typing import List, Tuple, Dict, Any
import numpy as np


class TraceLabelEncoder:
    """
    将连续标签 y 映射到二叉树叶子桶：
    - 基础输出：二进制决策序列 sequence + 叶子索引 leaf_idx
    - 进阶输出：多热标签 multi_hot（每一步的合法区间）
    """

    def __init__(self, v_min: float, v_max: float, depth: int = 10):
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.depth = int(depth)
        self.n_bins = 2 ** self.depth
        self.range = self.v_max - self.v_min

        if self.range <= 0:
            # 退化情况：所有 y 都映射到同一个桶
            self.bin_width = 0.0
            self._degenerate = True
        else:
            self.bin_width = self.range / self.n_bins
            self._degenerate = False

    # ===== 基础内部编码逻辑 =====
    def _encode_base(self, y: float) -> Tuple[List[int], int]:
        """
        只做 y → (sequence, leaf_idx)，不计算 multi_hot。
        保持为最小、稳定的核心逻辑。
        """
        if self._degenerate:
            # 退化情况：没有有效区间，统一映射到 0
            return [0] * self.depth, 0

        # 1) 先把 y 限制在 [v_min, v_max]
        y_clipped = float(np.clip(y, self.v_min, self.v_max))

        # 2) 归一化到 [0, 1]
        norm_y = (y_clipped - self.v_min) / self.range  # ∈ [0, 1]

        # 3) 计算 bin 索引，并用 clip 防止浮点误差导致越界
        raw_idx = np.floor(norm_y * self.n_bins)
        leaf_idx = int(np.clip(raw_idx, 0, self.n_bins - 1))

        # 4) 转成二进制序列（长度 = depth）
        binary_str = format(leaf_idx, f"0{self.depth}b")
        sequence = [int(bit) for bit in binary_str]

        return sequence, leaf_idx

    # ===== 对外基础接口（保持兼容） =====
    def encode(self, y: float) -> Tuple[List[int], int]:
        """
        基础接口：只返回决策序列和叶子索引。
        兼容原有调用方式。
        """
        return self._encode_base(y)

    # ===== 对外高级接口（推荐新代码使用） =====
    def encode_full(self, y: float) -> Dict[str, Any]:
        """
        高级接口：一次性返回所有常用信息。

        返回:
            {
                "sequence": List[int],          # 决策序列（长度 = depth）
                "leaf_idx": int,                # 叶子桶索引 [0, n_bins-1]
                "multi_hot": np.ndarray,        # (depth, n_bins) 每步合法区间
                "y_center": float,              # 对应桶中心值
            }
        """
        sequence, leaf_idx = self._encode_base(y)
        multi_hot = self.encode_multi_hot(leaf_idx)
        y_center = self.decode_bin_index(leaf_idx)

        return {
            "sequence": sequence,
            "leaf_idx": leaf_idx,
            "multi_hot": multi_hot,
            "y_center": y_center,
        }

    # ===== 多热标签编码 =====
    def encode_multi_hot(self, leaf_idx: int) -> np.ndarray:
        """
        给定叶子索引，生成每一步的“半区间多热标签”：
        - 形状: (depth, n_bins)
        - 第 t 步：标记当前还可能包含目标叶子的所有 bin = 1，其余为 0
        """
        multi_hot = np.zeros((self.depth, self.n_bins), dtype=np.float32)
        start, end = 0, self.n_bins

        for t in range(self.depth):
            mid = (start + end) // 2
            if leaf_idx < mid:
                # 走左子树：当前合法范围是 [start, mid)
                multi_hot[t, start:mid] = 1.0
                end = mid
            else:
                # 走右子树：当前合法范围是 [mid, end)
                multi_hot[t, mid:end] = 1.0
                start = mid

        return multi_hot

    # ===== 解码相关工具 =====
    def decode_bin_index(self, bin_idx: int) -> float:
        """
        给定叶子索引，返回该桶的中心点值。
        """
        return self.v_min + (bin_idx + 0.5) * self.bin_width

    def decode_sequence(self, sequence: List[int]) -> float:
        """
        给定二进制决策序列，恢复叶子索引并返回桶中心值。
        """
        bin_idx = 0
        for bit in sequence:
            bin_idx = (bin_idx << 1) | bit
        return self.decode_bin_index(bin_idx)

    def get_bin_edges(self, bin_idx: int) -> Tuple[float, float]:
        """
        给定叶子索引，返回该桶的左右边界 [lower, upper)。
        """
        lower = self.v_min + bin_idx * self.bin_width
        upper = lower + self.bin_width
        return lower, upper
