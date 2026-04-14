from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from tabseq.labels.trace_encoder import TraceLabelEncoder


class TabSeqDataset(Dataset):
    def __init__(
        self,
        *,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        y: np.ndarray,
        encoder: TraceLabelEncoder,
        is_train: bool = True,
        sos_token: int = 2,
        precompute_mht: bool = True,
    ):
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        n = int(y_arr.shape[0])

        if X_num is None:
            x_num_arr = np.zeros((n, 0), dtype=np.float32)
        else:
            x_num_arr = np.asarray(X_num, dtype=np.float32)
        if x_num_arr.shape[0] != n:
            raise ValueError(f"X_num.shape[0]={x_num_arr.shape[0]} must match len(y)={n}")

        if X_cat is None:
            x_cat_arr = np.zeros((n, 0), dtype=np.int64)
        else:
            x_cat_arr = np.asarray(X_cat, dtype=np.int64)
        if x_cat_arr.shape[0] != n:
            raise ValueError(f"X_cat.shape[0]={x_cat_arr.shape[0]} must match len(y)={n}")

        self.X_num = torch.from_numpy(x_num_arr).float()
        self.X_cat = torch.from_numpy(x_cat_arr).long()
        self.y_raw = torch.from_numpy(y_arr).float()
        self.y_clipped = torch.clamp(self.y_raw, min=float(encoder.v_min), max=float(encoder.v_max))

        self.encoder = encoder
        self.is_train = bool(is_train)
        self.sos_token = int(sos_token)
        self.precompute_mht = bool(precompute_mht)

        y_seqs = []
        y_leaf_idx = []
        y_multi_hots = []
        for value in self.y_clipped.tolist():
            seq, leaf_idx = self.encoder.encode(float(value))
            y_seqs.append(seq)
            y_leaf_idx.append(leaf_idx)
            if self.precompute_mht:
                y_multi_hots.append(self.encoder.encode_multi_hot(leaf_idx))

        self.y_seqs = torch.tensor(y_seqs, dtype=torch.long)
        self.y_leaf_idx = torch.tensor(y_leaf_idx, dtype=torch.long)
        if self.precompute_mht:
            self.y_multi_hots: Optional[torch.Tensor] = torch.from_numpy(np.asarray(y_multi_hots, dtype=np.float32))
        else:
            self.y_multi_hots = None

    def __len__(self) -> int:
        return int(self.y_raw.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target_seq = self.y_seqs[idx]
        dec_input = torch.empty_like(target_seq)
        dec_input[0] = self.sos_token
        dec_input[1:] = target_seq[:-1]

        y_mht = self.y_multi_hots[idx] if self.y_multi_hots is not None else torch.from_numpy(
            self.encoder.encode_multi_hot(int(self.y_leaf_idx[idx].item()))
        )

        return {
            "x_num": self.X_num[idx],
            "x_cat": self.X_cat[idx],
            "dec_input": dec_input,
            "y_seq": target_seq,
            "y_mht": y_mht.float(),
            "y_leaf_idx": self.y_leaf_idx[idx],
            "y_raw": self.y_raw[idx],
            "y_clipped": self.y_clipped[idx],
        }
