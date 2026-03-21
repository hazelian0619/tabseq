import numpy as np
import torch

from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder


def test_tabseq_dataset_smoke_shapes_and_shift() -> None:
    rng = np.random.default_rng(0)
    n = 64
    d = 8
    depth = 4

    X_num = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)

    enc = TraceLabelEncoder(v_min=float(y.min()), v_max=float(y.max()), depth=depth)
    ds = TabSeqDataset(X_num=X_num, X_cat=None, y=y, encoder=enc, is_train=True, sos_token=2)

    sample = ds[0]
    assert set(sample.keys()) >= {"x_num", "x_cat", "dec_input", "y_seq", "y_mht", "y_raw", "y_clipped"}

    assert tuple(sample["x_num"].shape) == (d,)
    assert tuple(sample["x_cat"].shape) == (0,)
    assert tuple(sample["dec_input"].shape) == (depth,)
    assert tuple(sample["y_seq"].shape) == (depth,)
    assert tuple(sample["y_mht"].shape) == (depth, 2**depth)

    # dec_input = [SOS] + y_seq[:-1]
    assert int(sample["dec_input"][0].item()) == 2
    assert torch.equal(sample["dec_input"][1:], sample["y_seq"][:-1])


def test_tabseq_dataset_mht_precompute_matches_on_the_fly() -> None:
    rng = np.random.default_rng(1)
    n = 32
    d = 3
    depth = 5

    X_num = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)

    enc = TraceLabelEncoder(v_min=float(y.min()), v_max=float(y.max()), depth=depth)
    # Force precompute path so we cover both code paths deterministically.
    ds = TabSeqDataset(
        X_num=X_num,
        X_cat=None,
        y=y,
        encoder=enc,
        precompute_mht=True,
    )
    assert ds.y_multi_hots is not None

    idx = 7
    leaf_idx = int(ds.y_leaf_idx[idx].item())
    mht_pre = ds.y_multi_hots[idx]
    mht_fly = torch.from_numpy(enc.encode_multi_hot(leaf_idx))
    assert torch.allclose(mht_pre, mht_fly)


def test_tabseq_dataloader_batch_shapes() -> None:
    rng = np.random.default_rng(2)
    n = 40
    d = 6
    depth = 4
    bs = 8

    X_num = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n,)).astype(np.float32)

    enc = TraceLabelEncoder(v_min=float(y.min()), v_max=float(y.max()), depth=depth)
    ds = TabSeqDataset(X_num=X_num, X_cat=None, y=y, encoder=enc, is_train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

    batch = next(iter(dl))
    assert tuple(batch["x_num"].shape) == (bs, d)
    assert tuple(batch["x_cat"].shape) == (bs, 0)
    assert tuple(batch["dec_input"].shape) == (bs, depth)
    assert tuple(batch["y_seq"].shape) == (bs, depth)
    assert tuple(batch["y_mht"].shape) == (bs, depth, 2**depth)
    assert tuple(batch["y_raw"].shape) == (bs,)
    assert tuple(batch["y_clipped"].shape) == (bs,)

