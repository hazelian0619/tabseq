import numpy as np
import pytest
import torch

from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.models.transformer_model import TransformerTabSeqModel


@pytest.mark.parametrize("encoder_type", ["vanilla", "ft_transformer"])
def test_transformer_tabseq_smoke_forward_backward(encoder_type: str) -> None:
    rng = np.random.default_rng(0)
    X_num = rng.normal(size=(12, 5)).astype(np.float32)
    y = rng.normal(size=(12,)).astype(np.float32)

    depth = 4
    enc = TraceLabelEncoder(v_min=float(y.min()), v_max=float(y.max()), depth=depth)
    ds = TabSeqDataset(X_num=X_num, X_cat=None, y=y, encoder=enc, is_train=True)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)))

    model = TransformerTabSeqModel(
        n_num_features=5,
        depth=depth,
        n_bins=2**depth,
        encoder_type=encoder_type,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    outputs = model(batch)

    assert outputs["mht_logits"].shape == (4, depth, 2**depth)
    assert outputs["bit_logits"].shape == (4, depth)
    assert outputs["leaf_logits"].shape == (4, 2**depth)

    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(outputs["mht_logits"], batch["y_mht"])
        + torch.nn.functional.binary_cross_entropy_with_logits(outputs["bit_logits"], batch["y_seq"].float())
        + torch.nn.functional.cross_entropy(outputs["leaf_logits"], batch["y_leaf_idx"])
    )
    loss.backward()
