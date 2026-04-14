import numpy as np

from tabseq.labels.trace_encoder import TraceLabelEncoder


def test_trace_encoder_round_trip_stays_inside_bin() -> None:
    enc = TraceLabelEncoder(v_min=0.0, v_max=1.0, depth=4)
    seq, leaf_idx = enc.encode(0.37)

    assert len(seq) == 4
    assert 0 <= leaf_idx < 16

    decoded = enc.decode_sequence(seq)
    lo, hi = enc.get_bin_edges(leaf_idx)
    assert lo <= decoded <= hi


def test_trace_encoder_multi_hot_halves_each_step() -> None:
    enc = TraceLabelEncoder(v_min=-1.0, v_max=1.0, depth=5)
    mht = enc.encode_multi_hot(leaf_idx=19)

    assert mht.shape == (5, 32)
    assert np.allclose(mht.sum(axis=1), np.array([16, 8, 4, 2, 1], dtype=np.float32))


def test_trace_encoder_quantile_bins_are_strictly_increasing() -> None:
    y = np.array([1.0, 2.0, 2.5, 4.0, 6.0, 9.0, 15.0, 30.0], dtype=np.float32)
    enc = TraceLabelEncoder.from_targets(y, depth=3, strategy="quantile")

    edges = enc.get_all_bin_edges()
    assert edges.shape == (9,)
    assert np.all(np.diff(edges) > 0)
    assert enc.binning_strategy == "quantile"
