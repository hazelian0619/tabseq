import numpy as np
import torch
from torch.utils.data import DataLoader

from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.models.transformer_model import TransformerTabSeqModel

enc = TraceLabelEncoder(0, 5, depth=6)

X_num = np.random.randn(8, 4).astype(np.float32)
X_cat = np.zeros((8, 0), dtype=np.int64)
y = np.random.rand(8).astype(np.float32) * 5

ds = TabSeqDataset(X_num=X_num, X_cat=X_cat, y=y, encoder=enc)
dl = DataLoader(ds, batch_size=4, shuffle=False)

batch = next(iter(dl))
model = TransformerTabSeqModel(n_num_features=4, depth=6, n_bins=2**6)

logits = model(batch)
print("logits shape:", logits.shape)