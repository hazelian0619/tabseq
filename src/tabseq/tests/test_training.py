import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from tabseq.data import datasets as datasets_module
from tabseq.training import TrainConfig, train_tabseq_model


def test_train_tabseq_model_smoke(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir) / "data"
        dataset_dir = data_root / "toy_regression"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(
            {
                "x_num": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "x_cat": ["a", "a", "b", "b", "c", "c"],
                "target": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            }
        )
        frame.to_csv(dataset_dir / "table.csv.gz", index=False, compression="gzip")
        with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump({"target_name": "target"}, f)

        monkeypatch.setattr(datasets_module, "DEFAULT_LOCAL_DATA_ROOTS", (data_root,))

        run_dir = train_tabseq_model(
            TrainConfig(
                dataset="toy_regression",
                epochs=1,
                batch_size=64,
                depth=4,
                encoder_type="ft_transformer",
                d_model=16,
                n_heads=4,
                n_layers=1,
                dropout=0.0,
                out_root=tmpdir,
                run_id="smoke",
                device="cpu",
            )
        )

        assert os.path.isdir(run_dir)
        assert os.path.isfile(os.path.join(run_dir, "checkpoint.pt"))
        assert os.path.isfile(os.path.join(run_dir, "config.json"))
        assert os.path.isfile(os.path.join(run_dir, "eval_config.json"))
        assert os.path.isfile(os.path.join(run_dir, "metrics_val_beam.json"))
