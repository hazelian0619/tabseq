import os
import tempfile

from tabseq.training import TrainConfig, train_tabseq_model


def test_train_tabseq_model_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = train_tabseq_model(
            TrainConfig(
                dataset="diabetes",
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
