import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from jingwei.data_io import load_matrix
from jingwei.datasets import CSVDataset
from jingwei.models import DMF
from jingwei.train import ImputationTrainer, parse_arguments

torch.set_num_threads(1)


def test_standalone_dmf_get_imputed_data_replaces_only_unobserved_entries():
    model = DMF(n_rows=4, n_cols=3, embedding_dim=4, hidden_dims=[8], mask_predictor_hidden_dim=8)
    data = torch.tensor(
        [[1.0, 2.0, 0.0], [3.0, 0.0, 5.0], [6.0, 7.0, 8.0], [0.0, 9.0, 10.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    imputed = model.get_imputed_data(data, mask)
    expected_full = model.get_full_matrix()
    assert imputed.shape == (4, 3)
    assert torch.equal(imputed[mask == 1], data[mask == 1])
    assert torch.allclose(imputed[mask == 0], expected_full[mask == 0])

    with pytest.raises(ValueError, match="must match"):
        model.get_imputed_data(torch.zeros((2, 3)), torch.ones((2, 3)))


def test_csv_dataset_inverse_transform_aligns_dtype_and_device(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    pd.DataFrame(
        [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0], [30.0, 60.0, 90.0]],
        index=["S1", "S2", "S3"],
        columns=["P1", "P2", "P3"],
    ).to_csv(csv_path)
    dataset = CSVDataset(csv_path, seed=7)
    float64_input = dataset.data_normalized.to(dtype=torch.float64)
    restored = dataset.inverse_transform(float64_input)
    assert restored.dtype == torch.float64
    assert torch.allclose(
        restored[dataset.supervision_mask.bool()],
        dataset.data_filled.to(dtype=torch.float64)[dataset.supervision_mask.bool()],
    )
    if torch.cuda.is_available():
        cuda_restored = dataset.inverse_transform(dataset.data_normalized.cuda())
        assert cuda_restored.is_cuda


def test_core_data_io_handles_blank_lines_zero_padded_ids_memory_limit_and_layer_only_h5ad(
    tmp_path, monkeypatch
):
    blank_csv = tmp_path / "blank_lines.csv"
    blank_csv.write_text("\ufeff\n   \nID,P1,P2\n001,2,4\n010,8,16\n", encoding="utf-8")
    loaded = load_matrix(blank_csv, index_column="first", transform="none")
    assert list(loaded.frame.index) == ["001", "010"]
    np.testing.assert_array_equal(loaded.frame.to_numpy(), [[2.0, 4.0], [8.0, 16.0]])

    with pytest.raises(ValueError, match="max-dense-mb"):
        load_matrix(blank_csv, max_dense_mb=0)

    large_csv = tmp_path / "large.csv"
    pd.DataFrame(np.ones((400, 400))).to_csv(large_csv, index=False)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pytest.fail("oversize CSV reached pandas"))
    with pytest.raises(ValueError, match="max-dense-mb"):
        load_matrix(large_csv, max_dense_mb=1)

    ad = pytest.importorskip("anndata")
    h5ad_path = tmp_path / "layer_only.h5ad"
    adata = ad.AnnData(shape=(2, 3))
    adata.layers["raw_counts"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    adata.write_h5ad(h5ad_path)
    with pytest.raises(ValueError, match="--layer"):
        load_matrix(h5ad_path)


@pytest.mark.parametrize("method", ["DMF", "DCAE"])
def test_train_cli_writes_flat_checkpoints_utf8_config_and_sample_ids(
    tmp_path, monkeypatch, method
):
    csv_path = tmp_path / "蛋白数据.csv"
    pd.DataFrame(
        [[1.5, 2.5, 3.5, 4.5], [2.5, 0.0, 4.5, 5.5], [3.5, 4.5, 5.5, 6.5], [4.5, 5.5, 6.5, 7.5]],
        index=["样本_01", "样本_02", "样本_03", "样本_04"],
        columns=["P1", "P2", "P3", "P4"],
    ).to_csv(csv_path, encoding="utf-8")
    results_dir = tmp_path / "输出目录"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jingwei.train",
            "--data-path",
            str(csv_path),
            "--method",
            method,
            "--max-epochs",
            "1",
            "--batch-size",
            "2",
            "--embedding-dim",
            "4",
            "--latent-dim",
            "4",
            "--hidden-dims",
            "8",
            "--device",
            "cpu",
            "--results-dir",
            str(results_dir),
        ],
    )
    args = parse_arguments()
    args.data_path = Path(args.data_path)
    before_keys = set(vars(args).keys())
    trainer = ImputationTrainer(args)
    trainer.train()

    assert set(vars(args).keys()) == before_keys
    ckpt_subdirs = [p for p in (results_dir / "checkpoints").iterdir() if p.is_dir()]
    ckpt_files = list((results_dir / "checkpoints").glob("*.ckpt"))
    assert not ckpt_subdirs
    assert len(ckpt_files) >= 1

    config_data = json.loads((trainer.run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_data["dataset_name"] == "蛋白数据"
    imputed_df = pd.read_csv(trainer.run_dir / "蛋白数据_imputed.csv")
    assert imputed_df.iloc[:, 0].tolist() == ["样本_01", "样本_02", "样本_03", "样本_04"]
    assert imputed_df.shape == (4, 5)
