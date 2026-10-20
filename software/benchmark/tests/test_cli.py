import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from jingwei_benchmark.cli import main


@pytest.mark.parametrize('mechanism', ['mcar', 'mnar-jingwei-v2'])
def test_all_35_methods_receive_identical_single_mask_input(tmp_path,monkeypatch,mechanism):
    import hashlib
    import jingwei_benchmark.benchmark as orchestration
    from jingwei_benchmark.methods import METHODS
    source=fixture(tmp_path)
    received=[]
    def availability(*args,names=None,**kwargs):
        return {name:{'available':True,'reason':None} for name in names}
    def worker(payload,*args):
        path=__import__('pathlib').Path(payload['input_file'])
        with np.load(path) as bundle:
            assert not bundle['training_hidden'].any()
            values=bundle['values'].copy()
        received.append((payload['method'],path,hashlib.sha256(path.read_bytes()).hexdigest()))
        filled=np.where(np.isfinite(values),values,np.nanmean(values,axis=0)[None,:])
        np.savez(__import__('pathlib').Path(payload['folder'])/'predictions.npz',values=filled)
        return {'status':'completed','time_s':0.}
    monkeypatch.setattr(orchestration,'availability',availability)
    monkeypatch.setattr(orchestration,'_run_worker',worker)
    output=tmp_path/'all-methods'
    assert main(['benchmark','--input',str(source),'--methods','all','--device','cpu',
                 '--mechanism',mechanism,'--output',str(output)])==0
    assert {r[0] for r in received}==set(METHODS) and len(received)==35
    assert len({(r[1],r[2]) for r in received})==1
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['mask_protocol_version']==3
    assert manifest['effective_protocol']['platform_mask_count']==1


def fixture(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame(
        np.random.default_rng(7).uniform(10, 100, (8, 6)),
        index=[f"S{i}" for i in range(8)],
        columns=[f"P{i}" for i in range(6)],
    ).to_csv(source)
    return source


def test_native_benchmark_emits_replayable_masks_and_real_metrics(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "run"
    assert (
        main(
            [
                "benchmark",
                "--input",
                str(source),
                "--methods",
                "mean",
                "knn",
                "--repeats",
                "2",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 4 and (metrics.status == "completed").all()
    assert np.isfinite(metrics.RMSE).all() and np.isfinite(metrics.W2).all()
    assert len(pd.read_csv(output / "paired_metrics.csv")) == 4
    for repeat in range(2):
        folder = output / f"repeat_{repeat:03d}"
        masks = np.load(folder / "masks.npz", allow_pickle=False)
        sent = np.load(folder / "method_input.npz", allow_pickle=False)
        assert np.isnan(sent["values"][masks["evaluation_hidden"]]).all()
        assert not (masks["evaluation_hidden"] & masks["training_hidden"]).any()
        assert not masks["training_hidden"].any()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["input_sha256"]) == 64
    assert (output / "report.html").is_file()


def test_missing_r_dependency_is_reported_as_skip_and_nonzero_exit(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "run"
    assert (
        main(
            [
                "benchmark",
                "--input",
                str(source),
                "--methods",
                "mean",
                "bpca",
                "--rscript",
                str(tmp_path / "missing.exe"),
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    metrics = pd.read_csv(output / "metrics.csv")
    assert metrics.set_index("method").loc["bpca", "status"] == "skipped"
    assert pd.read_csv(output / "paired_metrics.csv").empty


def test_imputation_preserves_observations_and_has_no_evaluation_mask(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "run"
    frame = pd.read_csv(source, index_col=0)
    frame.iloc[0, 0] = np.nan
    frame.to_csv(source)
    assert (
        main(
            [
                "impute",
                "--input",
                str(source),
                "--methods",
                "mean",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    values = pd.read_csv(
        output / "repeat_000" / "mean" / "imputed.csv", index_col=0
    ).to_numpy()
    np.testing.assert_allclose(
        values[np.isfinite(frame.to_numpy())],
        frame.to_numpy()[np.isfinite(frame.to_numpy())],
    )
    assert np.isfinite(values).all()
    assert not np.load(output / "repeat_000" / "masks.npz")["evaluation_hidden"].any()
    assert not np.load(output / "repeat_000" / "masks.npz")["training_hidden"].any()


def test_existing_output_is_not_overwritten(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    assert main(["benchmark", "--input", str(source), "--output", str(output)]) == 1
    assert (output / "keep.txt").read_text() == "keep"


def test_method_timeout_remains_visible_and_returns_incomplete_status(tmp_path):
    source = fixture(tmp_path)
    output = tmp_path / "run"
    assert (
        main(
            [
                "benchmark",
                "--input",
                str(source),
                "--methods",
                "mean",
                "--timeout",
                "0.01",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    result = json.loads((output / "repeat_000" / "mean" / "result.json").read_text())
    assert result["status"] == "timeout"
    assert not (output / "repeat_000" / "mean" / "predictions.npz").exists()


def test_constant_minimum_predictions_have_null_correlations(tmp_path):
    rng = np.random.default_rng(42)
    values = rng.lognormal(8, 0.65, (16, 80))
    values[rng.random(values.shape) < 0.025] = np.nan
    values[:4] = rng.lognormal(8, 0.65, (4, 80))
    source = tmp_path / "demo.csv"
    pd.DataFrame(values, index=[f"S{i}" for i in range(16)], columns=[f"P{i}" for i in range(80)]).to_csv(source)
    output = tmp_path / "constant-run"
    assert main(["benchmark", "--input", str(source), "--methods", "minimum", "--device", "cpu",
                 "--mask-rate", ".05", "--repeats", "1", "--output", str(output)]) == 0
    record = json.loads((output / "metrics.json").read_text())[0]
    assert record["Pearson"] is None and record["Spearman"] is None
    assert np.isfinite(record["RMSE"])
