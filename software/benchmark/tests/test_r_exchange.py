from pathlib import Path

import numpy as np
import pytest

from jingwei_benchmark.methods import availability, run_method, find_method_source


@pytest.fixture(scope="module")
def r_methods():
    names = ["bpca", "minprob", "missforest", "mice-pmm", "qrilc"]
    try:
        return availability(names=names, mice_backend="r")
    except (OSError, TimeoutError):
        return {}


@pytest.mark.parametrize("method", ["bpca", "minprob", "missforest", "mice-pmm", "qrilc"])
def test_binary_exchange_matches_original_csv_bridge(method, r_methods, tmp_path):
    if not r_methods.get(method, {}).get("available"):
        pytest.skip("R backend unavailable")
    rng = np.random.default_rng(12)
    matrix = rng.uniform(2, 20, (16, 8))
    matrix[rng.random(matrix.shape) < .15] = np.nan
    training_hidden = np.zeros(matrix.shape, dtype=bool)
    results = []
    for format in ("csv", "binary"):
        folder = tmp_path / format
        folder.mkdir()
        values, metadata = run_method(method, matrix, training_hidden,
            {"r_transfer_format": format, "threads": 1, "max_iter": 2, "trees": 10, "m": 2, "rank": 2, "mice_backend": "r"},
            device="cpu", seed=42, folder=folder)
        assert metadata["r_transfer_format"] == format
        np.testing.assert_array_equal(values[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
        results.append(values)
    np.testing.assert_allclose(results[0], results[1], rtol=2e-10, atol=2e-10)


def test_serial_small_missforest_preserves_native_predictions(r_methods, tmp_path):
    if not r_methods.get("missforest", {}).get("available"):
        pytest.skip("R backend unavailable")
    rng = np.random.default_rng(29)
    matrix = rng.normal(10, 2, (32, 12))
    matrix[rng.random(matrix.shape) < .2] = np.nan
    outputs = []
    for engine in ("native", "optimized"):
        folder = tmp_path / engine
        folder.mkdir()
        result, metadata = run_method("missforest", matrix, np.zeros(matrix.shape, dtype=bool),
            {"baseline_engine":engine,"threads":4}, device="cpu",seed=37,folder=folder)
        assert metadata["forest_threads"] == (4 if engine == "native" else 1)
        assert metadata["trees"] == 100 and metadata["forest_max_iterations"] == 10
        outputs.append(result)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_qrilc_matches_official_feature_by_sample_api(r_methods, tmp_path):
    if not r_methods.get("qrilc", {}).get("available"):
        pytest.skip("imputeLCMD unavailable")
    from jingwei_benchmark.methods import find_rscript, r_environment
    import subprocess
    rng=np.random.default_rng(31)
    matrix=rng.normal(10,2,(9,24))
    matrix[rng.random(matrix.shape)<.2]=np.nan
    actual,_=run_method("qrilc",matrix,np.zeros(matrix.shape,dtype=bool),{},device="cpu",seed=37,folder=tmp_path)
    source=tmp_path/'reference.bin'
    script=tmp_path/'reference.R'
    np.asarray(matrix,dtype='<f8').tofile(source)
    script.write_text("""
args <- commandArgs(trailingOnly=TRUE)
values <- readBin(args[1], 'double', n=9*24, size=8, endian='little')
input <- matrix(values,nrow=9,ncol=24,byrow=TRUE)
set.seed(37)
# The official API's rows are features and its columns are samples.
result <- t(imputeLCMD::impute.QRILC(t(input),tune.sigma=1)[[1]])
writeBin(as.double(t(result)),args[2],size=8,endian='little')
""",encoding='utf-8')
    output=tmp_path/'reference-output.bin'
    result=subprocess.run([find_rscript(),'--vanilla',str(script),str(source),str(output)],env=r_environment(),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    expected=np.fromfile(output,dtype='<f8').reshape(matrix.shape)
    np.testing.assert_array_equal(actual,expected)
