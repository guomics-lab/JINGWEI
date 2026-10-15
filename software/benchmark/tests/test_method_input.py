import numpy as np
import pytest

from jingwei_benchmark.methods import MethodInputError, run_method, availability


@pytest.mark.parametrize("method", ["seqknn", "impseq"])
def test_sequential_methods_explain_missing_bootstrap_profiles_without_changing_mask(method, tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(20, 12))
    matrix[np.arange(12), np.arange(12)] = np.nan
    before = matrix.copy()
    import jingwei_benchmark.methods as methods
    monkeypatch.setattr(methods.subprocess, "run", lambda *a, **k: pytest.fail("unsupported matrix reached R"))
    with pytest.raises(MethodInputError, match="fully observed.*found 0") as error:
        run_method(method, matrix, np.zeros(matrix.shape, dtype=bool), {},
                   device="cpu",seed=4,folder=tmp_path)
    assert error.value.failure_kind == "unsupported_input"
    np.testing.assert_array_equal(matrix,before)


def test_seqknn_complete_input_is_identity_without_native_empty_loop(tmp_path, monkeypatch):
    import jingwei_benchmark.methods as methods
    monkeypatch.setattr(methods.subprocess, "run", lambda *a, **k: pytest.fail("complete matrix reached R"))
    matrix=np.arange(24,dtype=float).reshape(6,4)
    result,_=run_method("seqknn",matrix,np.zeros(matrix.shape,dtype=bool),{},device="cpu",seed=4,folder=tmp_path)
    np.testing.assert_array_equal(result,matrix)


def test_seqknn_still_uses_original_backend_with_sufficient_profiles(tmp_path):
    if not availability(names=["seqknn"])["seqknn"]["available"]:
        pytest.skip("SeqKnn unavailable")
    matrix=np.random.default_rng(8).normal(10,2,(20,8))
    matrix[::3,2:]=np.nan  # two complete profiles; native code reduces k internally
    result,metadata=run_method("seqknn",matrix,np.zeros(matrix.shape,dtype=bool),{"k":10},device="cpu",seed=4,folder=tmp_path)
    assert np.isfinite(result).all() and metadata["r_transfer_format"]=="binary"
    observed=np.isfinite(matrix)
    np.testing.assert_array_equal(result[observed],matrix[observed])


def test_mle_rejects_missing_pattern_integer_overflow_before_native_fit(tmp_path,monkeypatch):
    import jingwei_benchmark.methods as methods
    monkeypatch.setattr(methods.subprocess,"run",lambda *a,**k:pytest.fail("overflow reached native norm"))
    matrix=np.random.default_rng(3).normal(size=(40,12))
    matrix[35,2]=np.nan
    with pytest.raises(MethodInputError,match="32-bit"):
        run_method("mle",matrix,np.zeros(matrix.shape,dtype=bool),{},device="cpu",seed=4,folder=tmp_path)


def test_mle_large_matrix_with_representable_patterns_is_not_blanket_rejected(tmp_path,monkeypatch):
    import jingwei_benchmark.methods as methods
    def reached_backend(*args,**kwargs):
        raise RuntimeError("supported pattern reached native backend")
    monkeypatch.setattr(methods.subprocess,"run",reached_backend)
    matrix=np.random.default_rng(3).normal(size=(40,12))
    matrix[2,3]=np.nan  # only the low-order pattern bits are used
    with pytest.raises(RuntimeError,match="supported pattern reached native"):
        run_method("mle",matrix,np.zeros(matrix.shape,dtype=bool),{},device="cpu",seed=4,folder=tmp_path)
