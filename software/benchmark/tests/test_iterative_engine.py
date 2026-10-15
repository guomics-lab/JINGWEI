import gc
import weakref

import numpy as np
import pytest

from sklearn.ensemble import RandomForestRegressor
from jingwei_benchmark.iterative_engine import make_imputer


@pytest.mark.parametrize("forest", [False, True])
@pytest.mark.parametrize("seed", [7, 23])
def test_single_matrix_fit_matches_all_native_steps(forest, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(10, 2, (24, 9))
    matrix[rng.random(matrix.shape) < .2] = np.nan
    # Include a complete feature: default native ordering still fits it.
    matrix[:, -1] = np.arange(24)
    options = dict(estimator=RandomForestRegressor(n_estimators=12, random_state=seed, n_jobs=1) if forest else None,
                   random_state=seed, max_iter=3, keep_empty_features=True)
    native, _ = make_imputer(optimized=False, **options)
    fast, applied = make_imputer(optimized=True, **options)
    expected = native.fit_transform(matrix)
    result = fast.fit_transform(matrix)
    assert applied
    np.testing.assert_array_equal(result, expected)
    assert fast.n_iter_ == native.n_iter_
    assert len(fast.imputation_sequence_) == len(native.imputation_sequence_)
    assert fast.n_features_with_missing_ == native.n_features_with_missing_
    np.testing.assert_array_equal(fast.random_state_.get_state()[1], native.random_state_.get_state()[1])
    with pytest.raises(RuntimeError, match="does not retain"):
        fast.transform(matrix)


def test_fitted_forests_are_released_not_accumulated(monkeypatch):
    original = RandomForestRegressor.fit
    references = []
    def fit(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        references.append(weakref.ref(self))
        return result
    monkeypatch.setattr(RandomForestRegressor, "fit", fit)
    matrix = np.random.default_rng(8).normal(size=(20, 7))
    matrix[::3, ::2] = np.nan
    imputer, _ = make_imputer(optimized=True, estimator=RandomForestRegressor(n_estimators=5, random_state=3),
                              random_state=3, max_iter=2, keep_empty_features=True)
    result = imputer.fit_transform(matrix)
    gc.collect()
    assert np.isfinite(result).all()
    assert len(references) == len(imputer.imputation_sequence_)
    assert len(references) >= 7
    assert all(ref() is None for ref in references)


def test_unverified_sklearn_version_uses_native(monkeypatch):
    import jingwei_benchmark.iterative_engine as engine
    monkeypatch.setattr(engine, "version", lambda _: "99.0")
    imputer, applied = make_imputer(optimized=True)
    assert not applied
    assert type(imputer).__name__ == "IterativeImputer"
