"""Independent feature-space posterior oracle and MICE/PMM invariants."""
import numpy as np
import pytest

from jingwei_benchmark.mice_engine import conditional_predictions, run_mice_pmm


@pytest.mark.parametrize("shape", [(16, 6), (8, 14), (12, 9)])
def test_conditional_draw_matches_independent_dense_posterior(shape):
    rng = np.random.default_rng(25)
    z = rng.normal(size=shape)
    z[:, -1] = z[:, 0]  # collinearity must not require dropping predictors
    z[:, -2] = 1.0
    target = 2
    rows = np.arange(shape[0] - 3)
    u, e = rng.normal(size=shape[1]), rng.normal(size=len(rows))
    a, chi, ridge = .7, 3.4, .001
    mean, drawn, variance = conditional_predictions(z, z @ z.T, target, rows, u, e, a, chi, ridge)
    keep = np.arange(shape[1]) != target
    x = z[:, keep] - z[rows][:, keep].mean(axis=0)
    y = z[rows, target] - z[rows, target].mean()
    penalty = ridge * len(rows)
    precision = x[rows].T @ x[rows] + penalty * np.eye(keep.sum())
    rhs = x[rows].T @ y
    beta = np.linalg.solve(precision, rhs)
    expected_variance = (y @ y - rhs @ beta) / chi
    noise_beta = np.linalg.solve(precision, np.sqrt(penalty) * u[keep] - x[rows].T @ e)
    expected_mean = z[rows, target].mean() + x @ beta
    expected_draw = expected_mean + np.sqrt(expected_variance) * (x @ noise_beta + a / np.sqrt(len(rows)))
    np.testing.assert_allclose(mean, expected_mean, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(variance, expected_variance, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(drawn, expected_draw, rtol=2e-7, atol=2e-8)


def fixture(shape):
    rng = np.random.default_rng(9)
    x = rng.normal(size=shape)
    x[rng.random(shape) < .22] = np.nan
    x[:3] = rng.normal(size=(3, shape[1]))
    x[:, -1] = 7  # a constant donor feature
    x[4:, -1] = np.nan
    return x


def test_pmm_draws_parameter_uncertainty_without_response_residual_noise():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(12, 5))
    rows, target, ridge = np.arange(9), 1, .1
    keep = np.arange(5) != target
    x = z[:, keep] - z[rows][:, keep].mean(0)
    precision = x[rows].T @ x[rows] + ridge * len(rows) * np.eye(4)
    _, _, variance = conditional_predictions(z, z @ z.T, target, rows, np.zeros(5),
                                              np.zeros(9), 0., 8., ridge)
    expected = variance * (x @ np.linalg.solve(precision, x.T) + np.ones((12, 12)) / len(rows))
    samples = []
    for _ in range(4000):
        mean, drawn, _ = conditional_predictions(z, z @ z.T, target, rows,
            rng.normal(size=5), rng.normal(size=9), rng.normal(), 8., ridge)
        samples.append(drawn - mean)
    actual = np.cov(np.asarray(samples), rowvar=False)
    # Gaussian sample-covariance uncertainty, including near-zero off-diagonals.
    # A relative tolerance alone would incorrectly reject ordinary MC noise.
    standard_error = np.sqrt((expected ** 2 + np.outer(np.diag(expected), np.diag(expected))) / (len(samples) - 1))
    assert np.all(np.abs(actual - expected) < 5 * standard_error + 1e-12)


@pytest.mark.parametrize("shape", [(18, 8), (8, 14)])
def test_cache_replay_draws_pooling_and_seed_reproducibility(shape, tmp_path):
    x = fixture(shape)
    before = x.copy()
    a, meta = run_mice_pmm(x, {}, seed=45, folder=tmp_path / "cached")
    b, _ = run_mice_pmm(x, {"mice_solver": "reference"}, seed=45, folder=tmp_path / "reference")
    repeated, _ = run_mice_pmm(x, {}, seed=45)
    other, _ = run_mice_pmm(x, {}, seed=46)
    draws = np.load(tmp_path / "cached/mice_draws.npz")["values"]
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, repeated)
    assert not np.array_equal(a, other)
    assert draws.shape == (5, *shape) and np.isfinite(draws).all()
    visible = np.isfinite(x)
    for draw in draws:
        np.testing.assert_array_equal(draw[visible], x[visible])
        for j in range(shape[1]):
            assert np.isin(draw[~visible[:, j], j], x[visible[:, j], j]).all()
    np.testing.assert_array_equal(a[~visible], draws.mean(axis=0)[~visible])
    np.testing.assert_array_equal(a[visible], x[visible])
    assert meta["mice_imputations"] == meta["mice_iterations"] == 5
    assert meta["conditional_updates"] == 25 * (~visible.all(axis=0)).sum()
    assert len(meta["iteration_trace"]) == 25 and len(set(meta["chain_seeds"])) == 5
    assert meta["all_predictors"]


def test_single_original_donor_has_only_one_possible_pmm_value():
    x = fixture((10, 5))
    x[:, 1] = np.nan
    x[2, 1] = 2.345
    result, meta = run_mice_pmm(x, {}, seed=42)
    np.testing.assert_array_equal(result[:, 1], np.full(10, 2.345))
    assert 1 in meta["single_observation_features"]


@pytest.mark.parametrize("options", [{"m": 0}, {"max_iter": 0}, {"donors": 0}, {"ridge": 0}, {"ridge": np.nan}])
def test_invalid_budgets_are_rejected(options):
    with pytest.raises(ValueError):
        run_mice_pmm(fixture((10, 5)), options, seed=42)


def test_no_observed_donors_is_reported_without_fabricated_values():
    x = fixture((10, 5))
    x[:, 1] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        run_mice_pmm(x, {}, seed=42)


def test_python_backend_needs_no_r_and_keeps_native_option(monkeypatch, tmp_path):
    import jingwei_benchmark.methods as methods
    monkeypatch.setattr(methods, "find_rscript", lambda explicit=None: None)
    assert methods.availability(names=["mice-pmm"])["mice-pmm"]["available"]
    assert not methods.availability(names=["mice-pmm"], mice_backend="r")["mice-pmm"]["available"]
    x = fixture((10, 5))
    result, meta = methods.run_method("mice-pmm", x, np.zeros(x.shape, dtype=bool), {},
                                      device="cpu", seed=42, folder=tmp_path)
    assert meta["mice_backend"] == "python" and meta["implementation"] == "python-linear-mice-pmm"
    assert np.isfinite(result).all()


def test_legacy_job_without_backend_keeps_its_original_r_method(monkeypatch, tmp_path):
    import json
    import sys
    import jingwei_benchmark.methods as methods
    import jingwei_benchmark.worker as worker
    source = tmp_path / "input.npz"
    np.savez(source, values=np.ones((5, 3)), training_hidden=np.zeros((5, 3), dtype=bool))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"method": "mice-pmm", "input_file": str(source),
        "folder": str(tmp_path), "parameters": {}, "device": "cpu", "seed": 42}))
    captured = []
    def run(name, matrix, hidden, params, **options):
        captured.append(params["mice_backend"])
        return matrix.copy(), {"device": "cpu"}
    monkeypatch.setattr(methods, "run_method", run)
    monkeypatch.setattr(sys, "argv", ["worker", str(job)])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "R_RANGER_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    assert worker.main() == 0
    assert captured == ["r"]
