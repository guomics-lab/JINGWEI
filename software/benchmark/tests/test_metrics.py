import itertools

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from jingwei_benchmark.metrics import (
    imputation_metrics,
    wasserstein_distance_1,
    wasserstein_distance_2,
    wasserstein_distance_2_1d,
)


def test_multivariate_w2_preserves_sample_structure():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    predicted = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert wasserstein_distance(reference.ravel(), predicted.ravel()) == 0
    assert wasserstein_distance_2(reference, predicted) == pytest.approx(1)


def test_w2_is_invariant_to_sample_permutation():
    samples = np.array([[0.0, 3.0], [2.0, 1.0], [4.0, 9.0]])
    assert wasserstein_distance_2(samples, samples[[2, 0, 1]]) == pytest.approx(0)


def test_multivariate_translation():
    samples = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]])
    assert wasserstein_distance_2(samples, samples + [3.0, 4.0]) == pytest.approx(5)


@pytest.mark.parametrize(
    "reference,predicted,expected",
    [
        ([[0.0], [2.0]], [[0.0], [1.0], [2.0]], np.sqrt(1 / 3)),
        ([[0.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]], np.sqrt(12.5)),
        ([[0.0], [0.0], [2.0]], [[0.0], [2.0]], np.sqrt(2 / 3)),
    ],
)
def test_unequal_sample_counts(reference, predicted, expected):
    assert wasserstein_distance_2(reference, predicted) == pytest.approx(expected)
    assert wasserstein_distance_2(predicted, reference) == pytest.approx(expected)


def test_multivariate_w2_against_exhaustive_transport_assignment():
    rng = np.random.default_rng(17)
    reference, predicted = rng.normal(size=(2, 4, 3))
    costs = [
        np.mean(np.sum((reference - predicted[list(p)]) ** 2, axis=1))
        for p in itertools.permutations(range(4))
    ]
    assert wasserstein_distance_2(reference, predicted) == pytest.approx(
        np.sqrt(min(costs))
    )


def test_unequal_sample_transport_scales_at_small_magnitudes():
    expected = np.sqrt(1 / 3) * 1e-12
    assert wasserstein_distance_2(
        [[0.0], [2e-12]], [[0.0], [1e-12], [2e-12]]
    ) == pytest.approx(expected, rel=1e-8, abs=1e-24)


@pytest.mark.parametrize(
    "reference,predicted",
    [
        ([1, 2], [1, 2]),
        ([[1.0, np.nan]], [[1.0, 2.0]]),
        ([[1.0, 2.0]], [[1.0]]),
        (np.empty((0, 2)), [[1.0, 2.0]]),
        ([[1.0, 2.0]], [[1.0, np.inf]]),
    ],
)
def test_invalid_sample_distribution_rejected(reference, predicted):
    with pytest.raises(ValueError):
        wasserstein_distance_2(reference, predicted)


def test_1d_w2_is_not_sqrt_w1():
    assert wasserstein_distance_1([0.0, 2.0], [0.0, 0.0]) == pytest.approx(1)
    assert wasserstein_distance_2_1d([0.0, 2.0], [0.0, 0.0]) == pytest.approx(
        np.sqrt(2)
    )


def test_1d_unequal_samples_and_scaling():
    assert wasserstein_distance_2_1d([0.0, 2.0], [0.0, 1.0, 2.0]) == pytest.approx(
        np.sqrt(1 / 3)
    )
    assert wasserstein_distance_2_1d([0.0, 6.0], [0.0, 3.0, 6.0]) == pytest.approx(
        np.sqrt(3)
    )


def test_1d_w1_matches_independent_scipy_reference():
    rng = np.random.default_rng(9)
    u, v = rng.normal(size=13), rng.normal(size=21)
    assert wasserstein_distance_1(u, v) == pytest.approx(wasserstein_distance(u, v))


def test_metrics_score_only_fixed_held_out_cells():
    truth = [[1.0, 3.0, np.nan], [4.0, 8.0, 9.0]]
    predictions = [[1e6, 5.0, np.nan], [np.inf, 6.0, -1e6]]
    mask = np.array([[False, True, False], [False, True, False]])
    assert imputation_metrics(truth, predictions, mask) == {
        "n_evaluated": 2,
        "MAE": 2.0,
        "RMSE": 2.0,
    }
    predictions[0][1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        imputation_metrics(truth, predictions, mask)


def test_empty_or_non_boolean_evaluation_mask_rejected():
    with pytest.raises(ValueError):
        imputation_metrics([[1.0]], [[1.0]], [[False]])
    with pytest.raises(ValueError):
        imputation_metrics([[1.0]], [[1.0]], [[1]])
