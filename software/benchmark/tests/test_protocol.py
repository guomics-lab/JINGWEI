import numpy as np
import pytest
from jingwei_benchmark.protocol import (
    missing_probabilities,
    calibrate_probabilities,
    protocol_masks,
    MECHANISMS,
)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_sequential_protocol_is_reproducible_and_never_reveals_test(mechanism):
    values = np.arange(1.0, 401.0).reshape(20, 20)
    observed = np.ones(values.shape, dtype=bool)
    observed[0, :3] = False
    first, metadata = protocol_masks(values, observed, mechanism=mechanism, seed=17)
    repeated, _ = protocol_masks(values, observed, mechanism=mechanism, seed=17)
    changed_training, _ = protocol_masks(
        values, observed, mechanism=mechanism, training_rate=0.1, seed=17
    )
    assert np.array_equal(first.evaluation_hidden, repeated.evaluation_hidden)
    assert np.array_equal(first.evaluation_hidden, changed_training.evaluation_hidden)
    assert not (first.evaluation_hidden & first.training_hidden).any()
    assert not (first.training_hidden & ~first.available).any()
    assert first.available.any(axis=0).all()
    assert first.available.any(axis=1).all()
    assert metadata["evaluation"]["hidden_count"] == int(first.evaluation_hidden.sum())


@pytest.mark.parametrize("mechanism", MECHANISMS[1:])
def test_mnar_probability_decreases_with_abundance(mechanism):
    values = np.arange(1.0, 101.0).reshape(10, 10)
    p = missing_probabilities(values, np.ones_like(values, dtype=bool), mechanism)
    assert (np.diff(p.ravel()) <= 1e-12).all()


def test_calibration_preserves_zero_support_and_requested_mean():
    p = np.array([[0.1, 0.3, 0.6, 0.0]])
    observed = np.ones_like(p, dtype=bool)
    result = calibrate_probabilities(p, observed, 0.4)
    assert result.mean() == pytest.approx(0.4)
    assert result[0, -1] == 0
    with pytest.raises(ValueError, match="support"):
        calibrate_probabilities(p, observed, 0.9)


def test_training_probabilities_ignore_original_and_evaluation_missing_values():
    values = np.arange(1.0, 101.0).reshape(10, 10)
    available = np.ones_like(values, dtype=bool)
    available[0] = False
    changed = values.copy()
    changed[~available] = 1e30
    assert np.array_equal(
        missing_probabilities(values, available, "mnar-logistic"),
        missing_probabilities(changed, available, "mnar-logistic"),
    )


def test_paper_parameters_are_not_rate_calibrated():
    values = np.arange(1.0, 10001.0).reshape(100, 100)
    masks, meta = protocol_masks(
        values,
        np.ones_like(values, dtype=bool),
        mechanism="mnar-jingwei",
        evaluation_rate=None,
        training_rate=None,
        preserve_marginals=False,
        seed=7,
    )
    assert meta["evaluation"]["requested_rate"] is None
    assert masks.evaluation_hidden.sum() > 0
    assert not masks.evaluation_hidden[values >= np.quantile(values, 0.6)].any()
