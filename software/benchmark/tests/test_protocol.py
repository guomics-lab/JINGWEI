import numpy as np
import pytest
from jingwei_benchmark.protocol import (
    missing_probabilities,
    calibrate_probabilities,
    protocol_masks,
    MECHANISMS,
    ORIGINAL_MECHANISM,
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


@pytest.mark.parametrize("mechanism", [m for m in MECHANISMS[1:] if m != ORIGINAL_MECHANISM])
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


def test_original_rule_preserves_strict_boundaries_and_three_probabilities():
    # These 77 values have exactly mean=0 and population std=1. All strict
    # boundaries must remain visible, unlike a corrected >= z-score rule.
    values = np.array([-4, -3, -2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5, 3, 4] + [0] * 65)[:, None]
    p = missing_probabilities(values, np.ones_like(values, dtype=bool), ORIGINAL_MECHANISM)
    np.testing.assert_allclose(p[:6, 0], [.9, 0, .5, 0, .3, 0], atol=3e-8, rtol=0)
    # Historical mixed-scale comparison changes under a shift; this test
    # prevents silently substituting fixed z-score thresholds.
    shifted = missing_probabilities(values + 100, np.ones_like(values, dtype=bool), ORIGINAL_MECHANISM)
    np.testing.assert_allclose(shifted, .9, atol=3e-8, rtol=0)


def test_original_training_stats_exclude_evaluation_truth_and_native_missing():
    values = np.arange(1., 121.).reshape(20, 6)
    eligible = np.ones(values.shape, dtype=bool)
    eligible[::3] = False
    changed = values.copy()
    changed[~eligible] = np.nan
    p = missing_probabilities(values, eligible, ORIGINAL_MECHANISM)
    np.testing.assert_array_equal(p, missing_probabilities(changed, eligible, ORIGINAL_MECHANISM))
    assert not p[~eligible].any()


def test_original_draw_matches_reference_get_mask_for_each_stage(tmp_path):
    import pandas as pd
    import torch
    from jingwei_benchmark._legacy.datasets import CSVDataset

    working = np.random.default_rng(8).normal(20, 3, (50, 8)).astype('float32')
    observed = np.ones(working.shape, dtype=bool)
    observed[0, 1] = False
    masks, meta = protocol_masks(2. ** working, observed, working_values=working,
        mechanism=ORIGINAL_MECHANISM, seed=4, evaluation_rate=.4, training_rate=.1)
    for stage, eligible, actual in [('evaluation', observed, masks.evaluation_hidden),
                                    ('training', masks.available, masks.training_hidden)]:
        path = tmp_path / (stage + '.csv')
        pd.DataFrame(np.where(eligible, working, np.nan)).to_csv(path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(meta[stage]['random_seed'])
            reference = CSVDataset(path)
        np.testing.assert_array_equal(actual, eligible & ~reference.mask.numpy().astype(bool))
        assert meta[stage]['requested_rate'] is None
        assert meta[stage]['marginal_repairs'] == 0
        assert meta[stage]['rng'] == 'torch-cpu-float32-column-major'
