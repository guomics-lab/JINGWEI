import numpy as np
import pytest

from jingwei.masking import BenchmarkMasks, make_benchmark_masks


def test_sequential_masking_and_rates():
    observed = np.ones((10, 10), dtype=bool)
    observed[0] = False
    masks = make_benchmark_masks(
        observed, evaluation_rate=1 / 3, training_rate=0.5, seed=7
    )
    assert masks.evaluation_hidden.sum() == 30
    assert masks.training_hidden.sum() == 30
    assert not (masks.evaluation_hidden & masks.training_hidden).any()
    assert not (masks.evaluation_hidden & ~observed).any()
    assert not (masks.training_hidden & ~masks.available).any()
    assert np.array_equal(masks.available, observed & ~masks.evaluation_hidden)
    assert masks.training_visible.sum() == 30


def test_training_settings_do_not_change_shared_evaluation_mask():
    observed = np.ones((20, 10), dtype=bool)
    first = make_benchmark_masks(observed, training_rate=0.1, seed=42)
    second = make_benchmark_masks(observed, training_rate=0.8, seed=42)
    replay = make_benchmark_masks(observed, training_rate=0.1, seed=42)
    assert np.array_equal(first.evaluation_hidden, second.evaluation_hidden)
    assert np.array_equal(first.training_hidden, replay.training_hidden)
    assert not np.array_equal(
        first.evaluation_hidden,
        make_benchmark_masks(observed, seed=43).evaluation_hidden,
    )


def test_methods_receive_same_evaluation_corrupted_input():
    values = np.arange(12.0, dtype=float).reshape(3, 4)
    observed = np.ones_like(values, dtype=bool)
    observed[0, 0] = False
    values[0, 0] = np.nan
    masks = make_benchmark_masks(observed, seed=10)
    method_input = masks.method_input(values)
    training_input = masks.training_input(values)
    assert np.isnan(method_input[~masks.available]).all()
    assert np.isnan(training_input[~masks.training_visible]).all()
    assert np.array_equal(method_input[masks.available], values[masks.available])


def test_overlap_and_native_missing_masking_rejected():
    observed = np.array([[True, False], [True, True]])
    native_missing = ~observed
    empty = np.zeros_like(observed)
    evaluation = np.array([[True, False], [False, False]])
    with pytest.raises(ValueError, match="originally observed"):
        BenchmarkMasks(observed, native_missing, empty)
    with pytest.raises(ValueError, match="remaining after"):
        BenchmarkMasks(observed, evaluation, evaluation)
    with pytest.raises(ValueError, match="remaining after"):
        BenchmarkMasks(observed, evaluation, native_missing)


def test_mask_snapshots_do_not_mutate_callers_or_get_mutated():
    observed = np.ones((3, 3), dtype=bool)
    masks = make_benchmark_masks(observed, seed=4)
    observed[:] = False
    assert masks.observed.all()
    with pytest.raises(ValueError):
        masks.evaluation_hidden[:] = False


@pytest.mark.parametrize("rate", [-0.1, 1.1, np.nan])
def test_invalid_masking_rate(rate):
    with pytest.raises(ValueError):
        make_benchmark_masks(np.ones((3, 3), dtype=bool), evaluation_rate=rate)


def test_entire_evaluation_holdout_leaves_no_training_corruption():
    masks = make_benchmark_masks(np.ones((2, 3), dtype=bool), evaluation_rate=1)
    assert masks.evaluation_hidden.all()
    assert not masks.training_hidden.any()
    assert not masks.training_visible.any()
