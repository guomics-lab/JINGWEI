"""Check notebook compatibility against its archived code, data and outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jingwei_benchmark.cli import main
from jingwei_benchmark.notebook_protocol import load_notebook_matrix, notebook_masks, notebook_score_mask


def test_preprocessing_and_mask_match_original_loader_without_changing_global_rng(tmp_path):
    import torch
    from jingwei_benchmark._legacy.datasets import CSVDataset

    source = tmp_path / 'input.csv'
    frame = pd.DataFrame({'REAL_PROTEIN_DROPPED': [64, 128, 256, 512],
                          'P1': [64, 128, np.nan, 256], 'P2': [0, 32, 128, 512]})
    frame.to_csv(source, index=False)
    expected_file = tmp_path / 'log.csv'
    expected = frame.copy().astype(float)
    values = expected.iloc[:, 1:].to_numpy()
    values[values > 0] = np.log2(values[values > 0])
    expected.iloc[:, 1:] = values
    expected.to_csv(expected_file, index=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        reference = CSVDataset(expected_file)
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    actual = load_notebook_matrix(source, seed=7)
    assert torch.equal(before, torch.get_rng_state())
    np.testing.assert_array_equal(actual.frame.to_numpy(), reference.data_filled.numpy())
    masks, meta = notebook_masks(actual)
    np.testing.assert_array_equal(masks.evaluation_hidden,
        (reference.mask.numpy() == 0) & (reference.data_filled.numpy() != 0))
    assert not masks.training_hidden.any()
    assert meta['training']['hidden_count'] == 0
    np.testing.assert_array_equal(actual.notebook_reference['mean'], reference.mean.numpy())
    np.testing.assert_array_equal(actual.notebook_reference['std'], reference.std.numpy())


def test_alzheimer_archived_shape_observations_and_mask_counts():
    source = Path(__file__).resolve().parents[3] / 'data/Alzheimer.csv'
    data = load_notebook_matrix(source, seed=0)
    masks, _ = notebook_masks(data)
    # Saved output in benchmark.ipynb at source revision 98686b54.
    assert data.frame.shape == (210, 1541)
    assert masks.observed.sum() == 254968
    assert masks.evaluation_hidden.sum() == 229546
    assert masks.available.sum() == 25422
    assert (~masks.available.any(axis=0)).sum() == 20


def test_notebook_scoring_reports_the_historical_exclusions():
    from jingwei_benchmark.masking import BenchmarkMasks
    truth = np.array([[10., 20, 30], [11, 21, 31]])
    hidden = np.array([[False, True, True], [True, False, True]])
    masks = BenchmarkMasks(np.ones_like(hidden), hidden, np.zeros_like(hidden))
    prediction = truth.copy()
    prediction[0, 1] = np.nan
    actual = notebook_score_mask(truth, prediction, masks)
    np.testing.assert_array_equal(actual, [[False, False, False], [True, False, False]])


@pytest.mark.parametrize('suffix', ['.h5ad', '.txt'])
def test_notebook_does_not_silently_apply_legacy_csv_rules_to_other_formats(tmp_path, suffix):
    with pytest.raises(ValueError, match='raw CSV'):
        load_notebook_matrix(tmp_path / ('input' + suffix))


def test_notebook_cli_preserves_masked_worker_boundary_and_one_run(tmp_path):
    source = tmp_path / 'input.csv'
    pd.DataFrame(np.random.default_rng(11).lognormal(8, 1, (40, 8))).to_csv(source)
    output = tmp_path / 'out'
    assert main(['benchmark', '--input', str(source), '--output', str(output),
        '--evaluation-protocol', 'original-notebook', '--seed', '0', '--repeats', '4',
        '--device', 'cpu', '--methods', 'knn', 'feature-minimum']) == 0
    assert not (output / 'repeat_001').exists()
    with np.load(output / 'repeat_000/masks.npz') as masks, np.load(output / 'repeat_000/method_input.npz') as sent:
        assert np.isnan(sent['values'][masks['evaluation_hidden']]).all()
        assert not sent['notebook_normalized'][masks['evaluation_hidden']].any()
        assert not masks['training_hidden'].any()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['config']['repeats'] == 1
    rows = pd.read_csv(output / 'metrics.csv')
    assert (rows.n_evaluated + rows.n_excluded == rows.n_held_out).all()


@pytest.mark.parametrize('evaluation,training', [('mcar', 'mnar-logistic'), ('mnar-logistic', 'mcar')])
def test_uncalibrated_custom_mnar_never_resets_other_stages_mcar_rate(tmp_path, evaluation, training):
    source = tmp_path / 'input.csv'
    pd.DataFrame(np.random.default_rng(3).uniform(10, 100, (40, 12)),
                 index=[f'S{i}' for i in range(40)], columns=[f'P{i}' for i in range(12)]).to_csv(source)
    output = tmp_path / 'out'
    assert main(['benchmark', '--input', str(source), '--output', str(output), '--methods', 'mean',
        '--repeats', '1', '--mechanism', evaluation, '--training-mechanism', training,
        '--mask-rate', '.4', '--training-mask-rate', '.3', '--uncalibrated-mnar',
        '--allow-empty-marginals', '--device', 'cpu']) == 0
    config = json.loads((output / 'repeat_000/mask_config.json').read_text())
    stage = 'evaluation' if evaluation == 'mcar' else 'training'
    assert config[stage]['requested_rate'] == (.4 if stage == 'evaluation' else .3)
    assert abs(config[stage]['actual_rate'] - config[stage]['requested_rate']) <= 1 / config[stage]['eligible_count']


def test_notebook_random_stream_survives_isolated_worker_boundary(tmp_path):
    import random
    import torch
    from jingwei_benchmark.notebook_rng import notebook_random_stream

    initial = torch.Generator().manual_seed(7).get_state().numpy()
    legacy = {'rng_state': initial}
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    with notebook_random_stream(legacy, 7, None, first, neural=True):
        random.random(), np.random.rand(), torch.rand(4)
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = torch.get_rng_state().clone()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    with notebook_random_stream(legacy, 7, first, second, neural=True):
        assert random.getstate() == expected_python
        np.testing.assert_array_equal(np.random.get_state()[1], expected_numpy[1])
        assert np.random.get_state()[2:] == expected_numpy[2:]
        assert torch.equal(torch.get_rng_state(), expected_torch)


def test_original_nmf_layout_compatibility_keeps_real_predictions_finite():
    from jingwei_benchmark.nmf_engine import run_nmf
    values = np.arange(1, 61, dtype='float32').reshape(10, 6)
    mask = np.ones_like(values)
    mask[::2, ::2] = 0
    prediction = run_nmf(values, mask)
    assert prediction.shape == values.shape and np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction[mask == 1], values[mask == 1])


@pytest.mark.parametrize('name', ['dmf', 'dcae'])
def test_hiding_evaluation_targets_from_workers_preserves_original_loss_and_predictions(name):
    import torch
    from jingwei_benchmark._legacy.dmf import DMFImputer
    from jingwei_benchmark._legacy.dcae import DCAEImputer

    torch.manual_seed(4)
    target = torch.randn(5, 80)
    mask = (torch.rand(5, 80) > .7).float()
    sanitized = target * mask
    cls = DMFImputer if name == 'dmf' else DCAEImputer
    extra = {'embedding_dim': 4, 'hidden_dims': [8], 'mask_weight': .5} if name == 'dmf' else {
        'ae_dim': 4, 'mask_predictor_hidden_dim': 8, 'lambda_mask': .5,
        'num_encoder_blocks': 2, 'num_decoder_blocks': 2, 'dilation': 2}
    reference = cls(target, mask, **extra)
    wrapped = cls(sanitized, mask, **extra)
    wrapped.load_state_dict(reference.state_dict())
    torch.manual_seed(99)
    reference_loss = reference.training_step((target * mask, target, mask), 0)
    reference_loss.backward()
    torch.manual_seed(99)
    wrapped_loss = wrapped.training_step((sanitized * mask, sanitized, mask), 0)
    wrapped_loss.backward()
    torch.testing.assert_close(reference_loss, wrapped_loss, rtol=0, atol=0)
    for left, right in zip(reference.parameters(), wrapped.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    torch.testing.assert_close(reference.get_imputed_data(), wrapped.get_imputed_data(), rtol=0, atol=0)
