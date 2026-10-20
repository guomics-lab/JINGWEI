"""Imputation builds JINGWEI training inputs without discarding known targets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from jingwei_benchmark.cli import main, parser


@pytest.mark.parametrize('mechanism', ['mcar', 'mnar-logistic', 'mnar-jingwei-original'])
def test_impute_dispatch_applies_training_corruption_only_to_jingwei(tmp_path, monkeypatch, mechanism):
    import jingwei_benchmark.benchmark as orchestration
    source = tmp_path / 'input.csv'
    values = np.random.default_rng(17).lognormal(3, .4, (16, 12))
    values[0, 0] = np.nan
    pd.DataFrame(values, index=[f'S{i}' for i in range(len(values))],
                 columns=[f'P{i}' for i in range(values.shape[1])]).to_csv(source)
    received = {}
    def worker(payload, *args):
        with np.load(payload['input_file']) as bundle:
            matrix = bundle['values'].copy()
            hidden = bundle['training_hidden'].copy()
        received[payload['method']] = (matrix, hidden)
        assert np.array_equal(np.isfinite(matrix), np.isfinite(values))
        prediction = np.where(np.isfinite(matrix), matrix, np.nanmean(matrix, axis=0)[None,:])
        np.savez(Path(payload['folder']) / 'predictions.npz', values=prediction)
        return {'status':'completed','time_s':0.}
    monkeypatch.setattr(orchestration, '_run_worker', worker)
    output = tmp_path / 'results'
    assert main(['impute', '--input', str(source), '--methods', 'jingwei-dmf', 'jingwei-dcae', 'mean',
                 '--mechanism', mechanism, '--mask-rate', '.3', '--device', 'cpu', '--output', str(output)]) == 0
    assert not received['mean'][1].any()
    a, hidden = received['jingwei-dmf']
    assert hidden.any() and not (hidden & ~np.isfinite(a)).any()
    np.testing.assert_array_equal(received['jingwei-dcae'][1], hidden)
    np.testing.assert_array_equal(received['mean'][0], a)
    assert np.isfinite(a[hidden]).all(), 'Training targets must remain available to the JINGWEI runner'
    with np.load(output / 'repeat_000/masks.npz') as masks:
        assert not masks['evaluation_hidden'].any()
        np.testing.assert_array_equal(masks['training_hidden'], hidden)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['effective_protocol']['platform_mask_count'] == 0
    assert manifest['effective_protocol']['training_methods'] == ['jingwei-dmf', 'jingwei-dcae']
    rows = {row['method']:row for row in manifest['runs']}
    assert rows['mean']['training_rate'] == 0 and rows['jingwei-dmf']['training_rate'] > 0


@pytest.mark.parametrize('engine', ['torch', 'lightning'])
def test_impute_trains_both_native_models_with_targets_and_preserves_observations(tmp_path, engine):
    source = tmp_path / 'input.csv'
    original = np.random.default_rng(4).normal(6, .5, (7, 17))
    original[0,0] = np.nan
    pd.DataFrame(original, index=[f'S{i}' for i in range(len(original))],
                 columns=[f'P{i}' for i in range(original.shape[1])]).to_csv(source)
    output = tmp_path / 'results'
    assert main(['impute','--input',str(source),'--methods','jingwei-dmf','jingwei-dcae',
                 '--scale','log2','--transform','none','--mechanism','mcar','--mask-rate','.3',
                 '--training-engine',engine,'--device','cpu','--threads','1','--max-epochs','2',
                 '--batch-size','3','--latent-dim','4','--hidden-dims','8','--output',str(output)]) == 0
    with np.load(output/'repeat_000/masks.npz') as masks:
        observed = masks['observed']; hidden = masks['training_hidden']
        assert hidden.any() and not masks['evaluation_hidden'].any()
    for method in ('jingwei-dmf','jingwei-dcae'):
        folder = output/'repeat_000'/method
        detail = json.loads((folder/'worker_result.json').read_text(encoding='utf-8'))
        assert detail['training_hidden_entries'] == hidden.sum()
        assert detail['supervision_entries_per_epoch'] == observed.sum()
        assert detail['training_input_entries'] == observed.sum() - hidden.sum()
        predicted = pd.read_csv(folder/'imputed.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(predicted[observed],original[observed])
        assert np.isfinite(predicted).all()


def test_cli_uses_jingwei_name(capsys):
    assert parser().prog == 'jingwei'
    with pytest.raises(SystemExit) as exited:
        parser().parse_args(['--help'])
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    assert 'usage: jingwei ' in help_text and 'jingwei-benchmark' not in help_text


@pytest.mark.parametrize('rate', ['1', '-.1', 'nan'])
def test_impute_rejects_invalid_training_rate_before_running(tmp_path, rate, capsys):
    assert main(['impute','--input',str(tmp_path/'unused.csv'),'--methods','jingwei-dmf',
                 '--mask-rate',rate,'--device','cpu','--output',str(tmp_path/'result')]) == 1
    assert 'training mask rate' in capsys.readouterr().err
