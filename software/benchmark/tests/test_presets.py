import json
import numpy as np
import pandas as pd
from jingwei_benchmark.cli import main
from jingwei_benchmark.presets import preset_parameters


def test_original_parameters_and_method_override_share_identical_holdouts(tmp_path, monkeypatch):
    import jingwei_benchmark.benchmark as benchmark
    source = tmp_path / 'input.csv'
    pd.DataFrame(np.arange(1, 61).reshape(10, 6), columns=[f'P{i}' for i in range(6)]).to_csv(source, index=False)
    captured = []
    def run(payload, *args):
        captured.append(payload['parameters'])
        matrix = np.load(payload['input_file'])['values']
        fill = np.where(np.isfinite(matrix), matrix, np.nanmean(matrix, axis=0))
        np.savez(payload['folder'] + '/predictions.npz', values=fill)
        return {'status': 'completed', 'time_s': .1}
    monkeypatch.setattr(benchmark, '_run_worker', run)
    config = tmp_path / 'method.json'
    config.write_text(json.dumps({'jingwei-dcae': {'max_epochs': 7}}))
    common = ['benchmark', '--input', str(source), '--methods', 'jingwei-dmf', 'jingwei-dcae',
              '--repeats', '1', '--device', 'cpu']
    assert main(common + ['--output', str(tmp_path / 'standard')]) == 0
    assert main(common + ['--output', str(tmp_path / 'preset'), '--jingwei-preset', 'original-benchmark',
                         '--method-config', str(config)]) == 0
    assert captured[2]['learning_rate'] == 5e-4 and captured[2]['batch_size'] == 1024
    assert captured[3]['latent_dim'] == 256 and captured[3]['mask_predictor_hidden_dim'] == 128
    assert captured[3]['encoder_blocks'] == captured[3]['decoder_blocks'] == captured[3]['dilation'] == 3
    assert captured[3]['learning_rate'] == 5e-5 and captured[3]['max_epochs'] == 7
    with np.load(tmp_path / 'standard/repeat_000/masks.npz') as a, np.load(tmp_path / 'preset/repeat_000/masks.npz') as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    scores = pd.read_csv(tmp_path / 'preset/metrics.csv')
    np.testing.assert_allclose(scores.MSE, scores.RMSE**2)
    assert 'MSE_mean' in pd.read_csv(tmp_path / 'preset/summary.csv')
    assert preset_parameters('original-benchmark', 'pimms-vae') == {}
