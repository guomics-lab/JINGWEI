"""Stopping must be driven by model-owned validation, never benchmark truth."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from jingwei_benchmark.convergence import ValidationMonitor, validation_split
from jingwei_benchmark.methods import run_method


def test_plateau_restores_best_weights_and_epoch_instead_of_last_epoch():
    model = torch.nn.Linear(1, 1, bias=False)
    monitor = ValidationMonitor({'training_policy':'convergence','patience':2,'min_epochs':1},
        enabled=True, validation_mask=np.ones((1,1), dtype=bool))
    for epoch, score in enumerate([3., 2., 2.1, 2.2], start=1):
        model.weight.data.fill_(epoch)
        stop = monitor.observe(epoch, score, model)
    assert stop
    monitor.restore(model)
    assert model.weight.item() == 2
    assert monitor.metadata()['best_epoch'] == 2
    assert monitor.metadata()['converged'] and monitor.metadata()['best_checkpoint_restored']


def test_epoch_cap_does_not_masquerade_as_convergence():
    model = torch.nn.Linear(1, 1)
    monitor = ValidationMonitor({'training_policy':'convergence','patience':30},
        enabled=True, validation_mask=np.ones((1,1), dtype=bool))
    assert not monitor.observe(1, 4., model)
    monitor.restore(model)
    assert monitor.metadata()['convergence_status'] == 'max_epochs'
    assert not monitor.metadata()['converged']


def test_insufficient_validation_is_explicit_instead_of_claiming_convergence():
    from jingwei_benchmark.methods import MethodInputError
    with pytest.raises(MethodInputError, match='convergence cannot be assessed'):
        validation_split(np.ones((1,1)),np.zeros((1,1),bool),{'training_policy':'convergence'},7)


def test_validation_is_a_subset_of_available_input_and_excludes_training_corruption():
    matrix = np.random.default_rng(3).normal(size=(40, 11))
    matrix[0, :4] = np.nan  # Neither final test nor native missing truth is present.
    training = np.zeros(matrix.shape, bool)
    training[1, :4] = True
    hidden = validation_split(matrix, training, {'training_policy':'convergence'}, 7)
    assert hidden.any()
    assert not (hidden & (~np.isfinite(matrix) | training)).any()
    assert (np.isfinite(matrix) & ~training & ~hidden).any(axis=0).all()
    repeated = validation_split(matrix, training, {'training_policy':'convergence'}, 7)
    np.testing.assert_array_equal(hidden, repeated)


@pytest.mark.parametrize('method', ['jingwei-dmf','jingwei-dcae','pimms-dae','pimms-vae','pimms-cf'])
def test_all_five_neural_models_stop_on_validation_and_restore_checkpoint(method, tmp_path):
    torch.set_num_threads(1)
    values = np.random.default_rng(4).normal(4, .3, (16, 17))
    values[0, 0] = np.nan
    params = {'training_policy':'convergence','max_epochs':5, 'patience':1,
              'min_epochs':2, 'min_delta':.99, 'validation_fraction':.15,
              'batch_size':4, 'latent_dim':4,'hidden_dims':[8],
              'encoder_blocks':1,'decoder_blocks':1, 'threads':1}
    prediction, meta = run_method(method, values, np.zeros_like(values, bool), params,
                                  device='cpu', seed=7, folder=tmp_path)
    assert meta['converged'] and meta['convergence_status'] == 'validation_plateau'
    assert meta['epochs_trained'] < params['max_epochs']
    assert 1 <= meta['best_epoch'] <= meta['epochs_trained']
    assert meta['best_checkpoint_restored'] and (tmp_path / 'best_model.pt').is_file()
    hidden = np.load(tmp_path / 'model_validation.npz')['hidden']
    assert not hidden[0,0] and hidden.sum() == meta['validation_observations']
    np.testing.assert_array_equal(prediction[np.isfinite(values)], values[np.isfinite(values)])
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize('method', ['jingwei-dmf','jingwei-dcae'])
def test_lightning_uses_the_same_validation_policy_and_best_checkpoint(tmp_path, method):
    values = np.random.default_rng(9).normal(4,.3,(12,11))
    params = {'training_engine':'lightning','training_policy':'convergence',
        'max_epochs':5,'patience':1,'min_epochs':2,'min_delta':.99,
        'batch_size':4,'latent_dim':3,'hidden_dims':[7]}
    prediction, meta = run_method(method, values, np.zeros_like(values,bool), params,
                                  device='cpu',seed=7,folder=tmp_path)
    assert meta['converged'] and meta['best_checkpoint_restored']
    assert meta['epochs_trained'] < 5
    np.testing.assert_allclose(prediction, values)


def test_changing_validation_truth_cannot_change_first_epoch_fitted_weights(tmp_path):
    values = np.random.default_rng(3).normal(4,.3,(12,11))
    params = {'training_policy':'convergence','max_epochs':1,'patience':3,
              'batch_size':4,'latent_dim':3,'hidden_dims':[7]}
    hidden = validation_split(values,np.zeros_like(values,bool),params,7)
    changed = values.copy(); changed[hidden] += 1000
    states = []
    for index, matrix in enumerate([values, changed]):
        folder = tmp_path / str(index); folder.mkdir()
        _, meta = run_method('jingwei-dmf',matrix,np.zeros_like(values,bool),params,
                             device='cpu',seed=7,folder=folder)
        states.append(torch.load(folder / 'best_model.pt',weights_only=True)['state_dict'])
        assert not meta['converged']
    for key in states[0]:
        torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
