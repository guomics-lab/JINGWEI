"""A fixed budget keeps one split and must never stop on validation patience."""
import numpy as np
import pytest
import torch

from jingwei_benchmark.convergence import ValidationMonitor, validation_split
from jingwei_benchmark.methods import run_method


def test_fixed_budget_records_worsening_validation_without_early_stopping():
    model=torch.nn.Linear(1,1,bias=False)
    monitor=ValidationMonitor({'training_policy':'fixed-validation','max_epochs':3,
        'patience':1,'min_epochs':1},enabled=True,validation_mask=np.ones((1,1),bool))
    for epoch,loss in enumerate([1.,2.,3.],1):
        model.weight.data.fill_(epoch)
        assert not monitor.observe(epoch,loss,model)
    monitor.restore(model)
    assert model.weight.item()==1
    detail=monitor.metadata()
    assert detail['convergence_status']=='fixed_budget' and detail['converged'] is None
    assert not detail['early_stopping_enabled'] and detail['best_checkpoint_restored']
    assert detail['requested_epochs']==3 and len(detail['validation_history'])==3


@pytest.mark.parametrize('method',['jingwei-dmf','jingwei-dcae','pimms-dae','pimms-vae','pimms-cf'])
def test_all_five_neural_models_run_full_budget_with_one_fixed_validation_split(method,tmp_path):
    torch.set_num_threads(1)
    matrix=np.random.default_rng(4).normal(4,.3,(16,17))
    matrix[0,0]=np.nan
    params={'training_policy':'fixed-validation','max_epochs':5,'patience':1,
        'min_epochs':1,'min_delta':.99,'validation_fraction':.15,'batch_size':4,
        'latent_dim':4,'hidden_dims':[8],'encoder_blocks':1,'decoder_blocks':1,'threads':1}
    hidden=np.zeros_like(matrix,bool)
    prediction,detail=run_method(method,matrix,hidden,params,device='cpu',seed=7,folder=tmp_path)
    assert detail['epochs_trained']==5
    assert detail['convergence_status']=='fixed_budget' and detail['converged'] is None
    assert len(detail['validation_history'])==5 and detail['best_checkpoint_restored']
    assert not detail['early_stopping_enabled']
    split=np.load(tmp_path/'model_validation.npz')
    np.testing.assert_array_equal(split['hidden'],validation_split(matrix,hidden,params,7))
    assert not split['hidden'][0,0] and not (split['hidden'] & split['training']).any()
    np.testing.assert_array_equal(split['hidden'] | split['training'],np.isfinite(matrix))
    final=torch.load(tmp_path/'last_model.pt',weights_only=True)
    best=torch.load(tmp_path/'best_model.pt',weights_only=True)
    assert final['epoch']==5 and best['best_epoch']==detail['best_epoch']
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction[np.isfinite(matrix)],matrix[np.isfinite(matrix)])
