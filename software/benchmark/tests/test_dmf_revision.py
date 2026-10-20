"""Factorization behavior, row identity and leakage checks for revised DMF."""
import numpy as np
import pytest
import torch
from jingwei_benchmark.dmf_models import BilinearDMF


def test_dense_and_index_predictions_preserve_repeated_shuffled_row_identity():
    torch.manual_seed(7)
    model=BilinearDMF(7,11,embedding_dim=4,hidden_dims=[8],interaction_dim=3).eval()
    rows=torch.tensor([6,0,6,2]);columns=torch.tensor([3,1,9,4])
    dense,mask=model(torch.zeros(4,11),row_indices=rows)
    indexed,other=model((rows,columns))
    torch.testing.assert_close(indexed,dense[torch.arange(4),columns])
    torch.testing.assert_close(dense,model.get_full_matrix()[rows])
    assert mask.shape==other.shape==(4,11)
    with pytest.raises(ValueError,match='row_indices'):
        model(torch.zeros(4,11),row_indices=rows[:2])


def test_mask_head_shares_row_latent_without_encoding_mask_in_intensity_outputs():
    model=BilinearDMF(5,9,embedding_dim=4,hidden_dims=[8],interaction_dim=3)
    reconstruction,logits=model(torch.zeros(5,9))
    reconstruction.retain_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits)).backward()
    assert reconstruction.grad is None
    assert model.row_embedding.weight.grad.abs().sum()>0
    assert model.col_embedding.weight.grad is None


def test_initialization_never_reads_unavailable_training_intensities():
    values=torch.randn(8,11)
    visible=torch.rand(8,11)>.2
    states=[]
    for missing in (float('nan'),1.e20):
        torch.manual_seed(31)
        model=BilinearDMF(8,11,embedding_dim=4,hidden_dims=[8],interaction_dim=3)
        altered=torch.where(visible,values,missing)
        detail=model.initialize_from_observed(altered,visible)
        assert detail['observations']==int(visible.sum())
        states.append(model.state_dict())
    for name in states[0]:torch.testing.assert_close(states[0][name],states[1][name],rtol=0,atol=0)


def test_bilinear_dmf_learns_low_rank_signal_without_access_to_test_values(tmp_path):
    from jingwei_benchmark.methods import run_method
    torch.set_num_threads(1)
    rng=np.random.default_rng(3)
    truth=rng.normal(size=(24,2))@rng.normal(size=(2,18))
    missing=rng.random(truth.shape)<.2
    values=np.where(missing,np.nan,truth)
    prediction,detail=run_method('jingwei-dmf',values,np.zeros_like(missing),
        {'dmf_architecture':'bilinear','dmf_rank':2,'dmf_initialization':'random',
         'training_policy':'fixed','max_epochs':200,'latent_dim':4,'hidden_dims':[8],
         'batch_size':24,'learning_rate':.01,'mask_weight':0.},device='cpu',seed=7,folder=tmp_path)
    mean=np.nanmean(values,axis=0)[None,:].repeat(len(values),axis=0)
    error=np.mean((prediction[missing]-truth[missing])**2)
    baseline=np.mean((mean[missing]-truth[missing])**2)
    assert error<baseline*.5
    assert detail['epochs_trained']==200 and detail['model_architecture_version']=='bilinear-residual-v1'
    np.testing.assert_array_equal(prediction[~missing],truth[~missing])
