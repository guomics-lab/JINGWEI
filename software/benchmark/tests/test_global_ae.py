"""Contracts for the experimental AE, its training boundary and user controls."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from jingwei_benchmark.ae_models import GlobalDCAE, build_dcae


def tiny(features=17, latent=4):
    return GlobalDCAE(features,latent,mask_predictor_hidden_dim=8,
                     num_encoder_blocks=2,num_decoder_blocks=1,dilation=2)


@pytest.mark.parametrize('features,latent',[(1,1),(17,4),(32,4),(1541,8)])
def test_global_ae_keeps_protein_dimension_for_short_odd_and_wide_inputs(features,latent):
    torch.set_num_threads(1)
    model=tiny(features,latent).eval()
    x=torch.randn(2,features)
    observed=torch.rand_like(x)>.4
    reconstruction,logits=model(x,observed)
    assert reconstruction.shape==logits.shape==x.shape
    assert torch.isfinite(reconstruction).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(model.reconstruct(x,observed),reconstruction)


def test_unavailable_values_cannot_change_network_outputs_or_input_gradients():
    torch.manual_seed(2)
    model=tiny().eval()
    mask=torch.rand(3,17)>.4
    first=torch.randn(3,17,requires_grad=True)
    second=first.detach().clone()
    second[~mask]=float('nan')
    actual=model(first,mask)
    expected=model(second,mask)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(part.square().mean() for part in actual).backward()
    assert torch.equal(first.grad[~mask],torch.zeros_like(first.grad[~mask]))


def test_true_zero_and_missing_zero_have_distinct_encoder_inputs():
    torch.manual_seed(4)
    model=tiny().eval()
    x=torch.zeros(2,17)
    observed=torch.ones_like(x,dtype=torch.bool)
    missing=observed.clone()
    missing[:,3]=False
    assert not torch.equal(model.encode(x,observed),model.encode(x,missing))


def test_auxiliary_head_updates_encoder_without_distorting_reconstruction_head():
    model=tiny().train()
    x=torch.randn(3,17)
    _,logits=model(x,torch.ones_like(x,dtype=torch.bool))
    logits.square().mean().backward()
    assert model.reconstruction.weight.grad is None
    assert model.bottleneck[1].weight.grad is not None
    assert model.mask_predictor[0].weight.grad.abs().sum()>0


def test_global_inference_with_same_weights_agrees_on_cpu_and_cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(3)
    cpu=tiny().eval()
    gpu=copy.deepcopy(cpu).cuda().eval()
    x=torch.randn(3,17)
    observed=torch.rand(3,17)>.3
    with torch.no_grad():
        expected=cpu(x,observed)
        actual=gpu(x.cuda(),observed.cuda())
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.cpu(),b,rtol=2e-4,atol=2e-5)


def test_legacy_architecture_factory_and_invalid_selection():
    from jingwei_benchmark._native.models import DCAE
    assert isinstance(build_dcae('native',input_feature_dim=17,ae_dim=4),DCAE)
    with pytest.raises(ValueError,match='unknown AE architecture'):
        build_dcae('unsupported',input_feature_dim=17,ae_dim=4)


def test_native_factory_preserves_initial_weights_and_outputs_exactly():
    from jingwei_benchmark._native.models import DCAE
    torch.manual_seed(31)
    reference=DCAE(17,4).eval()
    torch.manual_seed(31)
    model=build_dcae('native',input_feature_dim=17,ae_dim=4).eval()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
    x=torch.randn(2,17)
    for a,b in zip(reference(x),model(x)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_global_capacity_guard_rejects_excessive_matrix_before_allocating_weights():
    with pytest.raises(ValueError,match='parameters; reduce'):
        GlobalDCAE(1_000_000,256,mask_predictor_hidden_dim=512)


@pytest.mark.parametrize('dropout',[-.1,1.,float('nan')])
def test_global_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError,match='dropout'):
        GlobalDCAE(17,4,dropout=dropout)


@pytest.mark.parametrize('engine',['torch','lightning'])
def test_global_runners_preserve_observations_and_denoising_supervision(engine,tmp_path):
    from jingwei_benchmark.methods import run_method
    values=np.random.default_rng(5).normal(size=(7,17))
    values[0,0]=np.nan
    hidden=np.zeros_like(values,dtype=bool)
    hidden[1,:3]=True
    params=dict(ae_architecture='global',ae_dropout=.1,latent_dim=4,
        hidden_dims=[8],mask_predictor_hidden_dim=8,encoder_blocks=1,decoder_blocks=1,
        max_epochs=2,batch_size=3,training_engine=engine)
    prediction,metadata=run_method('jingwei-dcae',values,hidden,params,device='cpu',seed=9,folder=tmp_path)
    np.testing.assert_array_equal(prediction[np.isfinite(values)],values[np.isfinite(values)])
    assert np.isfinite(prediction).all()
    assert metadata['ae_architecture']=='global' and metadata['mask_head_input']=='latent'
    assert metadata['training_steps']==6


def test_original_notebook_rejects_new_model_instead_of_silently_relabeling_it(tmp_path,capsys):
    from jingwei_benchmark.cli import main
    assert main(['benchmark','--evaluation-protocol','original-notebook',
        '--ae-architecture','global','--methods','jingwei-dcae','--output',str(tmp_path/'out')])==1
    assert 'requires the original AE' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()
    config=tmp_path/'parameters.json'
    config.write_text(json.dumps({'dcae':{'ae_architecture':'global'}}))
    assert main(['benchmark','--evaluation-protocol','original-notebook','--methods','dcae',
        '--method-config',str(config),'--output',str(tmp_path/'other')])==1
    assert not (tmp_path/'other').exists()


def test_cli_global_ae_option_reaches_worker_and_is_recorded(tmp_path):
    from jingwei_benchmark.cli import main
    source=tmp_path/'data.csv'
    np.savetxt(source,np.random.default_rng(4).normal(size=(9,17)),delimiter=',')
    output=tmp_path/'out'
    assert main(['benchmark','--input',str(source),'--scale','log2','--transform','none',
        '--methods','jingwei-dcae','--ae-architecture','global','--max-epochs','1',
        '--latent-dim','4','--hidden-dims','8','--batch-size','4','--device','cpu',
        '--output',str(output)])==0
    rows=json.loads((output/'metrics.json').read_text())
    assert rows[0]['ae_architecture']=='global' and rows[0]['status']=='completed'


@pytest.mark.skipif(os.name!='nt',reason='Windows-only desktop')
def test_gui_global_selection_and_reset(monkeypatch,tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow,ParameterDialog,set_value,value
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    window=BenchmarkWindow(tmp_path/'desktop',load_default=False)
    dialog=ParameterDialog('jingwei-dcae',{'ae_architecture':'global'},window)
    assert dialog.values()['ae_architecture']=='global'
    assert dialog.fields['ae_dropout'].isEnabled()
    dialog._reset('jingwei-dcae')
    assert value(dialog.fields['ae_architecture'])=='native'
    assert not dialog.fields['ae_dropout'].isEnabled()
    from jingwei_benchmark.gui import RunTableModel
    table=RunTableModel(window.t)
    table.replace([{'method':'jingwei-dcae','ae_architecture':'global'}],['method'])
    assert 'jingwei-dcae' in table.data(table.index(0,0))
    assert table.data(table.index(0,0)) != 'jingwei-dcae'
    dialog.close()
    window.close()
