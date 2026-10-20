"""Updated default DCAE, public/bundled parity and the AD evaluation flow."""
from pathlib import Path
import os

import numpy as np
import pytest

torch=pytest.importorskip('torch')


def test_dcae_itself_uses_the_updated_architecture_without_any_dataset_preset():
    from jingwei_benchmark.presets import preset_parameters
    from jingwei_benchmark.ae_models import GlobalDCAE,build_dcae
    from jingwei_benchmark._native.models import DCAE
    assert preset_parameters('standard','jingwei-dcae')=={}
    assert DCAE is GlobalDCAE
    assert isinstance(build_dcae(input_feature_dim=17,ae_dim=4),DCAE)
    with pytest.raises(ValueError,match='Unknown JINGWEI preset'):
        preset_parameters('alzheimer-global-ae','jingwei-dcae')


def test_cli_uses_updated_dcae_without_an_architecture_or_dataset_option(tmp_path):
    import json
    from jingwei_benchmark.cli import main
    source=tmp_path/'any-dataset.csv'
    np.savetxt(source,np.random.default_rng(4).normal(size=(9,17)),delimiter=',')
    output=tmp_path/'out'
    assert main(['benchmark','--input',str(source),'--scale','log2','--transform','none',
        '--methods','jingwei-dcae','--max-epochs','1','--latent-dim','4',
        '--hidden-dims','8','--batch-size','4','--device','cpu','--output',str(output)])==0
    rows=json.loads((output/'metrics.json').read_text())
    assert rows[0]['ae_architecture']=='global' and rows[0]['mask_head_input']=='latent'


def test_public_and_bundled_global_model_are_numerically_identical(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'src'))
    from jingwei import GlobalDCAE as Public
    from jingwei_benchmark.ae_models import GlobalDCAE as Bundled
    torch.set_num_threads(1)
    kwargs=dict(input_feature_dim=17,ae_dim=4,mask_predictor_hidden_dim=8,
        num_encoder_blocks=1,num_decoder_blocks=1,dilation=2)
    torch.manual_seed(7); public=Public(**kwargs).eval()
    torch.manual_seed(7); bundled=Bundled(**kwargs).eval()
    for key,value in public.state_dict().items():
        torch.testing.assert_close(value,bundled.state_dict()[key],rtol=0,atol=0)
    mask=torch.rand(3,17)>.4
    x=torch.randn(3,17); x[~mask]=float('nan')
    for a,b in zip(public(x,mask),bundled(x,mask)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('architecture',['native','global'])
def test_public_imputer_matches_bundled_reconstruction_and_inference(architecture,monkeypatch):
    # Architecture, reconstruction and inference remain shared. The standalone
    # app's corrected full-mask BCE is independently tested in test_torch_engine;
    # this change does not revise the separately published package's objective.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'src'))
    from jingwei import DCAEImputer as Public
    from jingwei_benchmark._native.dcae import DCAEImputer as Bundled
    torch.set_num_threads(1)
    torch.manual_seed(20)
    data=torch.randn(4,17)
    supervision=torch.ones_like(data);supervision[0,0]=0;data[0,0]=float('nan')
    visible=supervision.clone();visible[1,1:4]=0
    kwargs=dict(full_data_tensor=data,full_mask_tensor=visible,supervision_mask_tensor=supervision,
        ae_dim=4,mask_predictor_hidden_dim=8,num_encoder_blocks=1,num_decoder_blocks=1,
        dilation=2,lambda_mask=0.,ae_architecture=architecture)
    torch.manual_seed(30);public=Public(**kwargs)
    torch.manual_seed(30);bundled=Bundled(**kwargs)
    batch=next(iter(public.train_dataloader()))
    monkeypatch.setattr(public,'log_dict',lambda *a,**kw:None)
    monkeypatch.setattr(bundled,'log_dict',lambda *a,**kw:None)
    torch.manual_seed(40);a=public.training_step(batch,0)
    torch.manual_seed(40);b=bundled.training_step(batch,0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.backward();b.backward()
    for p,q in zip(public.parameters(),bundled.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    actual=public.get_imputed_data()
    torch.testing.assert_close(actual,bundled.get_imputed_data(),rtol=0,atol=0)
    torch.testing.assert_close(actual[supervision.bool()],data[supervision.bool()],rtol=0,atol=0)
    from jingwei.models import DCAE
    assert isinstance(Public(data.nan_to_num(),visible).model,DCAE)


@pytest.mark.skipif(os.name!='nt',reason='Windows-only desktop')
def test_gui_keeps_one_dcae_without_a_dataset_or_architecture_selector(monkeypatch,tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow,ParameterDialog,set_value,value
    from jingwei_benchmark.presets import preset_parameters
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    window=BenchmarkWindow(tmp_path/'desktop',load_default=False)
    field=window.fields['jingwei_preset']
    assert all(field.itemData(i)!='alzheimer-global-ae' for i in range(field.count()))
    dialog=ParameterDialog('jingwei-dcae',{},window)
    assert 'ae_architecture' not in dialog.values()
    assert dialog.fields['ae_dropout'].isEnabled()
    dialog.close();window.close()
