"""User journeys for immutable running settings and exact configuration reuse."""
import json
import os
import re
from pathlib import Path
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton
from jingwei_benchmark.gui import BenchmarkWindow, ParameterDialog, set_value
from jingwei_benchmark.gui_jobs import TERMINAL
from jingwei_benchmark.methods import METHODS


def until(app, predicate):
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate():return
        QTest.qWait(20)
    pytest.fail('GUI configuration journey timed out')


@pytest.fixture
def window(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    states={name:{'available':True,'reason':None,'backend':spec.backend,'gpu':spec.gpu}
            for name,spec in METHODS.items()}
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',
        lambda self,**kw:self.apply_environment((states,False,None)))
    win=BenchmarkWindow(tmp_path/'workspace',load_default=False)
    errors=[];win.show_error=errors.append
    source=tmp_path/'input.csv'
    pd.DataFrame(np.random.default_rng(5).lognormal(4,.4,(12,8)),
        index=[f'S{i}' for i in range(12)],columns=[f'P{i}' for i in range(8)]).to_csv(source)
    win.set_input(source)
    until(app,lambda:win.preview_data is not None)
    set_value(win.fields['device'],'cpu')
    win.select_methods(['mean'])
    yield app,win,source,errors
    for info in win.store.active():win.store.cancel(info['id'],force=True)
    win.close();win.pool.waitForDone(10000);app.processEvents()


def test_running_and_cancelling_lock_configuration_but_keep_stop_and_logs(window,monkeypatch):
    app,win,source,errors=window
    win.select_methods(['jingwei-dmf'])
    for key,setting in {'max_epochs':100000,'latent_dim':4,'hidden_dims':[8],'batch_size':4}.items():
        set_value(win.fields[key],setting)
    before=win.configuration()
    win.start_job()
    assert win._submitting and not win.configuration_panel.isEnabled()
    assert not win.fields['max_epochs'].isEnabled()
    until(app,lambda:win.selected_job is not None)
    win.refresh_jobs()
    assert win.cancel_button.isEnabled() and not win.clone_button.isEnabled()
    assert all(not win.advanced_tabs.widget(i).isEnabled() for i in range(win.advanced_tabs.count()))
    assert win.log_view.isEnabled() and win.result_tabs.isEnabled()
    monkeypatch.setattr('jingwei_benchmark.gui.ParameterDialog',lambda *a,**kw:pytest.fail('Opened an editable running parameter dialog'))
    win.edit_parameters('jingwei-dmf')
    name=win.job_name.text();win.clone_job()
    assert win.job_name.text()==name
    win.set_language('en');win.refresh_environment()
    assert not win.configuration_panel.isEnabled() and not win.fields['max_epochs'].isEnabled()
    assert win.configuration()==before
    QTest.mouseClick(win.cancel_button,Qt.LeftButton)
    assert not win.configuration_panel.isEnabled()
    until(app,lambda:win.store.info(win.selected_job)['state'] in TERMINAL)
    win.refresh_jobs()
    assert win.configuration_panel.isEnabled() and win.fields['max_epochs'].isEnabled()
    assert win.clone_button.isEnabled() and win.start_button.isEnabled() and not errors


def test_imputation_exposes_training_mask_settings_and_keeps_them_on_language_switch(window):
    app,win,source,errors=window
    win.fields['command'].setCurrentIndex(win.fields['command'].findData('impute'))
    win.fields['mechanism'].setCurrentIndex(win.fields['mechanism'].findData('mnar-logistic'))
    set_value(win.fields['mask_rate'],.3)
    assert win.fields['mechanism'].isEnabled() and win.fields['mask_rate'].isEnabled()
    assert win.fields['logistic_slope'].isEnabled()
    assert win.field_labels['mechanism'].text() == win.t('训练缺失机制')
    win.set_language('en')
    assert win.field_labels['mask_rate'].text() == 'Training mask rate'
    assert win.configuration()['mask_rate'] == pytest.approx(.3)
    win.fields['mechanism'].setCurrentIndex(win.fields['mechanism'].findData('mnar-jingwei-original'))
    assert win.fields['mechanism'].isEnabled() and not win.fields['mask_rate'].isEnabled()
    assert win.mask_rate_stack.currentIndex() == 1
    win.fields['command'].setCurrentIndex(win.fields['command'].findData('benchmark'))
    assert win.field_labels['mask_rate'].text() == 'Mask rate'


def test_submission_failure_unlocks_configuration(window,monkeypatch):
    app,win,source,errors=window
    def fail(*a,**kw):raise OSError('cannot write selected output')
    monkeypatch.setattr(win.store,'submit',fail)
    win.start_job()
    assert not win.configuration_panel.isEnabled()
    until(app,lambda:bool(errors))
    assert not win._submitting and win.configuration_panel.isEnabled()
    assert win.fields['max_epochs'].isEnabled() and win.start_button.isEnabled()


def test_gui_imputation_runs_native_training_mask_and_preserves_input(window):
    app,win,source,errors=window
    win.select_methods(['jingwei-dmf','jingwei-dcae','mean'])
    win.fields['command'].setCurrentIndex(win.fields['command'].findData('impute'))
    for key,setting in {'training_policy':'fixed','mask_rate':.3,'max_epochs':2,'latent_dim':4,'hidden_dims':[8],'batch_size':4,'threads':1}.items():
        set_value(win.fields[key],setting)
    until(app,lambda:win.preview_data is not None)
    win.start_job()
    assert not win.configuration_panel.isEnabled()
    until(app,lambda:win.selected_job is not None)
    until(app,lambda:win.store.info(win.selected_job)['state'] in TERMINAL)
    win.refresh_jobs()
    info=win.store.info(win.selected_job)
    assert info['state']=='completed' and not errors
    folder=Path(info['results_folder'])/'repeat_000'
    with np.load(folder/'masks.npz') as masks:
        assert not masks['evaluation_hidden'].any() and masks['training_hidden'].any()
        hidden_count=int(masks['training_hidden'].sum())
    for method in ('jingwei-dmf','jingwei-dcae'):
        detail=json.loads((folder/method/'worker_result.json').read_text(encoding='utf-8'))
        assert detail['training_hidden_entries']==hidden_count
    with np.load(folder/'method_input.npz') as common:
        assert not common['training_hidden'].any()
    original=pd.read_csv(source,index_col=0).to_numpy()
    for method in ('jingwei-dmf','jingwei-dcae','mean'):
        predicted=pd.read_csv(folder/method/'imputed.csv',index_col=0).to_numpy()
        np.testing.assert_allclose(predicted,original)
    win.clone_job()
    assert win.configuration()['command']=='impute' and win.configuration()['mask_rate']==pytest.approx(.3)
    assert win.fields['mechanism'].isEnabled() and win.fields['mask_rate'].isEnabled()


def test_copy_cli_configuration_preserves_effective_overrides_and_can_run_again(window,tmp_path):
    app,win,source,errors=window
    from jingwei_benchmark.cli import main
    overrides=tmp_path/'parameters.json';overrides.write_text(json.dumps({'knn':{'k':3}}))
    output=tmp_path/'saved-cli'
    assert main(['benchmark','--input',str(source),'--methods','mean','knn',
        '--max-epochs','7','--batch-size','3','--hidden-dims','12','6',
        '--mechanism','mnar-logistic','--mask-rate','.15','--device','cpu',
        '--w2-features','P0','P1','P2',
        '--method-config',str(overrides),'--output',str(output)])==0
    original=(output/'manifest.json').read_bytes()
    overrides.unlink()  # Reuse must not depend on the original override file.
    set_value(win.fields['max_epochs'],99)
    win.method_parameters={'knn':{'k':9}}
    win.open_task(output)
    win.clone_job()
    until(app,lambda:win.preview_data is not None)
    config=win.configuration()
    assert config['methods']==['mean','knn']
    assert config['max_epochs']==7 and config['batch_size']==3 and config['hidden_dims']==[12,6]
    assert config['mechanism']=='mnar-logistic' and config['mask_rate']==pytest.approx(.15)
    assert win.method_parameters['knn']=={'k':3}
    assert win.feature_selection==['P0','P1','P2']
    dialog=ParameterDialog('knn',win.method_parameters['knn'],win)
    assert dialog.values()['k']==3;dialog.close()
    assert win.output_root.text()==str(output.parent)
    win.method_parameters['knn']['k']=2
    assert (output/'manifest.json').read_bytes()==original
    assert json.loads((output/'repeat_000/knn/job.json').read_text(encoding='utf-8'))['parameters']['k']==3
    win.method_parameters['knn']['k']=3
    old=win.selected_job;win.start_job()
    until(app,lambda:win.selected_job!=old and not win._submitting)
    until(app,lambda:win.store.info(win.selected_job)['state'] in TERMINAL)
    info=win.store.info(win.selected_job)
    assert info['state']=='completed' and not errors
    assert json.loads((Path(info['results_folder'])/'repeat_000/knn/job.json').read_text(encoding='utf-8'))['parameters']['k']==3


@pytest.mark.parametrize('language',['zh','en'])
def test_open_task_button_is_removed_in_both_languages(window,language):
    app,win,source,errors=window
    win.set_language(language)
    labels={button.text() for button in win.findChildren(QPushButton)}
    assert not {'打开任务…','Open task…'} & labels
    assert win.clone_button.text()==win.t('复制配置')


def test_blank_task_name_is_shown_as_timestamp_in_the_workbench(window):
    app,win,source,errors=window
    assert not win.job_name.text()
    win.start_job()
    until(app,lambda:win.selected_job is not None)
    win.refresh_jobs()
    name=win.store.info(win.selected_job)['name']
    assert re.fullmatch(r'task-\d{8}-\d{6}',name)
    assert win.current_task_label.text()==name
    until(app,lambda:win.store.info(win.selected_job)['state'] in TERMINAL)
    assert not errors
