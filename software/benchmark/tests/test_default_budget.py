import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest

from jingwei_benchmark.cli import parser
from jingwei_benchmark.gui_jobs import write_json,read_json


def test_cli_defaults_to_500_epochs_without_internal_validation_or_early_stop(tmp_path):
    for command in ('benchmark','impute'):
        args=parser().parse_args([command,'--input',str(tmp_path/'input.csv'),'--output',str(tmp_path/'out')])
        assert args.max_epochs==500 and args.training_policy=='fixed'


@pytest.mark.parametrize('fields,expected',[
    ({'max_epochs':50},('fixed',500)),
    ({'max_epochs':5000},('fixed',500)),
    ({'training_policy':'convergence','max_epochs':5000},('fixed',500)),
    ({'training_policy':'fixed','max_epochs':50},('fixed',500)),
    ({'training_policy':'convergence','max_epochs':50},('fixed',500)),
    ({'training_policy':'fixed','max_epochs':123},('fixed',123)),
    ({'training_policy':'convergence','max_epochs':123},('convergence',123)),
])
def test_gui_migrates_old_defaults_but_preserves_custom_budgets(tmp_path,monkeypatch,fields,expected):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow,value
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    write_json(tmp_path/'session.json',{'schema':1,'fields':fields,'parameters':{
        'jingwei-dmf':{'max_epochs':5000},'knn':{'k':3}},'job_name':'Keep me'})
    window=BenchmarkWindow(tmp_path,load_default=False)
    try:
        assert (value(window.fields['training_policy']),value(window.fields['max_epochs']))==expected
        assert window.method_parameters['jingwei-dmf']['max_epochs']==500
        assert window.method_parameters['knn']['k']==3 and window.job_name.text()=='Keep me'
        assert window.run_settings_form.rowCount()==3
    finally:window.close()
    assert read_json(tmp_path/'session.json')['fixed_budget_version']==1


def test_gui_fresh_neural_defaults_all_use_500(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow,value
    from jingwei_benchmark.gui_parameters import PARAMETERS
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    window=BenchmarkWindow(tmp_path,load_default=False)
    try:
        assert value(window.fields['training_policy'])=='fixed'
        assert value(window.fields['max_epochs'])==500
        for name in ('jingwei-dmf','jingwei-dcae','pimms-dae','pimms-vae','pimms-cf'):
            assert PARAMETERS[name]['max_epochs'][1]==500
    finally:window.close()
