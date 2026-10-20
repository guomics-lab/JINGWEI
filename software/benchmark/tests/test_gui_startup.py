import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from jingwei_benchmark.gui import BenchmarkWindow,set_value
from jingwei_benchmark.gui_startup import StartupGate
from jingwei_benchmark.methods import METHODS


def states():
    return {n:{'available':True,'backend':m.backend,'gpu':m.gpu,'reason':None} for n,m in METHODS.items()}


def test_workbench_waits_for_probe_and_retry_then_opens(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    pending=[]
    monkeypatch.setattr(BenchmarkWindow,'background',lambda self,fn,done,*a,**kw:pending.append((done,kw.get('error'))))
    win=BenchmarkWindow(tmp_path,load_default=False,startup=True)
    errors=[]
    win.show_error=errors.append
    gate=StartupGate(win,hidden=True)
    gate.start()
    app.processEvents()
    assert not win.isVisible() and not gate.revealed
    pending[-1][1]('probe failed')
    gate.poll()
    assert not errors and not win.isVisible() and gate.retry_button.isVisible()
    gate.retry()
    assert win._environment_checking and not win.isVisible()
    pending[-1][0]((states(),False,None))
    gate.poll()
    assert gate.ready_before_show and gate.revealed and win.isVisible()
    assert not win._environment_checking and not win._startup_pending
    assert not gate.panel.isVisible()
    win.close()


def test_cancel_during_probe_never_opens_workbench(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'background',lambda *a,**kw:None)
    win=BenchmarkWindow(tmp_path,load_default=False,startup=True)
    gate=StartupGate(win,hidden=True)
    gate.start()
    gate.panel.close()
    assert win._closed and win._environment_cancel.is_set()
    assert not win.isVisible() and not gate.revealed


def test_three_run_controls_and_shared_rate_in_both_tasks(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:self.apply_environment((states(),False,None)))
    win=BenchmarkWindow(tmp_path,load_default=False)
    assert win.run_settings_form.rowCount()==3
    assert not {'evaluation_protocol','seed','repeats','training_mechanism','training_mask_rate'} & win.fields.keys()
    assert not hasattr(win,'protocol_summary')
    assert win.brand_label.text()=='jingwei' and win.windowTitle().startswith('jingwei ')
    set_value(win.fields['mask_rate'],.35)
    for mode in ('benchmark','impute'):
        win.fields['command'].setCurrentIndex(win.fields['command'].findData(mode))
        config=win.configuration()
        assert config['mask_rate']==.35
        assert not {'training_mask_rate','training_mechanism'} & config.keys()
        assert config['evaluation_protocol']=='fair' and config['seed']==0
        assert win.fields['mechanism'].isEnabled()
    win.fields['mechanism'].setCurrentIndex(win.fields['mechanism'].findData('mnar-jingwei-original'))
    assert win.mask_rate_stack.currentIndex()==1 and not win.fields['mask_rate'].isEnabled()
    win.set_language('en')
    assert win.automatic_mask_rate.text()=='Fixed probabilities'
    assert win.fields['mechanism'].currentText()=='Legacy rule (old tasks)'
    win.close()
