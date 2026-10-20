"""Background Qt user journeys; these never send input to another application."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtTest import QTest
from jingwei_benchmark.gui import BenchmarkWindow, set_value
from jingwei_benchmark.gui_jobs import TERMINAL
from jingwei_benchmark.methods import METHODS


def until(app, predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(20)
    pytest.fail('user journey timed out')


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    states = {name: {'available': True, 'backend': spec.backend, 'gpu': spec.gpu, 'reason': None}
              for name, spec in METHODS.items()}
    monkeypatch.setattr(BenchmarkWindow, 'refresh_environment', lambda self: self.apply_environment((states, False, None)))
    win = BenchmarkWindow(tmp_path / 'desktop', load_default=False)
    win.mode_changed()
    errors = []
    win.show_error = errors.append
    source = tmp_path / '用户 数据.csv'
    values = np.random.default_rng(5).lognormal(5, .5, (12, 8))
    values[8, 2] = np.nan
    pd.DataFrame(values, index=[f'样本 {i}' for i in range(12)], columns=[f'P{i}' for i in range(8)]).to_csv(source, encoding='utf-8-sig')
    win.set_input(source)
    until(app, lambda: win.preview_data is not None)
    win.select_methods(['mean', 'knn'])
    assert 'repeats' not in win.fields
    set_value(win.fields['device'], 'cpu')
    yield app, win, source, errors
    for job in win.store.history():
        if job['state'] not in TERMINAL:
            win.store.cancel(job['id'], force=True)
    win.close()
    win.pool.waitForDone()
    app.processEvents()


def test_late_failed_preview_cannot_replace_successful_new_input(window, monkeypatch):
    app, win, source, errors = window
    callbacks = []
    monkeypatch.setattr(win, 'background', lambda fn, done, *a, **kw: callbacks.append((done, kw['error'])))
    win.inspect_data()
    old_failure = callbacks[-1][1]
    replacement = source.with_name('replacement.csv')
    pd.DataFrame(np.ones((4, 3)), index=list('ABCD'), columns=['X', 'Y', 'Z']).to_csv(replacement)
    win.set_input(replacement)
    from jingwei_benchmark.gui_jobs import inspect_input
    callbacks[-1][0](inspect_input(replacement, win.input_options()))
    before = win.data_summary.text()
    old_failure('old file read failed')
    assert not errors
    assert win.data_summary.text() == before
    assert win.preview_data['samples'] == 4 and win.start_button.isEnabled()


def test_changed_file_is_rejected_and_empty_method_selection_recovers(window):
    app, win, source, errors = window
    win.select_methods([])
    win.start_job()
    assert errors and not win.store.history()
    errors.clear()
    win.select_methods(['mean'])
    source.write_text(source.read_text(encoding='utf-8-sig') + '\n', encoding='utf-8')
    win.start_job()
    assert errors and not win.store.history()
    errors.clear()
    win.inspect_data()
    until(app, lambda: not win._inspecting)
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    until(app, lambda: win.store.info(win.selected_job)['state'] in TERMINAL)
    assert win.store.info(win.selected_job)['state'] == 'completed' and not errors


def test_benchmark_plot_export_clone_then_imputation(window, tmp_path, monkeypatch):
    app, win, source, errors = window
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    until(app, lambda: win.store.info(win.selected_job)['state'] in TERMINAL)
    win.refresh_jobs()
    info = win.store.info(win.selected_job)
    assert info['state'] == 'completed' and info['done'] == 2
    win.set_language('en')
    for metric in ('MSE', 'RMSE', 'MAE', 'NRMSE', 'Pearson', 'Spearman', 'W2', 'time_s'):
        win.metric_choice.setCurrentIndex(win.metric_choice.findData(metric))
        assert len(win.plot.source) == 2
    for extension, magic in [('png', b'\x89PNG'), ('pdf', b'%PDF'), ('svg', b'<?xml')]:
        target = tmp_path / ('图表.' + extension)
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, p=target, **kw: (str(p), ''))
        win.export_plot()
        assert target.read_bytes().startswith(magic)
        assert len(pd.read_csv(target.with_suffix('.source.csv'))) == 2
        assert target.with_suffix('.provenance.json').is_file()
    archive = tmp_path / '任务.zip'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(archive), ''))
    win.export_job()
    until(app, archive.is_file)
    until(app, lambda: not win._tasks)
    with zipfile.ZipFile(archive) as exported:
        assert exported.testzip() is None
        assert 'results/report.html' in exported.namelist()
        assert 'results/repeat_000/masks.npz' in exported.namelist()
        assert 'results/repeat_001/masks.npz' not in exported.namelist()
    win.clone_job()
    until(app, lambda: win.preview_data is not None)
    assert win.configuration()['methods'] == ['mean', 'knn']
    set_value(win.fields['command'], 'impute')
    win.mode_changed()
    previous_job = win.selected_job
    win.start_job()
    until(app, lambda: win.selected_job != previous_job and not win._submitting)
    until(app, lambda: win.store.info(win.selected_job)['state'] in TERMINAL)
    completed = win.store.info(win.selected_job)
    assert completed['state'] == 'completed'
    assert completed['config']['repeats'] == 1
    original = pd.read_csv(source, index_col=0).to_numpy()
    predicted = pd.read_csv(Path(completed['folder'])/'results/repeat_000/mean/imputed.csv', index_col=0).to_numpy()
    np.testing.assert_allclose(predicted[np.isfinite(original)], original[np.isfinite(original)])
    assert np.isfinite(predicted).all() and not errors


def test_close_running_job_defaults_no_then_confirmed_stop_is_durable(window):
    app, win, source, errors = window
    win.select_methods(['jingwei-dmf'])
    for key, setting in {'max_epochs': 100000, 'batch_size': 4, 'latent_dim': 4, 'hidden_dims': '8'}.items():
        set_value(win.fields[key], setting)
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    job = win.selected_job
    def answer_no():
        box = QApplication.activeModalWidget()
        assert isinstance(box, QMessageBox)
        assert box.defaultButton() == box.button(QMessageBox.No)
        box.button(QMessageBox.No).click()
    QTimer.singleShot(30, answer_no)
    assert not win.close()
    assert not win._closed and win.store.info(job)['state'] not in TERMINAL
    QTimer.singleShot(30, lambda: QApplication.activeModalWidget().button(QMessageBox.Yes).click())
    assert win.close()
    assert win.store.info(job)['state'] == 'cancelled'
    from jingwei_benchmark.gui_jobs import JobStore
    reopened = JobStore(win.store.root)
    reopened.attach(win.store.info(job)['folder'])
    assert reopened.info(job)['state'] == 'cancelled' and not errors


def test_custom_output_manual_open_moved_task_and_live_log(window, tmp_path):
    app, win, source, errors = window
    output = tmp_path / '自选 output'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('existing data')
    win.output_root.setText(str(output))
    win.select_methods(['mean'])
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    until(app, lambda: win.store.info(win.selected_job)['state'] in TERMINAL)
    win.refresh_jobs()
    folder = Path(win.store.info(win.selected_job)['folder'])
    assert folder.parent == output and sentinel.read_text() == 'existing data'
    assert 'mean' in win.log_view.toPlainText() and 'Completed' in win.log_view.toPlainText()
    win.result_table.selectRow(0)
    log = folder / 'results/repeat_000/mean/run.log'
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\nLIVE LOG UPDATE\n')
    win.refresh_jobs()  # Same result digest; logs must still update.
    assert 'LIVE LOG UPDATE' in win.log_view.toPlainText()
    win.close()
    reopened = BenchmarkWindow(win.store.root, load_default=False)
    try:
        assert reopened.selected_job is None and reopened.run_model.rowCount() == 0
        assert reopened.output_root.text() == str(output)
        assert not hasattr(reopened, 'job_combo')
        moved = output / 'moved task'
        folder.rename(moved)
        before = (moved / 'request.json').read_bytes()
        reopened.open_task(moved)
        assert reopened.store.info(reopened.selected_job)['state'] == 'completed'
        assert reopened.run_model.rowCount() == 1 and not reopened.cancel_button.isEnabled()
        assert (moved / 'request.json').read_bytes() == before
        moved.rename(output / 'moved again')
        reopened.refresh_jobs()
        assert reopened.selected_job is None and reopened.run_model.rowCount() == 0
    finally:
        reopened.close()
    assert not errors


def test_open_cli_result_and_invalid_folder_preserves_current_task(window, tmp_path):
    app, win, source, errors = window
    from jingwei_benchmark.cli import main
    target = tmp_path / 'cli results'
    assert main(['benchmark', '--input', str(source), '--output', str(target),
                 '--methods', 'mean', '--repeats', '1', '--device', 'cpu']) == 0
    before = (target / 'manifest.json').read_bytes()
    win.open_task(target)
    selected = win.selected_job
    assert selected and win.run_model.rowCount() == 1
    assert (target / 'manifest.json').read_bytes() == before
    win.open_task(tmp_path)
    assert errors and win.selected_job == selected


def test_original_mnar_gui_runs_one_shared_reference_mask(window, tmp_path):
    app, win, source, errors = window
    import torch
    from jingwei_benchmark._legacy.datasets import CSVDataset
    from jingwei_benchmark.data_io import load_matrix
    win.fields['mechanism'].setCurrentIndex(win.fields['mechanism'].findData('mnar-jingwei-original'))
    assert 'repeats' not in win.fields and win.configuration()['repeats'] == 1
    assert not win.fields['mask_rate'].isEnabled() and win.mask_rate_stack.currentIndex()==1
    assert not hasattr(win,'protocol_summary')
    assert win.run_settings_form.rowCount()==3
    win.select_methods(['mean'])
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    until(app, lambda: win.store.info(win.selected_job)['state'] in TERMINAL)
    win.refresh_jobs()
    info = win.store.info(win.selected_job)
    assert info['state'] == 'completed' and info['done'] == 1 and not errors
    actual = np.load(Path(info['folder']) / 'results/repeat_000/masks.npz')
    from jingwei_benchmark.gui_jobs import read_json
    meta=read_json(Path(info['folder'])/'results/repeat_000/mask_config.json')
    result=read_json(Path(info['folder'])/'results/metrics.json')[0]
    empty=~(actual['observed'] & ~actual['evaluation_hidden']).any(0)
    assert empty.any() and result['zero_cols']==int(empty.sum())
    assert result['n_excluded']==int(actual['evaluation_hidden'][:,empty].sum())
    exported=pd.read_csv(Path(info['folder'])/'results/repeat_000/mean/imputed.csv',index_col=0).to_numpy()
    assert np.isnan(exported[:,empty]).all() and np.isfinite(exported[:,~empty]).all()
    working=load_matrix(source).frame.to_numpy()
    for stage,eligible,key in [('evaluation',actual['observed'],'evaluation_hidden')]:
        path=tmp_path/(stage+'.csv')
        pd.DataFrame(np.where(eligible,working,np.nan)).to_csv(path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(meta[stage]['random_seed'])
            expected=CSVDataset(path)
        np.testing.assert_array_equal(actual[key],eligible & ~expected.mask.numpy().astype(bool))
    assert not actual['training_hidden'].any()
    assert meta['training']['probability_mode']=='not-applied'
    assert win.paired.isHidden() and win.result_table.isColumnHidden(0)
    win.set_language('en')
    assert win.automatic_mask_rate.text()=='Auto'
    win.fields['mechanism'].setCurrentIndex(win.fields['mechanism'].findData('mcar'))
    assert win.fields['mask_rate'].isEnabled()
    assert win.mask_rate_stack.currentIndex()==0


def test_old_repeat_setting_cannot_reappear_in_new_gui_jobs(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    from jingwei_benchmark.gui_jobs import write_json
    monkeypatch.setattr(BenchmarkWindow, 'refresh_environment', lambda self: None)
    write_json(tmp_path / 'session.json', {'schema': 1, 'fields': {'repeats': 99, 'mechanism': 'mnar-jingwei'}, 'methods': ['knn']})
    win = BenchmarkWindow(tmp_path, load_default=False)
    try:
        assert 'repeats' not in win.fields
        win.backend_states = {'knn': {'available': True}}
        assert win.configuration()['repeats'] == 1
        assert win.configuration()['evaluation_protocol'] == 'fair'
        assert win.fields['mechanism'].currentText() == 'MNAR · 分位数（自定义）'
    finally:
        win.close()
