"""Damaged saved tasks must not crash the workbench or poison its registry."""
import json
import os

import pytest

from jingwei_benchmark.gui_jobs import JobStore


def manifest():
    return {'config': {'command': 'benchmark', 'methods': ['mean'], 'repeats': 1},
            'runs': [{'repeat': 0, 'method': 'mean', 'status': 'completed',
                      'MSE': .2, 'time_s': .1}]}


@pytest.mark.parametrize('damage', ['array', 'null-row', 'string-row', 'bad-method',
    'bad-repeat', 'missing-status', 'bad-metric', 'infinite-metric', 'huge-metric', 'bad-repeats', 'huge-repeats'])
@pytest.mark.parametrize('kind', ['cli', 'gui'])
def test_bad_saved_records_are_rejected_without_registering_a_task(tmp_path, damage, kind):
    store=JobStore(tmp_path/'workspace')
    folder=tmp_path/'damaged';folder.mkdir()
    record=manifest()
    if damage=='array':record=[]
    elif damage=='null-row':record['runs']=[None]
    elif damage=='string-row':record['runs']=['broken']
    elif damage=='bad-method':record['runs'][0]['method']='../../outside'
    elif damage=='bad-repeat':record['runs'][0]['repeat']='0'
    elif damage=='missing-status':record['runs'][0].pop('status')
    elif damage=='bad-metric':record['runs'][0]['MSE']='broken'
    elif damage=='infinite-metric':record['runs'][0]['MSE']=float('inf')
    elif damage=='huge-metric':record['runs'][0]['MSE']=10**500
    elif damage=='huge-repeats':record['config']['repeats']=10**500
    else:record['config']['repeats']='1'
    target=folder
    if kind=='gui':
        (folder/'request.json').write_text(json.dumps({'schema_version':1,'id':'a'*32,
            'config':manifest()['config']}))
        (folder/'completion.json').write_text('{"return_code":0}')
        target=folder/'results';target.mkdir()
    (target/'manifest.json').write_text(json.dumps(record))
    # In a GUI task config lives in request.json, not the result manifest.
    if kind=='gui' and damage in ('bad-repeats','huge-repeats'):
        request=json.loads((folder/'request.json').read_text())
        request['config']['repeats']=record['config']['repeats']
        (folder/'request.json').write_text(json.dumps(request))
    with pytest.raises(ValueError):store.attach(folder)
    assert not store._folders and not store._cli_jobs


def test_cli_manifest_damaged_after_opening_is_a_controlled_error(tmp_path):
    store=JobStore(tmp_path/'workspace')
    folder=tmp_path/'task';folder.mkdir()
    path=folder/'manifest.json';path.write_text(json.dumps(manifest()))
    job=store.attach(folder)
    assert store.info(job)['state']=='completed'
    path.write_text('[]')
    with pytest.raises(ValueError):store.info(job)


@pytest.mark.skipif(os.name!='nt',reason='Windows-only workbench')
def test_opening_damaged_task_preserves_current_table_and_log(tmp_path, monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    window=BenchmarkWindow(tmp_path/'workspace',load_default=False)
    errors=[];window.show_error=errors.append
    try:
        folder=tmp_path/'valid';folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps(manifest()))
        logs=folder/'repeat_000/mean';logs.mkdir(parents=True)
        (logs/'run.log').write_text('Completed mean')
        window.open_task(folder)
        selected=window.selected_job
        window.result_table.selectRow(0);app.processEvents()
        assert window.run_model.rowCount()==1 and 'Completed mean' in window.log_view.toPlainText()
        bad=tmp_path/'bad';bad.mkdir();(bad/'manifest.json').write_text('[]')
        window.open_task(bad)
        assert errors and window.selected_job==selected and window.run_model.rowCount()==1
        # Damage while a task is already displayed: polling must reset safely.
        (folder/'manifest.json').write_text('[]')
        window.refresh_jobs()
        assert window.selected_job is None and window.run_model.rowCount()==0
    finally:window.close()
