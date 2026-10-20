import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from jingwei_benchmark.gui import BenchmarkWindow, set_value
from jingwei_benchmark.gui_jobs import read_json, write_json
from jingwei_benchmark.methods import METHODS


def states(available=True):
    return {name: {"available": available, "backend": method.backend, "gpu": method.gpu, "reason": None}
            for name, method in METHODS.items()}


def until(app, predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    pytest.fail("desktop state was not restored")


def test_reopen_restores_form_and_skips_environment_probe(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import jingwei_benchmark.gui_environment as environment
    calls = []
    def probe(options, **kwargs):
        calls.append(dict(options))
        return states(), False, None
    monkeypatch.setattr(environment, "inspect_environment", probe)
    data = tmp_path / "proteins.csv"
    pd.DataFrame(np.arange(120).reshape(12, 10) + 1,
                 index=[f"S{i}" for i in range(12)], columns=[f"P{i}" for i in range(10)]).to_csv(data)
    workspace = tmp_path / "desktop"
    first = BenchmarkWindow(workspace, load_default=False)
    first.set_input(data)
    until(app, lambda: first.preview_data is not None and bool(first.backend_states))
    first.set_language("en")
    first.select_methods(["mean", "knn"])
    for key, setting in {"device": "cpu", "mask_rate": .15, "hidden_dims": "32, 16"}.items():
        set_value(first.fields[key], setting)
    first.job_name.setText("Remember this run")
    first.method_parameters = {"knn": {"k": 3}}
    first.feature_selection = ["P0", "P1"]
    expected = first.configuration()
    first.close()
    assert len(calls) == 1
    reopened = BenchmarkWindow(workspace)
    assert reopened._environment_cache_hit
    assert reopened._environment_probe_count == 0
    until(app, lambda: reopened.preview_data is not None)
    assert Path(reopened.input_path.text()) == data
    assert reopened.configuration() == expected
    assert reopened.method_parameters == {"knn": {"k": 3}}
    assert reopened.job_name.text() == "Remember this run"
    assert reopened.language == "en"
    assert reopened.start_button.isEnabled()
    assert len(calls) == 1
    reopened.set_language("zh")
    reopened.refresh_environment(force=True)
    until(app, lambda: len(calls) == 2 and not reopened._environment_checking)
    assert reopened.configuration() == expected
    reopened.close()
    assert read_json(workspace / "session.json")["parameters"] == {"knn": {"k": 3}}


def test_late_environment_probe_cannot_replace_new_settings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    callbacks = []
    monkeypatch.setattr(BenchmarkWindow, "background", lambda self, fn, done, *a, **kw: callbacks.append((done, kw.get("error"))))
    window = BenchmarkWindow(tmp_path, load_default=False)
    errors = []
    window.show_error = errors.append
    set_value(window.fields["mice_backend"], "r")
    window.refresh_environment(force=True)
    callbacks[-1][0]((states(), False, None))
    callbacks[0][0]((states(False), False, None))
    callbacks[0][1]("old probe failed")
    assert all(state["available"] for state in window.backend_states.values())
    assert not errors
    window.close()


def test_task_switch_reuses_environment_without_changing_backends(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import jingwei_benchmark.gui_environment as environment
    calls = []
    def probe(options, **kwargs):
        calls.append(dict(options))
        result = states()
        result['missforest'].update(available=False, reason='R package missing')
        return result, False, None
    monkeypatch.setattr(environment, 'inspect_environment', probe)
    window = BenchmarkWindow(tmp_path, load_default=False)
    until(app, lambda: bool(window.backend_states))
    for command in ('impute','benchmark'):
        window.fields['command'].setCurrentIndex(window.fields['command'].findData(command))
        assert not window.backend_states['missforest']['available']
        assert window.backend_states['missforest']['backend'] == 'r'
    assert len(calls) == 1 and window._environment_probe_count == 1
    window.close()


def test_old_protocol_seed_and_training_controls_cannot_restore(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow, 'refresh_environment', lambda self: None)
    write_json(tmp_path/'session.json', {'schema': 1, 'fields': {'evaluation_protocol': 'original-notebook', 'seed':91,'training_mask_rate':.8}})
    window=BenchmarkWindow(tmp_path, load_default=False)
    assert not {'evaluation_protocol','seed','training_mask_rate','training_mechanism'} & window.fields.keys()
    window.backend_states=states()
    assert window.configuration()['evaluation_protocol']=='fair'
    assert window.configuration()['seed']==0
    assert not {'training_mask_rate','training_mechanism'} & window.configuration().keys()
    assert window.selected_job is None and 'repeats' not in window.fields
    window.close()
    reopened=BenchmarkWindow(tmp_path, load_default=False)
    assert 'evaluation_protocol' not in reopened.fields
    assert reopened.selected_job is None
    reopened.close()


def test_missing_saved_input_and_invalid_fields_do_not_reset_to_demo(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow, "refresh_environment", lambda self: None)
    source = tmp_path / "moved.csv"
    write_json(tmp_path / "session.json", {"schema": 1, "input": str(source),
               "fields": {"repeats": "bad", "hidden_dims": 42}, "parameters": "bad"})
    window = BenchmarkWindow(tmp_path)
    assert window.input_path.text() == str(source)
    assert window.preview_data is None
    assert not window.start_button.isEnabled()
    assert "repeats" not in window.fields
    window.close()


@pytest.mark.parametrize('saved', [10**500,10**12,-10**500])
def test_oversized_saved_numeric_field_cannot_crash_startup(tmp_path, monkeypatch, saved):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    write_json(tmp_path/'session.json',{'schema':1,'fields':{'max_epochs':saved}})
    window=BenchmarkWindow(tmp_path,load_default=False)
    try:
        field=window.fields['max_epochs']
        assert field.value()==(field.maximum() if saved>0 else field.minimum())
    finally:window.close()


def test_small_saved_mask_rate_and_bad_method_override_are_restored_safely(tmp_path, monkeypatch):
    app=QApplication.instance() or QApplication([])
    from jingwei_benchmark.gui import ParameterDialog
    monkeypatch.setattr(BenchmarkWindow,'refresh_environment',lambda self:None)
    write_json(tmp_path/'session.json',{'schema':1,'fields':{'mask_rate':.0001},
        'parameters':{'knn':{'k':'broken'},'jingwei-dcae':{'latent_dim':10**500,'ae_dropout':.2}}})
    window=BenchmarkWindow(tmp_path,load_default=False)
    try:
        assert window.fields['mask_rate'].value()==pytest.approx(.0001)
        assert window.method_parameters=={'knn':{},'jingwei-dcae':{'ae_dropout':.2}}
        dialog=ParameterDialog('knn',{'k':'broken'},window)
        assert dialog.values()['k']==5
        dialog.close()
    finally:window.close()


def test_startup_does_not_open_legacy_job_or_restore_its_parameters(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    from jingwei_benchmark.gui import value
    from jingwei_benchmark.gui_jobs import JobStore
    import jingwei_benchmark.gui_environment as environment
    job_id = "a" * 32
    folder = tmp_path / "jobs" / job_id
    folder.mkdir(parents=True)
    pd.DataFrame(np.arange(20).reshape(5, 4) + 1,
                 index=[f"S{i}" for i in range(5)], columns=[f"P{i}" for i in range(4)]).to_csv(folder / "input.csv")
    request = folder / "request.json"
    request.write_text('{"legacy":true}', encoding="utf-8")
    original = request.read_bytes()
    saved = {"id": job_id, "folder": str(folder), "name": "Old benchmark", "state": "completed",
             "config": {"methods": ["mice-pmm"], "hidden_dims": [12, 6], "seed": 17},
             "parameters": {"mice-pmm": {"m": 3}}}
    monkeypatch.setattr(JobStore, "history", lambda self: [saved])
    monkeypatch.setattr(BenchmarkWindow, "refresh_jobs", lambda self: None)
    monkeypatch.setattr(environment, "inspect_environment", lambda *a, **kw: (states(), False, None))
    window = BenchmarkWindow(tmp_path)
    until(app, lambda: bool(window.backend_states))
    assert window.selected_job is None
    assert window.configuration()["seed"] == 0
    assert window.method_parameters == {}
    assert Path(window.input_path.text()) != folder / "input.csv"
    assert not hasattr(window, "job_combo")
    assert request.read_bytes() == original
    window.close()


def test_environment_failure_rechecks_only_new_jobs_once_per_session(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    import jingwei_benchmark.gui_environment as environment
    monkeypatch.setattr(environment, "inspect_environment", lambda *a, **kw: (states(), False, None))
    window = BenchmarkWindow(tmp_path, load_default=False)
    until(app, lambda: bool(window.backend_states))
    calls = []
    monkeypatch.setattr(window, "refresh_environment", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(window, "render_results", lambda: None)
    monkeypatch.setattr(window, "show_method_log", lambda: None)
    info = {"id": "a" * 32, "name": "Failed job", "created_at": "2026-10-16T00:00:00Z",
            "state": "failed", "expected": 1, "done": 1, "folder": str(tmp_path),
            "runs": [{"status": "failed", "error": "No module named torch"}]}
    monkeypatch.setattr(window.store, "history", lambda: [info])
    monkeypatch.setattr(window.store, "info", lambda job: info)
    window.selected_job = info["id"]
    window.refresh_jobs()
    assert not calls  # Old failures must not invalidate a healthy cached environment.
    window._session_jobs.add(info["id"])
    info["runs"][0]["error"] = "Unsupported input matrix"
    window.refresh_jobs()
    assert not calls
    info["runs"][0]["error"] = "No module named torch"
    window.refresh_jobs()
    window.refresh_jobs()
    assert calls == [{"force": True}]
    window.close()
