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
    for key, setting in {"device": "cpu", "repeats": 3, "mask_rate": .15, "seed": 91, "hidden_dims": "32, 16"}.items():
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
    assert window.fields["repeats"].value() == 5
    window.close()


def test_first_upgrade_restores_legacy_job_without_changing_its_records(tmp_path, monkeypatch):
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
    until(app, lambda: window.preview_data is not None and bool(window.backend_states))
    assert value(window.fields["mice_backend"]) == "r"
    assert window.configuration()["methods"] == ["mice-pmm"]
    assert window.configuration()["hidden_dims"] == [12, 6]
    assert window.configuration()["seed"] == 17
    assert window.method_parameters == {"mice-pmm": {"m": 3}}
    assert Path(window.input_path.text()) == folder / "input.csv"
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
            "state": "failed", "expected": 1, "done": 1,
            "runs": [{"status": "failed", "error": "No module named torch"}]}
    monkeypatch.setattr(window.store, "history", lambda: [info])
    monkeypatch.setattr(window.store, "info", lambda job: info)
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
