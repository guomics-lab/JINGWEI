import os
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from jingwei_benchmark.gui import BenchmarkWindow, set_value

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only GUI")


@pytest.fixture(scope="module")
def app():
    from jingwei_benchmark.windows_identity import set_process_app_id
    set_process_app_id()
    application = QApplication.instance() or QApplication([])
    if os.name == "nt":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        application.setFont(QFont("Microsoft YaHei UI", 10))
    yield application


def until(app, predicate, timeout=45000):
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(30)
    pytest.fail("native GUI condition timed out")


def test_clone_legacy_mice_preserves_r_while_new_job_keeps_python(app, tmp_path, monkeypatch):
    from jingwei_benchmark.gui import value
    monkeypatch.setattr(BenchmarkWindow, "refresh_environment", lambda self: None)
    window = BenchmarkWindow(tmp_path / "desktop", load_default=False)
    assert value(window.fields["mice_backend"]) == "python"
    saved = {"config": {"methods": ["mice-pmm"]}, "parameters": {},
             "name": "legacy", "folder": str(tmp_path)}
    monkeypatch.setattr(window.store, "info", lambda job_id: saved)
    window.selected_job = "a" * 32
    window.clone_job()
    assert value(window.fields["mice_backend"]) == "r"
    saved["config"]["mice_backend"] = "python"
    window.clone_job()
    assert value(window.fields["mice_backend"]) == "python"
    window.close()


def test_native_window_load_run_plot_and_restore(app, tmp_path):
    window = BenchmarkWindow(tmp_path / "desktop", load_default=False)
    window.resize(1366, 768)
    window.show()
    errors = []
    window.show_error = errors.append
    window.load_demo()
    until(app, lambda: window.preview_data is not None and bool(window.backend_states))
    assert not window.logo.isNull()
    assert window.preview_data["samples"] == 16
    assert window.preview_data["features"] == 80
    assert window.preview_table.rowCount() == 12
    assert window.configuration()["methods"] == ["jingwei-dmf", "jingwei-dcae"]
    window.select_methods(["mean", "knn"])
    set_value(window.fields["repeats"], 1)
    set_value(window.fields["device"], "cpu")
    window.method_parameters = {"knn": {"k": 3}}
    window.feature_selection = window.preview_data["feature_ids"][:10]
    original_config = window.configuration()
    original_preview = window.preview_data
    # Native language control updates the existing widgets without resetting data.
    window.language_choice.setCurrentIndex(window.language_choice.findData("en"))
    assert window.start_button.text() == "Run"
    assert window.result_tabs.tabText(1) == "Plot"
    assert window.configuration() == original_config
    assert window.preview_data is original_preview
    assert window.method_parameters == {"knn": {"k": 3}}
    assert window.start_button.isVisible()
    assert window.rect().contains(window.start_button.mapTo(window, window.start_button.rect().bottomRight()))
    QTest.qWait(50)
    QTest.mouseClick(window.start_button, Qt.LeftButton)
    until(app, lambda: window.selected_job is not None)
    job_id = window.selected_job
    assert not window.start_button.isEnabled()
    window.language_choice.setCurrentIndex(window.language_choice.findData("zh"))
    assert window.start_button.text() == "开始"
    assert window.selected_job == job_id
    assert window.configuration() == original_config
    QTest.mouseClick(window.start_button, Qt.LeftButton)
    assert len(window.store.history()) == 1
    until(app, lambda: window.store.info(window.selected_job)["state"] in ("completed", "failed", "incomplete"))
    window.refresh_jobs()
    assert not errors
    assert window.store.info(window.selected_job)["state"] == "completed"
    assert window.result_table.rowCount() == 2
    assert len(window.plot.source) == 2
    assert window.plot.provenance["paired_complete_repeats_only"]
    assert len(window.plot.figure.axes[0].patches) == 2
    source = window.plot.source.copy()
    window.language_choice.setCurrentIndex(window.language_choice.findData("en"))
    pd.testing.assert_frame_equal(source, window.plot.source)
    assert window.plot.figure.axes[0].get_title(loc="left") == "Paired complete repeats"
    assert window.run_model.headerData(1, Qt.Horizontal) == "Method"
    original = window.store.info(window.selected_job)
    window.clone_job()
    until(app, lambda: window.preview_data is not None)
    assert window.configuration()["methods"] == original["config"]["methods"]
    assert window.configuration()["seed"] == original["config"]["seed"]
    window.pool.waitForDone()
    app.processEvents()
    window.close()


def test_language_persists_and_english_catalog_covers_controls(app, tmp_path, monkeypatch):
    from jingwei_benchmark.gui_i18n import MESSAGES
    from jingwei_benchmark.gui_parameters import PARAMETERS
    monkeypatch.setattr(BenchmarkWindow, "refresh_environment", lambda self: None)
    window = BenchmarkWindow(tmp_path, load_default=False)
    window.set_language("en")
    for _, _, label in window._translations:
        if any("\u4e00" <= char <= "\u9fff" for char in label):
            assert label in MESSAGES, label
    for _, items in window._combo_translations:
        for label, _ in items:
            if any("\u4e00" <= char <= "\u9fff" for char in label):
                assert label in MESSAGES, label
    for specs in PARAMETERS.values():
        for label, *_ in specs.values():
            assert label in MESSAGES
    assert window.language_choice.currentData() == "en"
    window.close()
    reopened = BenchmarkWindow(tmp_path, load_default=False)
    assert reopened.language == "en"
    assert reopened.start_button.text() == "Run"
    reopened.close()


def test_gui_starts_with_alzheimer_data(app, tmp_path, monkeypatch):
    from jingwei_benchmark.runtime import default_input_path
    monkeypatch.setattr(BenchmarkWindow, "refresh_environment", lambda self: None)
    window = BenchmarkWindow(tmp_path)
    until(app, lambda: window.preview_data is not None)
    assert Path(window.input_path.text()) == default_input_path()
    assert (window.preview_data["samples"], window.preview_data["features"]) == (210, 1542)
    assert window.preview_data["working_scale"] == "log2"
    window.set_language("en")
    assert Path(window.input_path.text()) == default_input_path()
    window.pool.waitForDone()
    app.processEvents()
    window.close()


def test_macos_cannot_launch_gui(app, tmp_path, monkeypatch):
    from jingwei_benchmark.gui import launch
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Windows-only"):
        launch(tmp_path)
