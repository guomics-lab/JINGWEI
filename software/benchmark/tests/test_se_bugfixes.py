"""Regression tests for GUI operability, path normalization, plot switching, and CLI/R checks."""

import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QLineEdit, QListWidget, QPushButton

from jingwei_benchmark.cli import main
from jingwei_benchmark.gui import BenchmarkWindow, clean_path_text, set_value
from jingwei_benchmark.gui_jobs import TERMINAL
from jingwei_benchmark.methods import METHODS, find_rscript, r_environment


def until(app, predicate, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(20)
    pytest.fail("condition timed out")


@pytest.fixture
def gui_env(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    states = {
        name: {"available": True, "backend": spec.backend, "gpu": spec.gpu, "reason": None}
        for name, spec in METHODS.items()
    }
    monkeypatch.setattr(
        BenchmarkWindow,
        "refresh_environment",
        lambda self, **kw: self.apply_environment((states, False, None)),
    )
    win = BenchmarkWindow(tmp_path / "workspace", load_default=False)
    errors = []
    win.show_error = errors.append
    source = tmp_path / "good data.csv"
    pd.DataFrame(
        np.random.default_rng(11).lognormal(4.0, 0.4, (10, 6)),
        index=[f"S{i}" for i in range(10)],
        columns=[f"P{i}" for i in range(6)],
    ).to_csv(source)
    set_value(win.fields["device"], "cpu")
    yield app, win, source, errors
    for info in win.store.active():
        win.store.cancel(info["id"], force=True)
    win.close()
    win.pool.waitForDone(10000)
    app.processEvents()


def test_quoted_paths_work_across_input_and_output_and_missing_path_updates_summary(gui_env, tmp_path):
    app, win, source, errors = gui_env
    assert clean_path_text(f'  "{source}"  ') == str(source)
    assert clean_path_text(f" '{source}' ") == str(source)

    # Pasting Windows "Copy as path" with surrounding double quotes auto-strips and inspects.
    win.input_path.setText(f'"{source}"')
    win.on_input_editing_finished()
    until(app, lambda: win.preview_data is not None)
    assert win.input_path.text() == str(source)
    assert win.start_button.isEnabled()

    # Quoted output_root also submits and completes cleanly.
    quoted_out = tmp_path / "quoted output"
    win.output_root.setText(f'"{quoted_out}"')
    win.select_methods(["mean"])
    win.start_job()
    until(app, lambda: win.selected_job is not None)
    until(app, lambda: win.store.info(win.selected_job)["state"] in TERMINAL)
    info = win.store.info(win.selected_job)
    assert info["state"] == "completed"
    assert Path(info["folder"]).parent == quoted_out.resolve()
    assert not errors

    # Typing a nonexistent path updates inline summary without popping modal errors on focus-out.
    win.input_path.setText(str(tmp_path / "does_not_exist.csv"))
    win.on_input_editing_finished()
    assert not errors
    assert win.data_summary.text() == "文件不存在，请检查路径或重新选择。"
    win.set_language("en")
    assert win.data_summary.text() == "File not found. Check the path or browse again."


def test_recheck_action_return_key_and_advanced_close_recover_broken_or_edited_csv(gui_env, tmp_path, monkeypatch):
    app, win, source, errors = gui_env

    # No extra QPushButton added to centralWidget (recheck is an inline QAction).
    main_buttons = {b.text() for b in win.centralWidget().findChildren(QPushButton)}
    assert not {"检查", "Check"} & main_buttons

    broken = tmp_path / "broken.csv"
    broken.write_text("", encoding="utf-8")
    win.set_input(broken)
    until(app, lambda: not win._inspecting and bool(errors))
    assert "CSV is empty" in win.data_summary.text()
    win.set_language("en")
    assert "Data check failed: CSV is empty" in win.data_summary.text()
    errors.clear()

    # Repair CSV on disk without changing input_path text, then press Return in input_path.
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["A", "B"], columns=["X", "Y"]).to_csv(broken)
    QTest.keyClick(win.input_path, Qt.Key_Return)
    until(app, lambda: win.preview_data is not None)
    assert win.preview_data["samples"] == 2 and win.start_button.isEnabled()

    # Modify CSV externally again; inline recheck_action refreshes preview_signature.
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=["A", "B", "C"], columns=["X", "Y"]).to_csv(broken)
    assert win.signature() != win.preview_signature
    win.recheck_action.trigger()
    until(app, lambda: not win._inspecting and win.preview_data is not None and win.preview_data["samples"] == 3)
    assert win.signature() == win.preview_signature

    # Modify CSV externally a third time; closing Advanced dialog also auto-refreshes signature.
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], index=["A", "B", "C", "D"], columns=["X", "Y"]).to_csv(broken)
    monkeypatch.setattr(win.advanced_dialog, "exec", lambda: QDialog.Rejected)
    win.show_advanced()
    until(app, lambda: not win._inspecting and win.preview_data["samples"] == 4)
    assert not errors


def test_method_tree_parameter_button_sync_and_group_expansion_on_clone_and_restore(gui_env, tmp_path):
    app, win, source, errors = gui_env

    # Startup highlights jingwei-dmf so Parameters... is immediately operable.
    assert win.method_tree.currentItem() is win.method_checks["jingwei-dmf"].item
    assert win.parameter_button.isEnabled()

    # Checking a configurable baseline method moves selection to it and keeps Parameters... enabled.
    win.select_methods([])
    win.method_checks["knn"].setChecked(True)
    app.processEvents()
    assert win.method_tree.currentItem() is win.method_checks["knn"].item
    assert win.parameter_button.isEnabled()

    # Checking a parameterless method disables Parameters..., checking perseus re-enables it.
    win.method_checks["mean"].setChecked(True)
    app.processEvents()
    assert win.method_tree.currentItem() is win.method_checks["mean"].item
    assert not win.parameter_button.isEnabled()
    win.method_checks["perseus"].setChecked(True)
    app.processEvents()
    assert win.parameter_button.isEnabled()

    # Cloning a job whose methods live in collapsed groups expands those groups automatically.
    win.method_groups["python"].setExpanded(False)
    win.method_groups["r"].setExpanded(False)
    cli_out = tmp_path / "cli_run"
    assert main(["benchmark", "--input", str(source), "--methods", "knn", "--device", "cpu", "--output", str(cli_out)]) == 0
    win.open_task(cli_out)
    win.clone_job()
    until(app, lambda: win.preview_data is not None)
    assert win.method_groups["python"].isExpanded()
    assert win.method_tree.currentItem() is win.method_checks["knn"].item
    assert win.parameter_button.isEnabled()


def test_export_job_disabled_while_running_and_missing_extensions_appended(gui_env, tmp_path, monkeypatch):
    app, win, source, errors = gui_env
    win.set_input(source)
    until(app, lambda: win.preview_data is not None)

    win.selected_job = "a" * 32
    state_holder = {
        "id": "a" * 32,
        "name": "live-job",
        "folder": str(tmp_path),
        "state": "running",
        "done": 0,
        "expected": 1,
        "runs": [],
        "config": {"command": "benchmark", "methods": ["mean"], "repeats": 1},
        "manifest": {},
    }
    monkeypatch.setattr(win.store, "info", lambda jid: state_holder)
    win.refresh_jobs()
    assert not win.export_job_button.isEnabled()

    # Direct call while running fails fast before opening QFileDialog.
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: pytest.fail("Save dialog opened while running"))
    win.export_job()
    assert errors and ("结束" in errors[-1] or "finish" in errors[-1].lower())
    errors.clear()

    # Once completed, export_job_button enables and auto-appends .zip / .pdf when omitted.
    job_dir = tmp_path / "completed_job"
    job_dir.mkdir()
    (job_dir / "marker.txt").write_text("ok", encoding="utf-8")
    state_holder.update(
        state="completed",
        done=1,
        folder=str(job_dir),
        runs=[{"repeat": 0, "method": "mean", "status": "completed", "MSE": 0.25, "time_s": 0.1}],
    )
    monkeypatch.setattr(win.store, "_folder", lambda jid: job_dir)
    win._last_result_digest = None
    win.refresh_jobs()
    assert win.export_job_button.isEnabled()

    zip_no_ext = tmp_path / "bundle_without_ext"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(zip_no_ext), "ZIP (*.zip)"))
    win.export_job()
    until(app, lambda: (tmp_path / "bundle_without_ext.zip").is_file() and not win._tasks)

    pdf_no_ext = tmp_path / "figure_without_ext"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **kw: (str(pdf_no_ext), "PDF (*.pdf)"))
    win.export_plot()
    assert (tmp_path / "figure_without_ext.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "figure_without_ext.source.csv").is_file()
    assert (tmp_path / "figure_without_ext.provenance.json").is_file()
    assert not errors


def test_impute_plot_auto_switches_to_runtime_and_restores_benchmark_metric(gui_env, tmp_path, monkeypatch):
    app, win, source, errors = gui_env
    from jingwei_benchmark.gui_jobs import read_json

    assert win.metric_choice.currentData() == "MSE"
    win.selected_job = "b" * 32
    impute_info = {
        "id": "b" * 32,
        "name": "impute-run",
        "folder": str(tmp_path),
        "state": "completed",
        "done": 2,
        "expected": 2,
        "runs": [
            {"repeat": 0, "method": "mean", "status": "completed", "time_s": 0.12},
            {"repeat": 0, "method": "knn", "status": "completed", "time_s": 0.34},
        ],
        "config": {"command": "impute", "methods": ["mean", "knn"], "repeats": 1},
        "manifest": {},
    }
    monkeypatch.setattr(win.store, "info", lambda jid: impute_info)
    win.refresh_jobs()
    assert win.metric_choice.currentData() == "time_s"
    assert len(win.plot.source) == 2
    assert not win.metric_choice.model().item(win.metric_choice.findData("MSE")).isEnabled()
    assert win.metric_choice.model().item(win.metric_choice.findData("time_s")).isEnabled()

    # Session persistence preserves user's benchmark metric ("MSE") rather than auto-switched time_s.
    win.save_session()
    assert read_json(win.store.root / "session.json")["metric"] == "MSE"

    # Switching to a benchmark job restores MSE and re-enables all metrics; constant predictions show correlation tooltip.
    bench_info = {
        "id": "c" * 32,
        "name": "bench-run",
        "folder": str(tmp_path),
        "state": "completed",
        "done": 1,
        "expected": 1,
        "runs": [{"repeat": 0, "method": "minimum", "status": "completed", "MSE": 0.5, "Pearson": None, "W2": None, "time_s": 0.1}],
        "config": {"command": "benchmark", "methods": ["minimum"], "repeats": 1},
        "manifest": {},
    }
    win.selected_job = "c" * 32
    monkeypatch.setattr(win.store, "info", lambda jid: bench_info)
    win.refresh_jobs()
    assert win.metric_choice.currentData() == "MSE"
    assert win.metric_choice.model().item(win.metric_choice.findData("MSE")).isEnabled()

    win.metric_choice.setCurrentIndex(win.metric_choice.findData("Pearson"))
    assert "W2" not in win.plot.toolTip()
    win.metric_choice.setCurrentIndex(win.metric_choice.findData("W2"))
    assert "W2" in win.plot.toolTip()


def test_w2_feature_dialog_keeps_open_on_empty_selection_and_ignores_enter_in_search(gui_env):
    app, win, source, errors = gui_env
    win.set_input(source)
    until(app, lambda: win.preview_data is not None)

    def interact():
        dlg = QApplication.activeModalWidget()
        assert isinstance(dlg, QDialog)
        search = dlg.findChild(QLineEdit)
        listing = dlg.findChild(QListWidget)
        buttons = dlg.findChild(QDialogButtonBox)
        # Pressing Enter in search box must NOT accept & close dialog prematurely.
        search.setFocus()
        search.setText("P0")
        QTest.keyClick(search, Qt.Key_Return)
        app.processEvents()
        assert dlg.isVisible()
        search.clear()
        app.processEvents()
        # Uncheck all items and click OK -> shows error while keeping dialog open!
        for i in range(listing.count()):
            listing.item(i).setCheckState(Qt.Unchecked)
        buttons.button(QDialogButtonBox.Ok).click()
        app.processEvents()
        assert errors and dlg.isVisible()
        errors.clear()
        # Now check P0 and P2 and click OK -> accepts cleanly!
        listing.item(0).setCheckState(Qt.Checked)
        listing.item(2).setCheckState(Qt.Checked)
        buttons.button(QDialogButtonBox.Ok).click()

    QTimer.singleShot(30, interact)
    win.choose_features()
    assert win.feature_selection == ["P0", "P2"]
    assert not errors


def test_cli_bounds_empty_available_nrmse_and_fast_rscript_resolution(tmp_path):
    source = tmp_path / "small.csv"
    pd.DataFrame([[10.0, 20.0], [30.0, 40.0]], index=["S1", "S2"], columns=["P1", "P2"]).to_csv(source)

    for bad_flag in (
        ["--dmf-rank", "0"],
        ["--learning-rate", "0"],
        ["--learning-rate", "nan"],
        ["--mask-weight", "-0.5"],
        ["--mask-weight", "inf"],
    ):
        assert main(["benchmark", "--input", str(source), "--methods", "mean", *bad_flag, "--output", str(tmp_path / "bad")]) == 1

    # Extreme masking leaving 0 available observations writes null NRMSE instead of crashing JSON allow_nan=False.
    out_all_masked = tmp_path / "all_masked"
    assert (
        main(
            [
                "benchmark",
                "--input",
                str(source),
                "--methods",
                "zero",
                "--mechanism",
                "mcar",
                "--mask-rate",
                "1.0",
                "--allow-empty-marginals",
                "--device",
                "cpu",
                "--output",
                str(out_all_masked),
            ]
        )
        == 0
    )
    record = json.loads((out_all_masked / "metrics.json").read_text(encoding="utf-8"))[0]
    assert record["status"] == "completed" and record["NRMSE"] is None

    # find_rscript accepts quoted paths, R root directories, and R bin directories.
    fake_r_root = tmp_path / "R-4.4.1"
    fake_bin = fake_r_root / "bin"
    fake_bin.mkdir(parents=True)
    exe_name = "Rscript.exe" if os.name == "nt" else "Rscript"
    fake_exe = fake_bin / exe_name
    fake_exe.write_bytes(b"stub")
    assert Path(find_rscript(f'"{fake_exe}"')) == fake_exe.resolve()
    assert Path(find_rscript(str(fake_bin))) == fake_exe.resolve()
    assert Path(find_rscript(str(fake_r_root))) == fake_exe.resolve()

    real_rscript = find_rscript()
    if real_rscript:
        bridge = Path(__file__).resolve().parents[1] / "src/jingwei_benchmark/r_backend.R"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [real_rscript, "--vanilla", str(bridge), "--check", "jsonlite", "nonexistent_pkg_xyz"],
            env=r_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        assert proc.returncode == 0 and elapsed < 15
        parsed = dict(line.split("\t")[:2] for line in proc.stdout.strip().splitlines())
        assert parsed["jsonlite"] == "TRUE"
        assert parsed["nonexistent_pkg_xyz"] == "FALSE"
