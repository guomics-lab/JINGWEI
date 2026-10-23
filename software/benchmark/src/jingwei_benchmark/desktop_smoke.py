"""Artifact-level native UI smoke check used by the build validation script."""

from pathlib import Path
import sys
import time


def main():
    from PySide6.QtWidgets import QApplication, QPushButton
    from PySide6.QtGui import QFontDatabase, QFont
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt, QTimer
    from jingwei_benchmark.gui import BenchmarkWindow, set_value
    from jingwei_benchmark.gui_jobs import read_json, write_json, TERMINAL
    from .windows_identity import APP_USER_MODEL_ID, set_process_app_id
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_process_app_id()
    app = QApplication.instance() or QApplication(["JINGWEI desktop validation"])
    if sys.platform == "win32":
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        families = QFontDatabase.applicationFontFamilies(font_id)
        app.setFont(QFont(families[0] if families else "Microsoft YaHei UI", 10))
    reopening = "--reopen" in sys.argv[2:]
    previous = read_json(output / "smoke-result.json") if reopening else None
    started = time.monotonic()
    from .gui_startup import StartupGate
    window = BenchmarkWindow(output / "workspace", load_default=reopening, startup=True)
    initial_language = window.language
    if "--expected-language" in sys.argv:
        assert initial_language == sys.argv[sys.argv.index("--expected-language") + 1]
    window.resize(1366, 768)
    if "--hidden" in sys.argv[2:]:
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
    errors = []
    window.show_error = errors.append
    gate = StartupGate(window, hidden="--hidden" in sys.argv[2:])
    gate.start()
    native_icons = {}
    app_id = None
    if sys.platform == "win32" and app.platformName() == "windows":
        import ctypes
        send = ctypes.WinDLL("user32").SendMessageW
        send.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        send.restype = ctypes.c_ssize_t
        app.processEvents()
        native_icons = {"small": bool(send(int(window.winId()), 0x007F, 0, 0)),
                        "large": bool(send(int(window.winId()), 0x007F, 1, 0))}
        getter = ctypes.WinDLL("shell32").GetCurrentProcessExplicitAppUserModelID
        getter.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        getter.restype = ctypes.c_long
        pointer = ctypes.c_void_p()
        assert getter(ctypes.byref(pointer)) == 0
        try:
            app_id = ctypes.wstring_at(pointer)
            assert app_id == APP_USER_MODEL_ID
        finally:
            free = ctypes.WinDLL("ole32").CoTaskMemFree
            free.argtypes = [ctypes.c_void_p]
            free(pointer)
    if not reopening:
        window.load_demo()
    state = {"phase": "loading"}
    timer = QTimer()

    def step():
        if errors or time.monotonic() - started > 300:
            write_json(output / "smoke-result.json", {"status": "failed", "errors": errors or ["timed out"]})
            timer.stop()
            app.quit()
            return
        if state["phase"] == "loading" and window._environment_checking:
            assert not window.isVisible(), "Workbench shown before environment initialization finished"
        if state["phase"] == "loading" and gate.revealed and window.preview_data and window.backend_states:
            timer.stop()
            state["ready_s"] = time.monotonic() - started
            assert window.selected_job is None and window.run_model.rowCount() == 0
            assert not hasattr(window, "job_combo")
            if reopening:
                assert window._environment_cache_hit and window._environment_probe_count == 0
                assert window.configuration() == previous["configuration"]
                assert window.method_parameters == previous["method_parameters"]
                assert window.language == "en"
                assert window.job_name.text() == "Reopen check"
            else:
                window.output_root.setText(str(output / "chosen output"))
                window.job_name.setText("Reopen check")
                window.method_parameters = {"knn": {"k": 3},
                    "jingwei-dcae": {"encoder_blocks": 1, "decoder_blocks": 1,
                        "mask_predictor_hidden_dim": 8}}
                window.feature_selection = window.preview_data["feature_ids"][:10]
                for key, value in {"training_policy":"fixed", "max_epochs": 2, "latent_dim": 4, "hidden_dims": "8", "batch_size": 4, "threads": 1}.items():
                    set_value(window.fields[key], value)
            window.select_methods(["mean", "knn", "jingwei-dcae"])
            assert "repeats" not in window.fields
            assert not {"evaluation_protocol", "training_mechanism", "training_mask_rate", "seed"} & window.fields.keys()
            assert not hasattr(window, "protocol_summary")
            assert window.run_settings_form.rowCount() == 3
            assert not any(button.text() == window.t("打开任务…") for button in window.findChildren(QPushButton))
            assert window.brand_label.text() == "jingwei"
            assert window.configuration()["repeats"] == 1
            set_value(window.fields["command"], "impute")
            window.mode_changed()
            assert window.fields["mechanism"].isEnabled()
            assert window.fields["mask_rate"].isEnabled()
            assert window.field_labels["mask_rate"].text() == window.t("训练掩蔽比例")
            set_value(window.fields["command"], "impute" if "--impute" in sys.argv else "benchmark")
            window.mode_changed()
            set_value(window.fields["device"], "cpu")
            window.set_language("zh")
            state["configuration"] = window.configuration()
            QTest.qWait(100)
            window.grab().save(str(output / "desktop-input-zh.png"))
            window.set_language("en")
            assert window.configuration() == state["configuration"]
            assert window.start_button.text() == "Run"
            QTest.qWait(100)
            window.grab().save(str(output / "desktop-input-en.png"))
            window.grab().save(str(output / "desktop-input.png"))
            QTest.mouseClick(window.start_button, Qt.LeftButton)
            assert not window.configuration_panel.isEnabled()
            assert not window.clone_button.isEnabled()
            state["phase"] = "running"
            timer.start(200)
        elif state["phase"] == "running" and window.selected_job:
            if not state.get("switched_while_running"):
                window.set_language("zh")
                assert window.configuration() == state["configuration"]
                state["switched_while_running"] = True
            info = window.store.info(window.selected_job)
            if info["state"] not in TERMINAL:
                assert not window.configuration_panel.isEnabled()
                assert not any(window.fields[key].isEnabled() for key in ("max_epochs", "mask_rate", "command"))
                if not state.get("captured_locked"):
                    window.grab().save(str(output / "desktop-running-locked.png"))
                    state["captured_locked"] = True
            if info["state"] in TERMINAL:
                timer.stop()
                window.refresh_jobs()
                assert window.configuration_panel.isEnabled()
                assert window.clone_button.isEnabled()
                assert "Completed" in window.log_view.toPlainText()
                assert Path(info["folder"]).parent == output / "chosen output"
                rows = read_json(Path(info["folder"]) / "results/metrics.json")
                if info["config"]["command"] == "impute":
                    assert window.result_notice.isHidden() and not window.result_notice.text()
                    import numpy as np
                    import pandas as pd
                    original = pd.read_csv(Path(info["folder"]) / "input.csv", index_col=0).to_numpy()
                    for row in rows:
                        csv = Path(info["folder"]) / "results" / row["imputed_csv"]
                        predicted = pd.read_csv(csv, index_col=0).to_numpy()
                        assert np.isfinite(predicted).all()
                        np.testing.assert_allclose(predicted[np.isfinite(original)], original[np.isfinite(original)])
                dcae = next(row for row in rows if row["method"] == "jingwei-dcae")
                assert dcae["ae_architecture"] == "global"
                assert dcae["model_architecture_version"] == "global-ae-v1"
                window.selected_job = None
                window.open_task(info["folder"])
                assert window.selected_job is not None
                window.grab().save(str(output / "desktop-log-zh.png"))
                QTest.qWait(100)
                window.grab().save(str(output / "desktop-results.png"))
                window.grab().save(str(output / "desktop-results-zh.png"))
                window.result_tabs.setCurrentIndex(1)
                QTest.qWait(100)
                window.grab().save(str(output / "desktop-plot-zh.png"))
                window.set_language("en")
                QTest.qWait(100)
                window.grab().save(str(output / "desktop-plot-en.png"))
                window.result_tabs.setCurrentIndex(0)
                QTest.qWait(100)
                window.grab().save(str(output / "desktop-results-en.png"))
                write_json(output / "smoke-result.json", {"status": info["state"],
                    "native_ui": app.platformName() == "windows", "logo_loaded": not window.logo.isNull(),
                    "initial_language": initial_language,
                    "imputation_csv_files": len(rows) if info["config"]["command"] == "impute" else 0,
                    "startup_without_task": True, "custom_output": str(output / "chosen output"),
                    "live_log": bool(window.log_view.toPlainText()),
                    "manual_task_open": True, "repeats_control_removed": "repeats" not in window.fields,
                    "configuration_locked_while_running": state.get("captured_locked", False),
                    "configuration_unlocked_after_completion": window.configuration_panel.isEnabled(),
                    "open_task_button_removed": True,
                    "impute_training_mask_controls": True,
                    "run_settings_rows": window.run_settings_form.rowCount(),
                    "startup_environment_ready_before_main": gate.ready_before_show,
                    "application_name": app.applicationName(),
                    "dcae_architecture_version": dcae["model_architecture_version"],
                    "application_icon_loaded": not app.windowIcon().isNull(),
                    "windows_native_icons": native_icons,
                    "app_user_model_id": app_id, "qt_platform": app.platformName(),
                    "languages": ["zh", "en"], "language_switch_keeps_configuration": True,
                    "default_timeout_seconds": state["configuration"]["timeout"],
                    "switched_while_running": state.get("switched_while_running"),
                    "methods_available": sum(row["available"] for row in window.backend_states.values()),
                    "rows": window.result_table.rowCount(), "job_folder": info["folder"],
                    "plot_rows": len(window.plot.source), "errors": errors})
                result = read_json(output / "smoke-result.json")
                result.update(environment_cache_hit=window._environment_cache_hit,
                              environment_probe_count=window._environment_probe_count,
                              ready_s=state["ready_s"], configuration=state["configuration"],
                              method_parameters=window.method_parameters, reopened=reopening)
                write_json(output / "smoke-result.json", result)
                write_json(output / ("reopen-result.json" if reopening else "cold-result.json"), result)
                timer.stop()
                window.close()
                app.quit()

    def poll():
        try:
            step()
        except Exception:
            import traceback
            errors.append(traceback.format_exc())
            write_json(output / "smoke-result.json", {"status": "failed", "errors": errors})
            timer.stop()
            for job in window.store.active():
                window.store.cancel(job["id"], force=True)
            app.quit()

    timer.timeout.connect(poll)
    timer.start(200)
    app.exec()
    window.pool.waitForDone()
    from .gui_jobs import read_json
    result = read_json(output / "smoke-result.json")
    return 0 if (result.get("status") == "completed" and result.get("logo_loaded")
                 and result.get("application_icon_loaded") and all(native_icons.values())
                 and result.get("rows") == 3 and not errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
