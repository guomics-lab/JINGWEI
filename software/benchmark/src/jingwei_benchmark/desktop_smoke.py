"""Artifact-level native UI smoke check used by the build validation script."""

from pathlib import Path
import sys
import time


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase, QFont
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt, QTimer
    from jingwei_benchmark.gui import BenchmarkWindow, set_value
    from jingwei_benchmark.gui_jobs import write_json, TERMINAL
    from .windows_identity import APP_USER_MODEL_ID, set_process_app_id
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_process_app_id()
    app = QApplication.instance() or QApplication(["JINGWEI desktop validation"])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        app.setFont(QFont("Microsoft YaHei UI", 10))
    window = BenchmarkWindow(output / "workspace", load_default=False)
    window.resize(1366, 768)
    errors = []
    window.show_error = errors.append
    window.show()
    native_icons = {}
    app_id = None
    if sys.platform == "win32":
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
    window.load_demo()
    state = {"phase": "loading"}
    started = time.monotonic()
    timer = QTimer()

    def poll():
        if errors or time.monotonic() - started > 300:
            write_json(output / "smoke-result.json", {"status": "failed", "errors": errors or ["timed out"]})
            timer.stop()
            app.quit()
            return
        if state["phase"] == "loading" and window.preview_data and window.backend_states:
            timer.stop()
            window.select_methods(["mean", "knn"])
            set_value(window.fields["repeats"], 1)
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
            state["phase"] = "running"
            timer.start(200)
        elif state["phase"] == "running" and window.selected_job:
            if not state.get("switched_while_running"):
                window.set_language("zh")
                assert window.configuration() == state["configuration"]
                state["switched_while_running"] = True
            info = window.store.info(window.selected_job)
            if info["state"] in TERMINAL:
                timer.stop()
                window.refresh_jobs()
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
                    "native_ui": True, "logo_loaded": not window.logo.isNull(),
                    "application_icon_loaded": not app.windowIcon().isNull(),
                    "windows_native_icons": native_icons,
                    "app_user_model_id": app_id, "qt_platform": app.platformName(),
                    "languages": ["zh", "en"], "language_switch_keeps_configuration": True,
                    "default_timeout_seconds": state["configuration"]["timeout"],
                    "switched_while_running": state.get("switched_while_running"),
                    "methods_available": sum(row["available"] for row in window.backend_states.values()),
                    "rows": window.result_table.rowCount(), "job_folder": info["folder"],
                    "plot_rows": len(window.plot.source), "errors": errors})
                timer.stop()
                app.quit()

    timer.timeout.connect(poll)
    timer.start(200)
    app.exec()
    window.pool.waitForDone()
    from .gui_jobs import read_json
    result = read_json(output / "smoke-result.json")
    return 0 if (result.get("status") == "completed" and result.get("logo_loaded")
                 and result.get("application_icon_loaded") and all(native_icons.values())
                 and result.get("rows") == 2 and not errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
