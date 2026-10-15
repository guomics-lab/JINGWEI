import ctypes
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget
from jingwei_benchmark.windows_identity import (
    APP_USER_MODEL_ID, set_process_app_id, application_icon, apply_application_icon,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows desktop identity")


def test_process_app_id_roundtrip():
    assert set_process_app_id()
    shell = ctypes.WinDLL("shell32")
    getter = shell.GetCurrentProcessExplicitAppUserModelID
    getter.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    getter.restype = ctypes.c_long
    pointer = ctypes.c_void_p()
    assert getter(ctypes.byref(pointer)) == 0
    try:
        assert ctypes.wstring_at(pointer) == APP_USER_MODEL_ID
    finally:
        free = ctypes.WinDLL("ole32").CoTaskMemFree
        free.argtypes = [ctypes.c_void_p]
        free(pointer)


def test_original_bird_icon_used_for_application_and_window():
    app = QApplication.instance() or QApplication([])
    icon = application_icon()
    assert not icon.isNull()
    sizes = {(size.width(), size.height()) for size in icon.availableSizes()}
    assert {(16, 16), (32, 32), (48, 48), (256, 256)} <= sizes
    window = QWidget()
    apply_application_icon(app, window)
    assert app.applicationName() == "JINGWEI Benchmark"
    assert not app.windowIcon().isNull()
    assert not window.windowIcon().isNull()
    assert app.windowIcon().pixmap(32, 32).toImage() == window.windowIcon().pixmap(32, 32).toImage()
    # The asset has colored content; a blank/default Qt icon does not pass.
    image = window.windowIcon().pixmap(64, 64).toImage()
    assert len({image.pixelColor(x, y).rgba() for x in range(0, 64, 4) for y in range(0, 64, 4)}) > 20
    window.close()
