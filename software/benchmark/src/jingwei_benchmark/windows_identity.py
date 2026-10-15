"""One Windows desktop identity for the process, Qt application and windows."""

import ctypes
from pathlib import Path
import sys

APP_USER_MODEL_ID = "JINGWEI.Benchmark.Desktop"


def set_process_app_id():
    """Set before QApplication / the first HWND so Explorer groups the app correctly."""
    if sys.platform != "win32":
        return False
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    setter = shell.SetCurrentProcessExplicitAppUserModelID
    setter.argtypes = [ctypes.c_wchar_p]
    setter.restype = ctypes.c_long
    result = setter(APP_USER_MODEL_ID)
    if result < 0:
        raise OSError(f"SetCurrentProcessExplicitAppUserModelID failed: 0x{result & 0xffffffff:08x}")
    return True


def application_icon():
    """Use the unmodified JINGWEI bird at the sizes Windows requests."""
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QIcon, QImageReader, QPixmap
    reader = QImageReader(str(Path(__file__).with_name("assets") / "Jingwei_logo.png"))
    reader.setScaledSize(QSize(256, 256))
    # PNG readers may still allocate the original 8334 × 8334 image internally.
    # This is a trusted, bundled asset; restore Qt's process-wide limit afterwards.
    limit = QImageReader.allocationLimit()
    try:
        QImageReader.setAllocationLimit(512)
        image = reader.read()
    finally:
        QImageReader.setAllocationLimit(limit)
    if image.isNull():
        raise RuntimeError(f"Cannot load JINGWEI icon: {reader.errorString()}")
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
    return icon


def apply_application_icon(app, window=None):
    icon = application_icon()
    from . import __version__
    app.setApplicationName("JINGWEI Benchmark")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("JINGWEI")
    app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)
    return icon
