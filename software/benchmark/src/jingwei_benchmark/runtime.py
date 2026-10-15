"""Resource and child-process paths in source installs and desktop bundles."""

import os
from pathlib import Path
import sys


def resource_root():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    if os.environ.get("JINGWEI_BENCHMARK_RESOURCES"):
        return Path(os.environ["JINGWEI_BENCHMARK_RESOURCES"])
    return Path(__file__).resolve().parents[4]


def default_input_path():
    """Resolve the bundled default independently of the shell's current folder."""
    path = resource_root() / "data" / "Alzheimer.csv"
    return path if path.is_file() else None


def python_command(module, *args):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--jingwei-internal", module, *map(str, args)]
    return [sys.executable, "-m", "jingwei_benchmark." + module, *map(str, args)]


def user_workspace():
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return root / "JINGWEI/Benchmark"


def prepare_standard_streams():
    """pythonw and windowed frozen processes have no default Python streams."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            path = Path(os.environ.get("JINGWEI_PROCESS_LOG", user_workspace() / "desktop-startup.log"))
            path.parent.mkdir(parents=True, exist_ok=True)
            setattr(sys, name, path.open("a", encoding="utf-8", buffering=1))
