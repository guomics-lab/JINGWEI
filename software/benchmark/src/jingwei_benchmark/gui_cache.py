"""Cheap identities and durable results for the desktop's expensive backend probe."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import time

from . import __version__
from .gui_jobs import read_json, write_json
from .methods import METHODS, find_method_source, find_rscript, r_environment
from .runtime import resource_root

CACHE_SCHEMA = 1


def file_identity(path):
    if not path:
        return None
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path), None, None


def environment_identity(options):
    """No imports of torch/R and no subprocesses on a warm desktop launch."""
    root = resource_root()
    rscript = find_rscript(options.get("rscript"))
    library = r_environment(options.get("r_library")).get("R_LIBS_USER")
    packages = {"jsonlite", "DreamAI", *(p for m in METHODS.values() if m.backend == "r" for p in m.packages)}
    inputs = {
        "schema": CACHE_SCHEMA, "version": __version__,
        "executable": file_identity(sys.executable),
        "platform": [sys.platform, platform.machine(), platform.version()],
        "options": options, "rscript": file_identity(rscript),
        "library": file_identity(library),
        "r_packages": {p: file_identity(Path(library) / p / "DESCRIPTION") for p in sorted(packages)} if library else {},
        "sources": {name: file_identity(options.get(key) or find_method_source(name))
                    for name, key in (("NAguideR", "naguider_source"), ("DreamAI", "dreamai_source"))},
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "JINGWEI_RSCRIPT", "JINGWEI_R_LIBRARY")},
    }
    if sys.platform == "win32":
        inputs["driver"] = file_identity(Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvcuda.dll")
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\VIDEO") as key:
                inputs["video"] = [winreg.EnumValue(key, i)[:2] for i in range(winreg.QueryInfoKey(key)[1])]
        except OSError:
            inputs["video"] = None
    if not getattr(sys, "frozen", False):
        inputs["source"] = [file_identity(Path(__file__).with_name(name)) for name in
                            ("gui_cache.py", "gui_environment.py", "methods.py", "r_backend.R")]
        inputs["python_packages"] = {}
        for name in ("torch", "pytorch-lightning", "pimms-learn", "numpy", "scipy", "scikit-learn", "anndata"):
            try:
                inputs["python_packages"][name] = version(name)
            except PackageNotFoundError:
                inputs["python_packages"][name] = None
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_environment(path, identity):
    data = read_json(path)
    if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA or data.get("identity") != identity:
        return None
    states, gpu, gpu_name = data.get("states"), data.get("gpu"), data.get("gpu_name")
    if not isinstance(states, dict) or set(states) != set(METHODS) or not isinstance(gpu, bool):
        return None
    if gpu_name is not None and not isinstance(gpu_name, str):
        return None
    for state in states.values():
        if (not isinstance(state, dict) or not isinstance(state.get("available"), bool)
                or not isinstance(state.get("backend"), str)
                or not isinstance(state.get("gpu"), bool)
                or state.get("reason") is not None and not isinstance(state.get("reason"), str)):
            return None
    return states, gpu, gpu_name


def save_environment(path, identity, result):
    states, gpu, gpu_name = result
    write_json(path, {"schema": CACHE_SCHEMA, "identity": identity, "checked_at": time.time(),
                      "states": states, "gpu": gpu, "gpu_name": gpu_name})
