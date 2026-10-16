import json
from pathlib import Path

from jingwei_benchmark import gui_cache as cache
from jingwei_benchmark.methods import METHODS


def states():
    return {name: {"available": True, "backend": method.backend, "gpu": method.gpu, "reason": None}
            for name, method in METHODS.items()}


def test_cache_roundtrip_rejects_stale_and_partial_results(tmp_path):
    path = tmp_path / "environment.json"
    result = states(), True, "Example GPU"
    cache.save_environment(path, "build-one", result)
    assert cache.load_environment(path, "build-one") == result
    assert cache.load_environment(path, "build-two") is None
    data = json.loads(path.read_text())
    data["states"].pop(next(iter(METHODS)))
    path.write_text(json.dumps(data))
    assert cache.load_environment(path, "build-one") is None
    path.write_text("[1,2,3]")
    assert cache.load_environment(path, "build-one") is None
    path.write_text("incomplete JSON")
    assert cache.load_environment(path, "build-one") is None


def test_identity_changes_for_backend_runtime_and_package_updates(tmp_path, monkeypatch):
    rscript = tmp_path / "Rscript"
    rscript.write_text("r-one")
    library = tmp_path / "library"
    library.mkdir()
    monkeypatch.setattr(cache, "find_rscript", lambda *a: str(rscript))
    monkeypatch.setattr(cache, "r_environment", lambda *a: {"R_LIBS_USER": str(library)})
    monkeypatch.setattr(cache, "find_method_source", lambda *a: None)
    options = {"mice_backend": "python"}
    initial = cache.environment_identity(options)
    assert cache.environment_identity(options) == initial
    assert cache.environment_identity({"mice_backend": "r"}) != initial
    rscript.write_text("r-two-new-build")
    changed = cache.environment_identity(options)
    assert changed != initial
    description = library / "mice/DESCRIPTION"
    description.parent.mkdir()
    description.write_text("Version: new")
    assert cache.environment_identity(options) != changed


def test_frozen_rscript_prefers_bundled_runtime(tmp_path, monkeypatch):
    import sys
    import jingwei_benchmark.methods as methods
    bundled = tmp_path / ".r-runtime/bin/Rscript.exe"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled")
    host = tmp_path / "host-Rscript.exe"
    host.write_bytes(b"host")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(methods, "resource_root", lambda: tmp_path)
    monkeypatch.setenv("JINGWEI_RSCRIPT", str(host))
    monkeypatch.setattr(methods.shutil, "which", lambda _: str(host))
    assert Path(methods.find_rscript()) == bundled
    assert Path(methods.find_rscript(str(host))) == host
