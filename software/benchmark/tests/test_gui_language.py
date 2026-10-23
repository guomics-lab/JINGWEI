"""Installer selection must apply before the first splash/window is shown."""
import json
from pathlib import Path

import pytest

from jingwei_benchmark.gui_language import initial_language, remember_language


@pytest.mark.parametrize("installer,expected", [("english", "en"), ("chinesesimp", "zh")])
def test_install_language_overrides_old_preferences_once(tmp_path, monkeypatch, installer, expected):
    import jingwei_benchmark.runtime as runtime
    install = tmp_path / "installed"
    install.mkdir()
    monkeypatch.setattr(runtime, "resource_root", lambda: install)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ui.json").write_text(json.dumps({"language": "zh" if expected == "en" else "en"}))
    marker = install / "installation-language.txt"
    marker.write_text(installer + "\n1.0.8\nfirst installation\n", encoding="utf-8")
    assert initial_language(workspace) == expected
    chosen = "zh" if expected == "en" else "en"
    remember_language(workspace, chosen)
    assert initial_language(workspace) == chosen
    # Reinstallation must apply the newly selected language, even for the
    # same version/destination and an existing Chinese/English user profile.
    marker.write_text(installer + "\n1.0.8\nreinstallation\n", encoding="utf-8")
    assert initial_language(workspace) == expected


def test_portable_and_invalid_install_marker_keep_saved_choice(tmp_path, monkeypatch):
    import jingwei_benchmark.runtime as runtime
    monkeypatch.setattr(runtime, "resource_root", lambda: tmp_path)
    remember_language(tmp_path, "en")
    assert initial_language(tmp_path) == "en"
    marker = tmp_path / "installation-language.txt"
    for value in (b"", b"unknown\n", b"\xff"):
        marker.write_bytes(value)
        assert initial_language(tmp_path) == "en"


def test_frozen_installation_marker_is_beside_executable(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "jingwei.exe"))
    (tmp_path / "installation-language.txt").write_text("english\n", encoding="utf-8")
    assert initial_language(tmp_path) == "en"
