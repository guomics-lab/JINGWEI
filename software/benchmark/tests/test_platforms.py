import argparse
import sys
from pathlib import Path
import pytest
from jingwei_benchmark import platforms
from jingwei_benchmark.cli import parser
from jingwei_benchmark.methods import r_environment, find_rscript, availability


def test_mac_cli_has_cpu_only_and_no_gpu_id(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    p = parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    assert "gui" not in sub.choices
    for name in ("impute", "benchmark"):
        help_text = sub.choices[name].format_help()
        assert "--device {cpu}" in help_text
        assert "--gpu-id" not in help_text
    assert platforms.resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="macOS"):
        platforms.resolve_device("cuda")
    with pytest.raises(ValueError, match="macOS"):
        platforms.resolve_device("cpu", 0)
    states = availability(names=["jingwei-dmf", "pimms-vae"])
    assert not any(s["gpu"] for s in states.values())


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_windows_linux_keep_cuda_arguments(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    p = argparse.ArgumentParser()
    platforms.add_device_arguments(p)
    args = p.parse_args(["--device", "cuda", "--gpu-id", "0"])
    assert args.device == "cuda" and args.gpu_id == 0
    with pytest.raises(ValueError):
        platforms.validate_device("cpu", 0)
    with pytest.raises(ValueError):
        platforms.validate_device("auto", -1)


def test_linux_r_does_not_inherit_repo_windows_library(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("R_LIBS_USER", raising=False)
    monkeypatch.delenv("JINGWEI_R_LIBRARY", raising=False)
    env = r_environment()
    if "R_LIBS_USER" in env:
        assert Path(env["R_LIBS_USER"]).name == "linux"


def test_conda_rscript_precedes_system_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    monkeypatch.delenv("JINGWEI_RSCRIPT", raising=False)
    rscript = tmp_path / "bin" / "Rscript"
    rscript.parent.mkdir()
    rscript.touch()
    assert find_rscript() == str(rscript.resolve())
