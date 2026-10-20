import argparse
import sys
import pytest
from jingwei import platforms


def test_mac_cli_has_cpu_only_and_no_gpu_id(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    p = argparse.ArgumentParser()
    platforms.add_device_arguments(p)
    help_text = p.format_help()
    assert "--device {cpu}" in help_text
    assert "--gpu-id" not in help_text
    assert platforms.resolve_device("auto") == "cpu"
    with pytest.raises(ValueError, match="macOS"):
        platforms.resolve_device("cuda")
    with pytest.raises(ValueError, match="macOS"):
        platforms.resolve_device("cpu", 0)


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
