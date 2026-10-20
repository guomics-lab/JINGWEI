"""Runtime device policy shared by the CLI and Python adapters."""

import os
import sys


def platform_name():
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def cuda_supported():
    return sys.platform != "darwin"


def add_device_arguments(parser):
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto") if cuda_supported() else ("cpu",),
        default="auto" if cuda_supported() else "cpu",
    )
    parser.set_defaults(gpu_id=None)
    if cuda_supported():
        parser.add_argument("--gpu-id", type=int)


def validate_device(device, gpu_id=None):
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError(f"unknown device {device}")
    if not cuda_supported() and (device == "cuda" or gpu_id is not None):
        raise ValueError("macOS supports CPU training only")
    if gpu_id is not None:
        if gpu_id < 0:
            raise ValueError("--gpu-id must be nonnegative")
        if device == "cpu":
            raise ValueError("--gpu-id requires --device cuda or auto")


def resolve_device(device, gpu_id=None, neural=True):
    validate_device(device, gpu_id)
    if not cuda_supported() or device == "cpu":
        return "cpu"
    if device == "auto" and not neural and gpu_id is None:
        return "cpu"
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    available = torch.cuda.is_available()
    if (device == "cuda" or gpu_id is not None) and not available:
        raise ValueError("CUDA requested but unavailable in this PyTorch environment")
    return "cuda" if available else "cpu"
