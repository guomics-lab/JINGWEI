"""Probe heavy CUDA/R dependencies outside the Qt process."""

import sys
from pathlib import Path
import tempfile
import subprocess
import os
import time

from .gui_jobs import read_json, write_json
from .methods import availability, find_method_source
from .platforms import cuda_supported
from .runtime import python_command, prepare_standard_streams


def inspect_environment(options, *, cancel_event=None):
    with tempfile.TemporaryDirectory(prefix="jingwei-environment-") as temporary:
        folder = Path(temporary)
        request, result = folder / "request.json", folder / "result.json"
        write_json(request, options)
        with (folder / "probe.log").open("w", encoding="utf-8") as log:
            command = python_command("gui_environment", request, result)
            process = subprocess.Popen(command,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "PYTHONUTF8": "1", "JINGWEI_PROCESS_LOG": str(folder / "probe.log")})
            deadline = time.monotonic() + (180 if getattr(sys, "frozen", False) else 65)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("计算环境检查已取消")
                    if time.monotonic() > deadline:
                        raise TimeoutError("计算环境检查超时")
                    try:
                        process.wait(timeout=0.15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    import psutil
                    try:
                        descendants = psutil.Process(process.pid).children(recursive=True)
                        for child in descendants:
                            try:
                                child.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        process.terminate()
                        _, alive = psutil.wait_procs(descendants, timeout=1)
                        for child in alive:
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                    finally:
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        if process.returncode or not result.is_file():
            raise RuntimeError((folder / "probe.log").read_text(encoding="utf-8", errors="replace")[-2000:] or "计算环境检查失败")
        data = read_json(result)
        return data["states"], data["gpu"], data["gpu_name"]


def main():
    prepare_standard_streams()
    options = read_json(sys.argv[1])
    states = availability(options.get("rscript"), options.get("r_library"),
                          options.get("naguider_source") or find_method_source("NAguideR"),
                          options.get("dreamai_source") or find_method_source("DreamAI"),
                          mice_backend=options.get("mice_backend") or "python")
    gpu, gpu_name = False, None
    if cuda_supported():
        try:
            import torch
            gpu = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu else None
        except ImportError:
            pass
    write_json(sys.argv[2], {"states": states, "gpu": gpu, "gpu_name": gpu_name})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
