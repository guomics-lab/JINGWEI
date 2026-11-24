"""Regression tests verifying zero orphaned or deadlocked GPU worker processes."""

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pytest


def _is_alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _wait_until(predicate, timeout: float = 30.0, step: float = 0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    pytest.fail("timed out waiting for process condition")


def _gpu_compute_pids() -> set[int]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode != 0:
            return set()
        pids = set()
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids
    except (OSError, subprocess.SubprocessError):
        return set()


def _make_csv(tmp_path: Path) -> Path:
    source = tmp_path / "matrix.csv"
    pd.DataFrame(
        np.random.default_rng(19).lognormal(4.0, 0.4, (16, 12)),
        index=[f"S{i}" for i in range(16)],
        columns=[f"P{i}" for i in range(12)],
    ).to_csv(source)
    return source


def _preferred_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def test_abrupt_kill_of_cli_parent_leaves_zero_gpu_or_python_worker_orphans(tmp_path):
    """Simulate Terminal close / IDE Stop / TerminateProcess on CLI parent while GPU worker trains."""
    source = _make_csv(tmp_path)
    out_dir = tmp_path / "cli_out"
    device = _preferred_device()

    cli_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jingwei_benchmark",
            "benchmark",
            "--input",
            str(source),
            "--output",
            str(out_dir),
            "--methods",
            "jingwei-dmf",
            "--device",
            device,
            "--max-epochs",
            "100000",
            "--batch-size",
            "4",
            "--latent-dim",
            "4",
            "--hidden-dims",
            "8",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    worker_pids = []
    try:
        run_log = out_dir / "repeat_000/jingwei-dmf/run.log"
        _wait_until(
            lambda: run_log.is_file() and "Epoch 1/" in run_log.read_text(encoding="utf-8", errors="replace"),
            timeout=60.0,
        )
        parent_ps = psutil.Process(cli_proc.pid)
        worker_pids = [c.pid for c in parent_ps.children(recursive=True)]
        assert worker_pids, "Expected active child worker process"
        if device == "cuda":
            gpu_pids = _gpu_compute_pids()
            if gpu_pids:
                assert set(worker_pids) & gpu_pids

        # Kill ONLY the parent CLI process (no /T), reproducing abrupt parent termination.
        parent_ps.kill()
        cli_proc.wait(timeout=5)

        _wait_until(lambda: not any(_is_alive(pid) for pid in worker_pids), timeout=5.0)
        assert not (set(worker_pids) & _gpu_compute_pids())
    finally:
        if cli_proc.poll() is None:
            cli_proc.kill()
        for pid in worker_pids:
            if _is_alive(pid):
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass


def test_abrupt_kill_of_gui_host_leaves_zero_runner_or_gpu_worker_orphans(tmp_path):
    """Simulate Task Manager / crash killing GUI process while grandchild GPU worker trains."""
    source = _make_csv(tmp_path)
    device = _preferred_device()
    pid_file = tmp_path / "spawned.json"
    host_script = tmp_path / "gui_host.py"
    host_script.write_text(
        f"""
import json, time
from pathlib import Path
import psutil
from jingwei_benchmark.gui_jobs import JobStore

store = JobStore(Path({str(tmp_path / 'ws')!r}))
job_id = store.submit(
    Path({str(source)!r}),
    {{
        "command": "benchmark",
        "methods": ["jingwei-dmf"],
        "device": {device!r},
        "repeats": 1,
        "max_epochs": 100000,
        "batch_size": 4,
        "latent_dim": 4,
        "hidden_dims": [8],
    }},
)
info = store.info(job_id)
run_log = Path(info["results_folder"]) / "repeat_000/jingwei-dmf/run.log"
while not (run_log.is_file() and "Epoch 1/" in run_log.read_text(encoding="utf-8", errors="replace")):
    time.sleep(0.1)
runner_pid = info["pid"]
children = [c.pid for c in psutil.Process(runner_pid).children(recursive=True)]
Path({str(pid_file)!r}).write_text(json.dumps({{"runner": runner_pid, "workers": children}}), encoding="utf-8")
time.sleep(120)
""",
        encoding="utf-8",
    )
    host_proc = subprocess.Popen(
        [sys.executable, str(host_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    tracked = []
    try:
        _wait_until(pid_file.is_file, timeout=60.0)
        recorded = json.loads(pid_file.read_text(encoding="utf-8"))
        tracked = [recorded["runner"], *recorded["workers"]]
        assert len(tracked) >= 2
        assert all(_is_alive(pid) for pid in tracked)

        # Kill ONLY the top-level GUI host process.
        psutil.Process(host_proc.pid).kill()
        host_proc.wait(timeout=5)

        _wait_until(lambda: not any(_is_alive(pid) for pid in tracked), timeout=5.0)
        assert not (set(tracked) & _gpu_compute_pids())
    finally:
        if host_proc.poll() is None:
            host_proc.kill()
        for pid in tracked:
            if _is_alive(pid):
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass


def test_forced_stop_prevents_spawning_second_method_worker(tmp_path):
    """Forced stop must suspend/kill atomically so runner cannot spawn the next GPU method."""
    from jingwei_benchmark.gui_jobs import JobStore

    source = _make_csv(tmp_path)
    store = JobStore(tmp_path / "ws")
    job_id = store.submit(
        source,
        {
            "command": "benchmark",
            "methods": ["jingwei-dmf", "jingwei-dcae"],
            "device": "cpu",
            "repeats": 1,
            "max_epochs": 100000,
            "batch_size": 4,
            "latent_dim": 4,
            "hidden_dims": [8],
        },
    )
    info = store.info(job_id)
    results_dir = Path(info["results_folder"])
    dmf_log = results_dir / "repeat_000/jingwei-dmf/run.log"
    _wait_until(
        lambda: dmf_log.is_file() and "Epoch 1/" in dmf_log.read_text(encoding="utf-8", errors="replace"),
        timeout=45.0,
    )
    descendants = [c.pid for c in psutil.Process(info["pid"]).children(recursive=True)]
    assert descendants

    store.cancel(job_id, force=True)
    _wait_until(lambda: not any(_is_alive(pid) for pid in [info["pid"], *descendants]), timeout=5.0)
    assert not (results_dir / "repeat_000/jingwei-dcae").exists()


def test_parent_watchdog_self_terminates_without_job_object(tmp_path):
    """Fallback watchdog must terminate child even when Job Objects are bypassed (e.g. POSIX SIGKILL)."""
    child_pid_file = tmp_path / "child_pid.txt"
    parent_script = tmp_path / "parent_no_job.py"
    child_script = tmp_path / "child_watched.py"

    child_script.write_text(
        f"""
import os, time
from pathlib import Path
from jingwei_benchmark.process_guard import start_parent_watchdog

start_parent_watchdog(interval=0.2)
Path({str(child_pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(120)
""",
        encoding="utf-8",
    )
    parent_script.write_text(
        f"""
import subprocess, sys, time
from jingwei_benchmark.process_guard import parent_watchdog_env

subprocess.Popen([sys.executable, {str(child_script)!r}], env=parent_watchdog_env())
time.sleep(120)
""",
        encoding="utf-8",
    )
    parent_proc = subprocess.Popen([sys.executable, str(parent_script)])
    child_pid = None
    try:
        _wait_until(child_pid_file.is_file, timeout=15.0)
        child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
        assert _is_alive(child_pid)

        psutil.Process(parent_proc.pid).kill()
        parent_proc.wait(timeout=5)

        _wait_until(lambda: not _is_alive(child_pid), timeout=4.0)
    finally:
        if parent_proc.poll() is None:
            parent_proc.kill()
        if child_pid and _is_alive(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.Error:
                pass


def test_finalize_worker_exit_breaks_post_main_teardown_deadlock(tmp_path):
    """Simulates a C++/CUDA atexit deadlock after worker_result.json is written."""
    hang_script = tmp_path / "hang_on_exit.py"
    hang_script.write_text(
        """
import atexit, time
from jingwei_benchmark.process_guard import finalize_worker_exit

@atexit.register
def _simulated_cuda_destructor_deadlock():
    time.sleep(60)

raise SystemExit(finalize_worker_exit(0, timeout=0.6))
""",
        encoding="utf-8",
    )
    t0 = time.perf_counter()
    completed = subprocess.run([sys.executable, str(hang_script)], timeout=10)
    elapsed = time.perf_counter() - t0
    assert completed.returncode == 0
    assert elapsed < 4.0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GUI close cleanup")
def test_closing_window_during_live_environment_probe_leaves_zero_probe_processes(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from jingwei_benchmark.gui import BenchmarkWindow

    app = QApplication.instance() or QApplication([])
    current_ps = psutil.Process(os.getpid())
    win = BenchmarkWindow(tmp_path / "live_probe_ws", load_default=False, startup=True)
    # Wait briefly until inspect_environment actually spawns gui_environment subprocess
    _wait_until(lambda: len(current_ps.children(recursive=True)) > 0 or not win._environment_checking, timeout=10.0)
    win.close()
    app.processEvents()
    _wait_until(
        lambda: not any(
            "gui_environment" in " ".join(child.cmdline())
            for child in current_ps.children(recursive=True)
            if child.is_running()
        ),
        timeout=5.0,
    )
