"""OS-level process-tree containment, parent-death watchdogs, and exit guards.

Prevents orphaned or deadlocked ``python.exe`` GPU workers on Windows and POSIX:
1. Windows Job Objects with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ensure the
   NT kernel atomically terminates all child/grandchild workers (CUDA, DataLoader
   spawn workers, joblib/loky workers, Rscript) even if the parent CLI, GUI, or
   test runner crashes or is killed via ``TerminateProcess`` / Task Manager.
2. Parent-liveness watchdogs inside child workers self-terminate if the parent
   process disappears unexpectedly (covering POSIX ``SIGKILL`` and environments
   where Job Object assignment is restricted).
3. Exit watchdogs and explicit executor/DataLoader teardown prevent PyTorch
   CUDA / WDDM / OpenMP finalization deadlocks after worker outputs are saved.
"""

from __future__ import annotations

import ctypes
import gc
import os
import signal
import subprocess
import sys
import threading
import time

import psutil

PARENT_PID_ENV = "JINGWEI_PARENT_PID"
PARENT_CREATED_ENV = "JINGWEI_PARENT_CREATED"

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800


if os.name == "nt":
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateJobObjectW = _kernel32.CreateJobObjectW
    _CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _CreateJobObjectW.restype = wintypes.HANDLE

    _SetInformationJobObject = _kernel32.SetInformationJobObject
    _SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _SetInformationJobObject.restype = wintypes.BOOL

    _AssignProcessToJobObject = _kernel32.AssignProcessToJobObject
    _AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _AssignProcessToJobObject.restype = wintypes.BOOL

    _TerminateJobObject = _kernel32.TerminateJobObject
    _TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _TerminateJobObject.restype = wintypes.BOOL

    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL


def parent_watchdog_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Attach the current process's PID and creation timestamp to child env."""
    env = dict(os.environ if base_env is None else base_env)
    try:
        created = psutil.Process(os.getpid()).create_time()
        env[PARENT_PID_ENV] = str(os.getpid())
        env[PARENT_CREATED_ENV] = f"{created:.6f}"
    except psutil.Error:
        pass
    return env


def _kill_psutil_tree(pid: int) -> None:
    """Suspend parent first so it cannot wake up and spawn new workers while children die."""
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    try:
        parent.suspend()
    except psutil.Error:
        pass
    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        parent.kill()
    except psutil.Error:
        pass
    psutil.wait_procs([parent, *children], timeout=3)


class ManagedProcessTree:
    """Bind a spawned ``subprocess.Popen`` and all its descendants to parent lifetime."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.pid = getattr(process, "pid", None)
        self._lock = threading.Lock()
        self._job_handle = None
        if os.name == "nt":
            proc_handle = getattr(process, "_handle", None)
            if isinstance(proc_handle, int) and proc_handle > 0:
                job = _CreateJobObjectW(None, None)
                if job:
                    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    ok = _SetInformationJobObject(
                        job,
                        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                        ctypes.byref(info),
                        ctypes.sizeof(info),
                    )
                    if ok and _AssignProcessToJobObject(job, wintypes.HANDLE(proc_handle)):
                        self._job_handle = job
                    else:
                        _CloseHandle(job)

    @property
    def has_job_object(self) -> bool:
        return self._job_handle is not None

    def terminate(self) -> None:
        """Atomically kill the root process and every descendant in its tree."""
        with self._lock:
            job = self._job_handle
            self._job_handle = None
        still_running = self.process.poll() is None
        if os.name == "nt":
            if job is not None:
                try:
                    _TerminateJobObject(job, 1)
                finally:
                    _CloseHandle(job)
            if still_running and isinstance(self.pid, int) and self.pid > 0:
                _kill_psutil_tree(self.pid)
                if self.process.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(self.pid), "/T", "/F"],
                        capture_output=True,
                    )
        else:
            if isinstance(self.pid, int) and self.pid > 0:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    if still_running:
                        _kill_psutil_tree(self.pid)
        try:
            self.process.wait(timeout=3)
        except Exception:
            pass

    def close(self) -> None:
        """Sweep any stray grandchild workers and release OS handles after completion."""
        self.terminate()


def start_parent_watchdog(interval: float = 1.0) -> threading.Thread | None:
    """Self-terminate this worker and its children if the parent process vanishes."""
    raw_pid = os.environ.get(PARENT_PID_ENV)
    raw_created = os.environ.get(PARENT_CREATED_ENV)
    if not raw_pid:
        return None
    try:
        parent_pid = int(raw_pid)
        expected_created = float(raw_created) if raw_created else None
    except ValueError:
        return None

    def _parent_alive() -> bool:
        try:
            parent = psutil.Process(parent_pid)
            if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                return False
            if expected_created is not None and abs(parent.create_time() - expected_created) > 0.5:
                return False
            return True
        except psutil.Error:
            return False

    def _watch() -> None:
        while True:
            time.sleep(interval)
            if not _parent_alive():
                try:
                    current = psutil.Process(os.getpid())
                    for child in reversed(current.children(recursive=True)):
                        try:
                            child.kill()
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                os._exit(130)

    thread = threading.Thread(target=_watch, name="jingwei-parent-watchdog", daemon=True)
    thread.start()
    return thread


def shutdown_background_workers() -> None:
    """Shut down reusable joblib/loky pools and synchronize/release CUDA state."""
    reusable_mod = sys.modules.get("joblib.externals.loky.reusable_executor")
    executor = getattr(reusable_mod, "_executor", None) if reusable_mod is not None else None
    if executor is not None:
        try:
            executor.shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available() and torch_mod.cuda.is_initialized():
                torch_mod.cuda.synchronize()
                torch_mod.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def finalize_worker_exit(code: int, *, timeout: float = 5.0) -> int:
    """Flush streams, release pools/CUDA, and arm a hard-exit watchdog against C++ teardown hangs."""
    shutdown_background_workers()
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None:
            try:
                stream.flush()
            except Exception:
                pass
    watchdog = threading.Timer(timeout, os._exit, args=(int(code),))
    watchdog.daemon = True
    watchdog.start()
    return int(code)
