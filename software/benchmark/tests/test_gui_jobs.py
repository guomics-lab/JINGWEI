import json
from pathlib import Path
import time
import threading

import numpy as np
import pandas as pd
import psutil
import pytest

from jingwei_benchmark.cli import main
from jingwei_benchmark.gui_jobs import JobStore, make_argv, TERMINAL
from jingwei_benchmark.runtime import python_command, resource_root


def fixture(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame(np.random.default_rng(7).uniform(10, 100, (8, 6)),
                 index=[f"S{i}" for i in range(8)], columns=[f"P{i}" for i in range(6)]).to_csv(path)
    return path


def wait_job(store, job_id, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = store.info(job_id)
        if info["state"] in TERMINAL:
            return info
        time.sleep(0.1)
    store.cancel(job_id, force=True)
    pytest.fail("desktop job did not finish")


def live(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_desktop_cli_parity_history_and_export(tmp_path):
    source = fixture(tmp_path)
    store = JobStore(tmp_path / "desktop")
    config = {"command": "benchmark", "methods": ["mean", "knn"], "repeats": 2, "device": "cpu"}
    job_id = store.submit(source, config, name="comparison")
    info = wait_job(store, job_id)
    assert info["state"] == "completed" and info["done"] == info["expected"] == 4
    assert JobStore(store.root).history()[0]["state"] == "completed"
    folder = Path(info["folder"])
    assert main(make_argv(config, source, tmp_path / "cli")) == 0
    for repeat in range(2):
        a = np.load(folder / f"results/repeat_{repeat:03d}/masks.npz")
        b = np.load(tmp_path / f"cli/repeat_{repeat:03d}/masks.npz")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    a = pd.read_csv(folder / "results/metrics.csv")
    b = pd.read_csv(tmp_path / "cli/metrics.csv")
    np.testing.assert_allclose(a[["RMSE", "MAE", "W2"]], b[["RMSE", "MAE", "W2"]])
    archive = tmp_path / "complete.zip"
    store.export(job_id, archive)
    import zipfile
    with zipfile.ZipFile(archive) as bundle:
        assert "results/manifest.json" in bundle.namelist()
        assert "input.csv" in bundle.namelist()
        assert "results/repeat_000/masks.npz" in bundle.namelist()


def test_omitted_repeat_setting_uses_the_single_run_cli_default(tmp_path):
    store=JobStore(tmp_path/'desktop')
    job=store.submit(fixture(tmp_path),{'command':'benchmark','methods':['mean'],'device':'cpu'})
    info=wait_job(store,job)
    assert info['state']=='completed' and info['done']==info['expected']==1


def test_cancel_stops_owned_worker_and_retains_cancelled_status(tmp_path):
    store = JobStore(tmp_path / "desktop")
    job_id = store.submit(fixture(tmp_path), {
        "command": "benchmark", "methods": ["jingwei-dmf"], "device": "cpu", "repeats": 1,
        "max_epochs": 100000, "batch_size": 4, "latent_dim": 4, "hidden_dims": [8],
    })
    deadline = time.monotonic() + 30
    descendants = []
    while time.monotonic() < deadline:
        info = store.info(job_id)
        try:
            descendants = psutil.Process(info["pid"]).children(recursive=True)
        except psutil.Error:
            pass
        if descendants:
            break
        time.sleep(0.1)
    assert descendants
    store.cancel(job_id)
    info = wait_job(store, job_id)
    assert info["state"] == "cancelled"
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and any(live(child) for child in descendants):
        time.sleep(0.05)
    assert not any(live(child) for child in descendants)
    assert any(run["status"] == "cancelled" for run in info["runs"])


def test_configuration_and_job_paths_are_validated(tmp_path):
    with pytest.raises(ValueError):
        make_argv({"methods": ["mean"], "output": "/other"}, "a.csv", "output")
    with pytest.raises(ValueError):
        make_argv({"methods": ["unknown"]}, "a.csv", "output")
    with pytest.raises(ValueError):
        JobStore(tmp_path).info("../../elsewhere")


def test_frozen_process_dispatch_uses_same_executable(monkeypatch, tmp_path):
    import sys
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert python_command("worker", "job.json")[1:] == ["--jingwei-internal", "worker", "job.json"]
    assert resource_root() == tmp_path


def test_slow_process_launch_never_publishes_an_interrupted_job(monkeypatch,tmp_path):
    import jingwei_benchmark.gui_jobs as jobs
    original=jobs.subprocess.Popen
    entered,release=threading.Event(),threading.Event()
    def slow_launch(*args,**kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args,**kwargs)
    monkeypatch.setattr(jobs.subprocess,"Popen",slow_launch)
    store=JobStore(tmp_path/'desktop')
    result=[]
    failure=[]
    def submit():
        try:
            result.append(store.submit(fixture(tmp_path),{"command":"benchmark","methods":["mean"],"repeats":1,"device":"cpu"}))
        except Exception as exc:
            failure.append(exc)
    thread=threading.Thread(target=submit)
    thread.start()
    try:
        assert entered.wait(10)
        # Poll repeatedly while launch is paused at exactly the former race.
        for _ in range(5):
            assert store.history()==[]
    finally:
        release.set()
        thread.join(15)
    assert not failure and not thread.is_alive()
    assert store.info(result[0])["state"] in ("running","completed")
    assert wait_job(store,result[0])["state"]=="completed"


def test_failed_process_launch_is_still_saved_in_history(monkeypatch,tmp_path):
    import jingwei_benchmark.gui_jobs as jobs
    def fail(*args,**kwargs):
        raise OSError("simulated launch failure")
    monkeypatch.setattr(jobs.subprocess,"Popen",fail)
    store=JobStore(tmp_path/'desktop')
    with pytest.raises(OSError,match="simulated launch"):
        store.submit(fixture(tmp_path),{"command":"benchmark","methods":["mean"],"repeats":1,"device":"cpu"})
    history=store.history()
    assert len(history)==1 and history[0]["state"]=="failed"
    assert "simulated launch" in history[0]["completion"]["error"]
