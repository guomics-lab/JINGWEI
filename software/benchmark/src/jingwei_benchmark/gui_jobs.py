"""Desktop jobs use the same CLI and isolated workers, with durable results."""

from datetime import datetime, timezone
import json
import math
import os
import re
import hashlib
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

import psutil

from .cli import parser
from .methods import canonical_name
from .platforms import validate_device
from .runtime import python_command, user_workspace
from .atomic import write_json
from .protocol import ORIGINAL_MECHANISM, JINGWEI_MECHANISM, uses_mask_rate

TERMINAL = {"completed", "incomplete", "failed", "cancelled", "interrupted"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def _validate_task_records(config, runs):
    """Reject damaged saved records before widgets or plotting consume them."""
    if (not isinstance(config, dict)
            or not isinstance(config.get("methods"), list)
            or not all(isinstance(name, str) and name for name in config["methods"])
            or type(config.get("repeats", 1)) is not int or config.get("repeats", 1) < 1
            or not isinstance(runs, list)):
        raise ValueError("任务配置无效")
    if len(config["methods"]) * config.get("repeats", 1) > 2147483647:
        raise ValueError("任务配置无效")
    for row in runs:
        if (not isinstance(row, dict)
                or not isinstance(row.get("method"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", row["method"])
                or not isinstance(row.get("status"), str)
                or type(row.get("repeat", 0)) is not int or row.get("repeat", 0) < 0):
            raise ValueError("任务结果记录无效")
        for key in ("MSE", "RMSE", "MAE", "NRMSE", "Pearson", "Spearman", "W2", "time_s"):
            value = row.get(key)
            if value is not None:
                try:
                    valid = type(value) in (int, float) and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError("任务结果指标无效")


def make_argv(config, input_path, output_path, cancel_file=None, parameter_file=None):
    mode = config.get("command", "benchmark")
    if mode not in ("benchmark", "impute"):
        raise ValueError("请选择评测或插补任务")
    argv = [mode, "--input", str(input_path), "--output", str(output_path)]
    defaults = vars(parser().parse_args(argv))
    forbidden = {"input", "output", "cancel_file", "method_config"}
    for key, value in config.items():
        if key == "command":
            continue
        if key not in defaults or key in forbidden:
            raise ValueError(f"不支持的设置：{key}")
        if value is None:
            continue
        if isinstance(defaults[key], bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须为布尔值")
            if value:
                argv.append("--" + key.replace("_", "-"))
        elif isinstance(value, list):
            if value:
                argv += ["--" + key.replace("_", "-"), *map(str, value)]
        else:
            argv += ["--" + key.replace("_", "-"), str(value)]
    if cancel_file:
        argv += ["--cancel-file", str(cancel_file)]
    if parameter_file:
        argv += ["--method-config", str(parameter_file)]
    try:
        args = parser().parse_args(argv)
    except SystemExit as exc:
        raise ValueError("配置无效，请检查设备、数据和算法参数") from exc
    if not args.methods or args.methods == ["all"]:
        raise ValueError("请选择具体的插补方法")
    [canonical_name(name) for name in args.methods]
    validate_device(args.device, args.gpu_id)
    notebook = mode == "benchmark" and args.evaluation_protocol == "original-notebook"
    if not notebook and mode == "benchmark" and uses_mask_rate(args.mechanism, args.paper_probabilities) and not 0 < args.mask_rate < 1:
        raise ValueError("评测掩蔽率应在 0 和 1 之间")
    if mode == "impute" and uses_mask_rate(args.mechanism, args.paper_probabilities) and not 0 <= args.mask_rate < 1:
        raise ValueError("训练掩蔽率应在 0（含）和 1 之间")
    custom = set() if notebook else {args.mechanism} - {"mcar", ORIGINAL_MECHANISM, JINGWEI_MECHANISM}
    if custom and not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("分位数应满足 0 ≤ 低分位 < 高分位 ≤ 1")
    if custom and (args.low_probability < args.high_probability or args.logistic_slope > 0):
        raise ValueError("MNAR 缺失概率必须随丰度降低而增加")
    if args.mechanism == JINGWEI_MECHANISM and not args.logistic_slope < 0:
        raise ValueError("JINGWEI MNAR slope must be negative")
    for key in ("max_epochs", "batch_size", "latent_dim", "threads", "timeout", "repeats", "patience", "min_epochs"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} 必须大于 0")
    if not 0 < args.validation_fraction < 1 or not 0 <= args.min_delta < 1:
        raise ValueError("验证比例或改善阈值无效")
    if args.seed < 0 or any(x <= 0 for x in args.hidden_dims):
        raise ValueError("种子必须非负，隐藏层维度必须为正数")
    return argv


class JobStore:
    def __init__(self, root=None):
        self.root = Path(root or user_workspace()).resolve()
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._processes = {}
        self._folders = {}
        self._cli_jobs = set()

    def _folder(self, job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("无效任务标识")
        path = self._folders.get(job_id, self.jobs / job_id)
        if not path.is_dir():
            raise ValueError("任务不存在")
        return path

    def active(self):
        rows = []
        for job in list(self._processes):
            try:
                info = self.info(job)
            except (ValueError, OSError):
                continue
            if info["state"] not in TERMINAL:
                rows.append(info)
        return rows

    def attach(self, directory):
        """Open saved results as data, without executing recorded commands."""
        folder = Path(directory).resolve()
        cli_job = False
        if folder.name == "results" and (folder.parent / "request.json").is_file():
            folder = folder.parent
        request = read_json(folder / "request.json")
        if isinstance(request, dict) and request.get("schema_version") == 1:
            job = request.get("id", "")
            config = request.get("config")
            if (not isinstance(job, str) or len(job) != 32 or any(c not in "0123456789abcdef" for c in job)
                    or not isinstance(config, dict) or not isinstance(config.get("methods"), list)):
                raise ValueError("任务配置无效")
        elif (folder / "manifest.json").is_file():
            manifest = read_json(folder / "manifest.json")
            if not isinstance(manifest, dict):
                raise ValueError("任务配置无效")
            _validate_task_records(manifest.get("config"), manifest.get("runs"))
            job = hashlib.sha256(str(folder).encode("utf-8")).hexdigest()[:32]
            cli_job = True
        else:
            raise ValueError("请选择包含 request.json 或 manifest.json 的任务目录")
        if job in self._processes and self._folders.get(job, self.jobs / job).resolve() != folder:
            raise ValueError("任务标识与当前运行的任务冲突")
        previous = self._folders.get(job)
        was_cli = job in self._cli_jobs
        self._folders[job] = folder
        if cli_job:
            self._cli_jobs.add(job)
        else:
            self._cli_jobs.discard(job)
        try:
            self.info(job)
        except (ValueError, OSError, KeyError, TypeError):
            if previous is None:
                self._folders.pop(job, None)
            else:
                self._folders[job] = previous
            if was_cli:
                self._cli_jobs.add(job)
            else:
                self._cli_jobs.discard(job)
            raise
        return job

    def history(self):
        rows = []
        requests = set(self.jobs.glob("*/request.json")) | {
            folder / "request.json" for job, folder in self._folders.items() if job not in self._cli_jobs}
        for path in requests:
            if not path.is_file():
                continue
            try:
                request = read_json(path)
                job = request.get("id", path.parent.name)
                self._folders[job] = path.parent
                rows.append(self.info(job))
            except (ValueError, TypeError, OSError):
                continue
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def _alive(self, request):
        try:
            process = psutil.Process(request["pid"])
            return process.is_running() and abs(process.create_time() - request["process_created"]) < 0.01
        except (KeyError, TypeError, psutil.Error):
            return False

    def info(self, job_id):
        folder = self._folder(job_id)
        if job_id in self._cli_jobs:
            manifest = read_json(folder / "manifest.json")
            if not isinstance(manifest, dict):
                raise ValueError("任务配置无效")
            config = manifest.get("config")
            runs = manifest.get("runs")
            _validate_task_records(config, runs)
            return {"id": job_id, "name": folder.name, "created_at": manifest.get("created_at", ""),
                    "state": "completed" if runs and all(row.get("status") == "completed" for row in runs) else "incomplete",
                    "config": config, "parameters": {}, "runs": runs, "done": len(runs),
                    "expected": len(config.get("methods", [])) * (config.get("repeats", 1) if config.get("command") == "benchmark" else 1),
                    "folder": str(folder), "results_folder": str(folder), "manifest": manifest, "completion": {}}
        request = read_json(folder / "request.json")
        done = read_json(folder / "completion.json")
        manifest = read_json(folder / "results/manifest.json")
        if (not isinstance(request, dict) or not request
                or not isinstance(request.get("config", {}), dict)
                or not isinstance(done, dict) or not isinstance(manifest, dict)
                or not isinstance(manifest.get("runs", []), list)):
            raise ValueError("任务配置无效")
        _validate_task_records(request.get("config"), manifest.get("runs", []))
        if any(key in request and not isinstance(request[key], str) for key in ("name", "created_at")):
            raise ValueError("任务配置无效")
        if done and type(done.get("return_code")) is not int:
            raise ValueError("任务完成记录无效")
        if done:
            state = {0: "completed", 2: "incomplete", 3: "cancelled"}.get(done.get("return_code"), "failed")
        elif self._alive(request):
            state = "cancelling" if (folder / "cancel.request").exists() else "running"
        else:
            state = "interrupted"
        runs = manifest.get("runs", [])
        expected = len(request.get("config", {}).get("methods", [])) * (
            request.get("config", {}).get("repeats", 1) if request.get("config", {}).get("command") == "benchmark" else 1)
        return {**request, "name": request.get("name", folder.name), "id": job_id, "state": state, "runs": runs, "done": len(runs),
                "expected": expected, "folder": str(folder), "results_folder": str(folder / "results"), "manifest": manifest,
                "completion": done}

    def submit(self, source, config, parameters=None, name="", output_root=None):
        source = Path(source).resolve()
        if not source.is_file() or source.suffix.lower() not in (".csv", ".h5ad"):
            raise ValueError("请选择有效的 CSV 或 h5ad 文件")
        with self._lock:
            if self.active():
                raise ValueError("已有任务在运行，请先等待完成或停止当前任务")
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            supplied_name = name.strip()
            task_name = supplied_name or f"task-{stamp}"
            job_id = uuid.uuid4().hex
            if output_root is None:
                folder = self.jobs / job_id
            else:
                base = Path(output_root).expanduser().resolve()
                base.mkdir(parents=True, exist_ok=True)
                label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', task_name).strip(' .')[:80] or 'benchmark'
                suffix = f"-{stamp}" if supplied_name else ""
                folder = base / f"{label}{suffix}-{job_id[:8]}"
            folder.mkdir()
            self._folders[job_id] = folder
            target = folder / ("input" + source.suffix.lower())
            parameter_file = folder / "method_parameters.json"
            request = None
            try:
                argv = make_argv(config, target, folder / "results", folder / "cancel.request", parameter_file)
                shutil.copy2(source, target)
                write_json(parameter_file, parameters or {})
                request = {"schema_version": 1, "id": job_id, "name": task_name,
                           "source_name": source.name, "created_at": datetime.now(timezone.utc).isoformat(),
                           "config": config, "parameters": parameters or {}, "argv": argv}
                # Publish history only after PID/creation-time identity exists.
                # Frozen process launch can take long enough for GUI polling to
                # otherwise mistake the half-created request for an interruption.
                runner_request = folder / "runner_request.json"
                write_json(runner_request, request)
                with (folder / "launch.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(python_command("gui_job_runner", runner_request),
                        stdout=log, stderr=subprocess.STDOUT, cwd=folder,
                        start_new_session=os.name != "nt",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                             "PYTHONUNBUFFERED": "1",
                             "MPLBACKEND": "Agg", "JINGWEI_PROCESS_LOG": str(folder / "launch.log")})
                request.update(pid=process.pid, process_created=psutil.Process(process.pid).create_time())
                write_json(folder / "request.json", request)
                self._processes[job_id] = process
                threading.Thread(target=process.wait, daemon=True).start()
            except Exception as exc:
                write_json(folder / "completion.json", {"return_code": 1, "error": str(exc)})
                if request is not None:
                    write_json(folder / "request.json", request)
                raise
            return job_id

    def _stop_tree(self, job_id):
        info = self.info(job_id)
        owned = self._processes.get(job_id)
        if owned is None or owned.pid != info.get("pid") or not self._alive(info):
            return
        try:
            parent = psutil.Process(info["pid"])
            children = parent.children(recursive=True)
            # Isolated method workers have their own POSIX process groups.
            # Enumerate only descendants of this recorded, creation-time-verified job.
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            parent.kill()
            psutil.wait_procs([parent, *children], timeout=3)
        except psutil.Error:
            pass
        write_json(self._folder(job_id) / "completion.json", {"return_code": 3, "forced_stop": True})

    def cancel(self, job_id, force=False):
        folder = self._folder(job_id)
        if self.info(job_id)["state"] in TERMINAL:
            return
        if job_id not in self._processes:
            raise ValueError("只能停止当前窗口启动的任务")
        (folder / "cancel.request").touch()
        if force:
            self._stop_tree(job_id)
        else:
            def fallback():
                time.sleep(8)
                if self.info(job_id)["state"] not in TERMINAL:
                    self._stop_tree(job_id)
            threading.Thread(target=fallback, daemon=True).start()

    def export(self, job_id, destination):
        folder = self._folder(job_id)
        if self.info(job_id)["state"] not in TERMINAL:
            raise ValueError("请在任务结束后导出结果")
        destination = Path(destination).resolve()
        if destination.is_relative_to(folder):
            raise ValueError("导出位置应位于任务目录之外")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder):
                    archive.write(path, path.relative_to(folder))


def inspect_input(path, options):
    from .data_io import load_matrix
    options = dict(options)
    protocol = options.pop("evaluation_protocol", "fair")
    if protocol == "original-notebook":
        from .notebook_protocol import load_notebook_preview
        data = load_notebook_preview(path, max_dense_mb=options.get("max_dense_mb", 2048))
    else:
        data = load_matrix(path, **options)
    frame = data.frame
    preview = frame.iloc[:12, :10]
    return {"samples": len(frame), "features": frame.shape[1],
            "missing_rate": float(1 - data.observed.mean()),
            "complete_rows": int(data.observed.all(axis=1).sum()),
            "working_scale": data.scale, "csv_detection": data.csv_detection,
            "preview": preview, "feature_ids": list(frame.columns),
            "layers": list(data.adata.layers) if data.adata is not None else []}
