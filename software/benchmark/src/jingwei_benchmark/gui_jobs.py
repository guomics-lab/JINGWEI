"""Desktop jobs use the same CLI and isolated workers, with durable results."""

from datetime import datetime, timezone
import json
import os
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

TERMINAL = {"completed", "incomplete", "failed", "cancelled", "interrupted"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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
    if not 0 <= args.training_mask_rate <= 1 or not 0 < args.mask_rate < 1:
        raise ValueError("评测掩蔽率应在 0 和 1 之间；训练掩蔽率应在 0 和 1 之间（可包含端点）")
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("分位数应满足 0 ≤ 低分位 < 高分位 ≤ 1")
    if args.low_probability < args.high_probability or args.logistic_slope > 0:
        raise ValueError("MNAR 缺失概率必须随丰度降低而增加")
    for key in ("max_epochs", "batch_size", "latent_dim", "threads", "timeout", "repeats"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} 必须大于 0")
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

    def _folder(self, job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("无效任务标识")
        path = self.jobs / job_id
        if not path.is_dir():
            raise ValueError("任务不存在")
        return path

    def history(self):
        rows = []
        for path in self.jobs.glob("*/request.json"):
            try:
                rows.append(self.info(path.parent.name))
            except ValueError:
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
        request = read_json(folder / "request.json")
        done = read_json(folder / "completion.json")
        manifest = read_json(folder / "results/manifest.json")
        if done:
            state = {0: "completed", 2: "incomplete", 3: "cancelled"}.get(done.get("return_code"), "failed")
        elif self._alive(request):
            state = "cancelling" if (folder / "cancel.request").exists() else "running"
        else:
            state = "interrupted"
        runs = manifest.get("runs", [])
        expected = len(request.get("config", {}).get("methods", [])) * (
            request.get("config", {}).get("repeats", 5) if request.get("config", {}).get("command") == "benchmark" else 1)
        return {**request, "id": job_id, "state": state, "runs": runs, "done": len(runs),
                "expected": expected, "folder": str(folder), "manifest": manifest,
                "completion": done}

    def submit(self, source, config, parameters=None, name=""):
        source = Path(source).resolve()
        if not source.is_file() or source.suffix.lower() not in (".csv", ".h5ad"):
            raise ValueError("请选择有效的 CSV 或 h5ad 文件")
        with self._lock:
            if any(job["state"] not in TERMINAL for job in self.history()):
                raise ValueError("已有任务在运行，请先等待完成或停止当前任务")
            job_id = uuid.uuid4().hex
            folder = self.jobs / job_id
            folder.mkdir()
            target = folder / ("input" + source.suffix.lower())
            parameter_file = folder / "method_parameters.json"
            request = None
            try:
                argv = make_argv(config, target, folder / "results", folder / "cancel.request", parameter_file)
                shutil.copy2(source, target)
                write_json(parameter_file, parameters or {})
                request = {"schema_version": 1, "id": job_id, "name": name.strip() or source.stem,
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
        if not self._alive(info):
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
    data = load_matrix(path, **options)
    frame = data.frame
    preview = frame.iloc[:12, :10]
    return {"samples": len(frame), "features": frame.shape[1],
            "missing_rate": float(1 - data.observed.mean()),
            "complete_rows": int(data.observed.all(axis=1).sum()),
            "working_scale": data.scale, "csv_detection": data.csv_detection,
            "preview": preview, "feature_ids": list(frame.columns),
            "layers": list(data.adata.layers) if data.adata is not None else []}
