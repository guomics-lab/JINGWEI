"""Check the actual frozen desktop UI and backend without conda/Python on PATH."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    console = executable.with_name("jingwei-benchmark.exe")
    if not console.is_file():
        raise RuntimeError("Missing standalone console executable beside the GUI")
    env = {key: value for key, value in os.environ.items()
           if key not in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "JINGWEI_RSCRIPT", "JINGWEI_R_LIBRARY", "R_HOME", "R_LIBS_USER", "PYTHONPATH")}
    env["PATH"] = str(Path(os.environ.get("WINDIR", "C:/Windows")) / "System32") if os.name == "nt" else "/usr/bin:/bin"
    env["QT_QPA_PLATFORM"] = "windows"
    env["JINGWEI_PROCESS_LOG"] = str(output / "frozen-validation.log")
    def run(module, *argv, timeout=900):
        command = [str(console), *map(str, argv)] if module == "cli" else [str(executable), "--jingwei-internal", module, *map(str, argv)]
        result = subprocess.run(command,
                                cwd=ROOT, env=env, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"Frozen {module} failed ({result.returncode}); see {env['JINGWEI_PROCESS_LOG']}")
    run("desktop_smoke", output / "native-ui", timeout=360)
    ui = json.loads((output / "native-ui/smoke-result.json").read_text(encoding="utf-8"))
    assert ui["status"] == "completed" and ui["logo_loaded"] and ui["native_ui"]
    assert ui["qt_platform"] == "windows" and all(ui["windows_native_icons"].values())
    assert ui["app_user_model_id"] == "JINGWEI.Benchmark.Desktop"
    assert ui["languages"] == ["zh", "en"] and ui["language_switch_keeps_configuration"]
    assert ui["default_timeout_seconds"] == 1800
    run("desktop_smoke", output / "native-ui", "--reopen", timeout=360)
    reopened = json.loads((output / "native-ui/reopen-result.json").read_text(encoding="utf-8"))
    assert reopened["status"] == "completed" and reopened["environment_cache_hit"]
    assert reopened["environment_probe_count"] == 0
    assert reopened["configuration"] == ui["configuration"]
    assert reopened["method_parameters"] == ui["method_parameters"]
    help_result = subprocess.run([str(console), "--help"], env=env, capture_output=True, text=True, check=True)
    assert "jingwei-benchmark" in help_result.stdout
    benchmark_help = subprocess.run([str(console), "benchmark", "--help"], env=env, capture_output=True, text=True, check=True)
    assert "all 35 methods" in benchmark_help.stdout
    assert "1800; 30 min" in " ".join(benchmark_help.stdout.split())
    default_inspection = subprocess.run([str(console), "inspect"], env=env, capture_output=True, text=True, check=True)
    default_data = json.loads(default_inspection.stdout)
    assert (default_data["samples"], default_data["features"]) == (210, 1542)
    fixture = Path(ui["job_folder"]) / "input.csv"
    methods = ["all"] if args.all_methods else ["mean", "bpca", "jingwei-dmf", "pimms-vae"]
    common = ["benchmark", "--input", fixture, "--repeats", "1", "--mask-rate", ".05", "--threads", "1",
              "--max-epochs", "1", "--batch-size", "4", "--latent-dim", "4", "--hidden-dims", "8", "--timeout", "180"]
    run("cli", *common, "--methods", *methods, "--device", "cpu", "--output", output / "cpu")
    import pandas as pd
    cpu = pd.read_csv(output / "cpu/metrics.csv")
    assert (cpu.status == "completed").all()
    if args.all_methods:
        assert len(cpu) == 35
        for name in ("pimms-dae", "pimms-vae", "pimms-cf"):
            detail = json.loads((output / f"cpu/repeat_000/{name}/worker_result.json").read_text())
            assert detail["pimms_acceleration"]["applied"] and detail["training_steps"] > 0
        detail = json.loads((output / "cpu/repeat_000/mice-norm/worker_result.json").read_text())
        assert detail["r_acceleration"]["applied"] and detail["mice_imputations"] == 5 and detail["mice_iterations"] == 5
        assert (output / "cpu/repeat_000/mice-norm/r_adapter/r_acceleration.R").is_file()
        detail = json.loads((output / "cpu/repeat_000/mice-pmm/worker_result.json").read_text())
        assert detail["implementation"] == "python-linear-mice-pmm" and detail["mice_backend"] == "python"
        assert detail["mice_imputations"] == 5 and detail["mice_iterations"] == 5 and detail["all_predictors"]
        assert (output / "cpu/repeat_000/mice-pmm/mice_draws.npz").is_file()
        detail = json.loads((output / "cpu/repeat_000/iterative-rf/worker_result.json").read_text())
        assert detail["retained_estimators"] == 0 and detail["trees"] == 100
    if args.cuda:
        neural = ["jingwei-dmf", "jingwei-dcae", "pimms-dae", "pimms-vae", "pimms-cf"]
        run("cli", *common, "--methods", *neural, "--device", "cuda", "--gpu-id", "0", "--output", output / "cuda")
        gpu = pd.read_csv(output / "cuda/metrics.csv")
        assert len(gpu) == 5 and (gpu.status == "completed").all()
        assert gpu.actual_device.str.startswith("cuda").all()
    full_input_checked = False
    if args.all_methods:
        # The integration matrix has complete profiles and cannot expose the
        # real default dataset's bootstrap/encoding limitations or QRILC axes.
        full_output = output / "full-input-check"
        checked = subprocess.run([str(console), "benchmark", "--methods", "seqknn", "impseq", "mle", "qrilc", "mice-pmm",
            "--repeats", "1", "--device", "cpu", "--output", str(full_output)], cwd=ROOT, env=env, timeout=300)
        assert checked.returncode == 2
        full = json.loads((full_output / "metrics.json").read_text(encoding="utf-8"))
        assert json.loads((full_output / "manifest.json").read_text(encoding="utf-8"))["config"]["timeout"] == 1800
        by_name = {row["method"]: row for row in full}
        for name in ("seqknn", "impseq", "mle"):
            assert by_name[name]["status"] == "failed" and by_name[name]["failure_kind"] == "unsupported_input"
        assert by_name["qrilc"]["status"] == "completed" and by_name["qrilc"]["n_evaluated"] > 0
        assert by_name["mice-pmm"]["status"] == "completed" and by_name["mice-pmm"]["n_evaluated"] > 0
        full_input_checked = True
    (output / "validation.json").write_text(json.dumps({"native_ui": ui, "reopened_ui": reopened, "cpu_completed": len(cpu),
        "cuda_completed": 5 if args.cuda else 0, "conda_paths_removed": True,
        "public_console_validated": True, "full_default_input_checked": full_input_checked}, indent=2), encoding="utf-8")
    print("Frozen desktop UI and backend validation passed")


if __name__ == "__main__":
    main()
