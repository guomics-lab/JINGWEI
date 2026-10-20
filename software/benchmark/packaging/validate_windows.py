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
    parser.add_argument("--background", action="store_true", help="render native Qt windows without showing them on the user's desktop")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    console = executable.with_name("jingwei-cli.exe")
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
    hidden = ["--hidden"] if args.background else []
    run("desktop_smoke", output / "native-ui", *hidden, timeout=360)
    ui = json.loads((output / "native-ui/smoke-result.json").read_text(encoding="utf-8"))
    assert ui["status"] == "completed" and ui["logo_loaded"] and ui["native_ui"]
    assert ui["qt_platform"] == "windows" and all(ui["windows_native_icons"].values())
    assert ui["app_user_model_id"] == "JINGWEI.Benchmark.Desktop"
    assert ui["languages"] == ["zh", "en"] and ui["language_switch_keeps_configuration"]
    assert ui["default_timeout_seconds"] == 1800
    assert ui["startup_without_task"] and ui["live_log"] and ui["manual_task_open"] and ui["repeats_control_removed"]
    assert ui["dcae_architecture_version"] == "global-ae-v1"
    assert ui["run_settings_rows"] == 3 and ui["startup_environment_ready_before_main"]
    assert ui["application_name"] == "jingwei"
    run("desktop_smoke", output / "native-ui", "--reopen", *hidden, timeout=360)
    reopened = json.loads((output / "native-ui/reopen-result.json").read_text(encoding="utf-8"))
    assert reopened["status"] == "completed" and reopened["environment_cache_hit"]
    assert reopened["environment_probe_count"] == 0
    assert reopened["configuration"] == ui["configuration"]
    assert reopened["method_parameters"] == ui["method_parameters"]
    help_result = subprocess.run([str(console), "--help"], env=env, capture_output=True, text=True, check=True)
    assert "usage: jingwei " in help_result.stdout and "jingwei-benchmark" not in help_result.stdout
    benchmark_help = subprocess.run([str(console), "benchmark", "--help"], env=env, capture_output=True, text=True, check=True)
    assert "all 35 methods" in benchmark_help.stdout
    assert "1800; 30 min" in " ".join(benchmark_help.stdout.split())
    default_inspection = subprocess.run([str(console), "inspect"], env=env, capture_output=True, text=True, check=True)
    default_data = json.loads(default_inspection.stdout)
    assert (default_data["samples"], default_data["features"]) == (210, 1542)
    fixture = Path(ui["job_folder"]) / "input.csv"
    methods = ["all"] if args.all_methods else ["mean", "bpca", "jingwei-dmf", "jingwei-dcae", "pimms-vae"]
    common = ["benchmark", "--input", fixture, "--repeats", "1", "--mask-rate", ".05", "--threads", "1",
              "--training-policy", "fixed", "--max-epochs", "1", "--batch-size", "4", "--latent-dim", "4", "--hidden-dims", "8", "--timeout", "180"]
    run("cli", *common, "--methods", *methods, "--device", "cpu", "--output", output / "cpu")
    import pandas as pd
    cpu = pd.read_csv(output / "cpu/metrics.csv")
    assert (cpu.status == "completed").all()
    updated = cpu[cpu.method == "jingwei-dcae"].iloc[0]
    assert updated.ae_architecture == "global" and updated.model_architecture_version == "global-ae-v1"
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
    impute_output = output / "impute-training-mask"
    run("cli", "impute", "--input", fixture, "--methods", "mean", "jingwei-dmf", "jingwei-dcae",
        "--mechanism", "mcar", "--mask-rate", ".25", "--device", "cpu", "--threads", "1",
        "--training-policy", "fixed", "--max-epochs", "2", "--batch-size", "4", "--latent-dim", "4", "--hidden-dims", "8",
        "--output", impute_output)
    import numpy as np
    imputed = json.loads((impute_output / "metrics.json").read_text(encoding="utf-8"))
    assert len(imputed) == 3 and all(row["status"] == "completed" for row in imputed)
    with np.load(impute_output / "repeat_000/masks.npz") as masks:
        assert not masks["evaluation_hidden"].any() and masks["training_hidden"].any()
        observed_count = int(masks["observed"].sum())
        hidden_count = int(masks["training_hidden"].sum())
    for method in ("jingwei-dmf", "jingwei-dcae"):
        detail = json.loads((impute_output / f"repeat_000/{method}/worker_result.json").read_text(encoding="utf-8"))
        assert detail["training_hidden_entries"] == hidden_count
        assert detail["supervision_entries_per_epoch"] == observed_count
        assert detail["training_input_entries"] == observed_count - hidden_count
    with np.load(impute_output / "repeat_000/method_input.npz") as common:
        assert not common["training_hidden"].any()
    original_input = pd.read_csv(fixture, index_col=0).to_numpy()
    for row in imputed:
        prediction = pd.read_csv(impute_output / f"repeat_000/{row['method']}/imputed.csv", index_col=0).to_numpy()
        np.testing.assert_allclose(prediction[np.isfinite(original_input)], original_input[np.isfinite(original_input)])
    # Exercise the actual default dataset through one shared original MNAR mask.
    # Its all-empty columns exposed the missing notebook preprocessing adapter.
    original_output = output / "alzheimer-original-mnar"
    run("cli", "benchmark", "--methods", "mean", "knn", "jingwei-dmf", "jingwei-dcae", "pimms-dae", "pimms-vae", "pimms-cf",
        "--mechanism", "mnar-jingwei-original", "--seed", "0", "--device", "cpu",
        "--training-policy", "fixed", "--max-epochs", "2", "--latent-dim", "8", "--hidden-dims", "32",
        "--batch-size", "32", "--threads", "2", "--output", original_output)
    import numpy as np
    original = pd.read_csv(original_output / "metrics.csv")
    assert len(original) == 7 and (original.status == "completed").all()
    with np.load(original_output / "repeat_000/masks.npz") as masks:
        available = masks["observed"] & ~masks["evaluation_hidden"]
        empty = ~available.any(axis=0)
        assert empty.any() and masks["training_hidden"].any()
        assert not (masks["training_hidden"] & masks["evaluation_hidden"]).any()
        excluded = int(masks["evaluation_hidden"][:, empty].sum())
        assert (original.zero_cols == int(empty.sum())).all()
        assert (original.n_excluded == excluded).all()
        assert (original.n_evaluated + original.n_excluded == original.n_held_out).all()
        assert original.n_evaluated.nunique() == 1
        for method in original.method:
            exported = pd.read_csv(original_output / f"repeat_000/{method}/imputed.csv", index_col=0).to_numpy()
            assert np.isnan(exported[:, empty]).all() and np.isfinite(exported[:, ~empty]).all()
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
        "public_console_validated": True, "alzheimer_original_mnar_completed": len(original),
        "platform_mask_count": 1, "full_default_input_checked": full_input_checked,
        "impute_training_mask_completed": len(imputed)}, indent=2), encoding="utf-8")
    print("Frozen desktop UI and backend validation passed")


if __name__ == "__main__":
    main()
