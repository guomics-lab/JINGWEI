"""Validate the actual offline CLI release outside the repository and host environments."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    executable, output = args.executable.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in os.environ.items()
             if not key.startswith(("CONDA", "R_", "JINGWEI_", "PYTHON"))
             and key not in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LD_PRELOAD")}
    clean["PATH"] = "/usr/bin:/bin"
    clean["JINGWEI_BENCHMARK_CACHE"] = str(output / "cache with spaces")
    clean["MPLBACKEND"] = "Agg"
    log = output / "validation.log"
    with tempfile.TemporaryDirectory(prefix="jingwei-outside-checkout-") as cwd:
        def run(*argv, timeout=1200):
            completed = subprocess.run([str(executable), *map(str, argv)], cwd=cwd,
                env=clean, capture_output=True, text=True, timeout=timeout)
            with log.open("a", encoding="utf-8") as handle:
                handle.write("COMMAND " + " ".join(map(str, argv)) + "\n" + completed.stdout + completed.stderr + "\n")
            if completed.returncode:
                raise RuntimeError(f"Standalone CLI failed ({completed.returncode}); see {log}")
            return completed.stdout
        help_text = run("--help")
        assert "gui" not in help_text and "benchmark" in help_text
        benchmark_help = run("benchmark", "--help")
        assert "all 35 methods" in benchmark_help and "1800; 30 min" in " ".join(benchmark_help.split())
        default_data = json.loads(run("inspect"))
        assert (default_data["samples"], default_data["features"]) == (210, 1542)
        states = json.loads(run("methods", "--json"))
        assert len(states) == 35 and all(state["available"] for state in states.values()), states
        runtime = next((output / "cache with spaces").glob("*/.ready")).parent
        check = subprocess.run([str(runtime / "bin/python"), "-I", "-c",
            "import importlib.util,jingwei_benchmark; assert importlib.util.find_spec('PySide6') is None; print(jingwei_benchmark.__file__)"],
            env=clean, cwd=cwd, capture_output=True, text=True, check=True)
        assert str(runtime) in check.stdout
        source = output / "input.csv"
        rng = random.Random(42)
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", *[f"P{i:03d}" for i in range(80)]])
            for i in range(16):
                row = [math.exp(rng.gauss(8, .6)) for _ in range(80)]
                if i >= 8:
                    for j in range(80):
                        if rng.random() < .04:
                            row[j] = ""
                writer.writerow([f"S{i:02d}", *row])
        common = ["benchmark", "--input", source, "--repeats", "1", "--mask-rate", ".05",
                  "--threads", "1", "--max-epochs", "1", "--batch-size", "4", "--latent-dim", "4",
                  "--hidden-dims", "8", "--timeout", "180"]
        methods = ["all"] if args.all_methods else ["mean", "bpca", "jingwei-dmf", "pimms-vae"]
        run(*common, "--methods", *methods, "--device", "cpu", "--output", output / "cpu")
        with (output / "cpu/metrics.csv").open() as handle:
            cpu = list(csv.DictReader(handle))
        assert len(cpu) == (35 if args.all_methods else 4) and all(row["status"] == "completed" for row in cpu)
        if args.all_methods:
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
        gpu = []
        if args.cuda:
            neural = ["jingwei-dmf", "jingwei-dcae", "pimms-dae", "pimms-vae", "pimms-cf"]
            run(*common, "--methods", *neural, "--device", "cuda", "--gpu-id", "0", "--output", output / "cuda")
            with (output / "cuda/metrics.csv").open() as handle:
                gpu = list(csv.DictReader(handle))
            assert len(gpu) == 5 and all(row["status"] == "completed" and row["actual_device"].startswith("cuda") for row in gpu)
        full_input_checked = False
        if args.all_methods:
            full_output = output / "full-input-check"
            checked = subprocess.run([str(executable), "benchmark", "--methods", "seqknn", "impseq", "mle", "qrilc", "mice-pmm",
                "--repeats", "1", "--device", "cpu", "--output", str(full_output)], cwd=cwd, env=clean,
                capture_output=True, text=True, timeout=300)
            with log.open("a", encoding="utf-8") as handle:
                handle.write("FULL DEFAULT INPUT CHECK\n"+checked.stdout+checked.stderr+"\n")
            assert checked.returncode == 2
            full = json.loads((full_output / "metrics.json").read_text(encoding="utf-8"))
            assert json.loads((full_output / "manifest.json").read_text(encoding="utf-8"))["config"]["timeout"] == 1800
            by_name = {row["method"]: row for row in full}
            for name in ("seqknn", "impseq", "mle"):
                assert by_name[name]["status"] == "failed" and by_name[name]["failure_kind"] == "unsupported_input"
            assert by_name["qrilc"]["status"] == "completed" and by_name["qrilc"]["n_evaluated"] > 0
            assert by_name["mice-pmm"]["status"] == "completed" and by_name["mice-pmm"]["n_evaluated"] > 0
            full_input_checked = True
        result = {"method_count": len(states), "cpu_completed": len(cpu), "cuda_completed": len(gpu),
                  "bundled_default_data": "Alzheimer.csv", "default_shape": [210, 1542],
                  "gui_included": False, "runtime": str(runtime), "host_paths_removed": True,
                  "outside_checkout": True, "cache_path_contains_spaces": True,
                  "full_default_input_checked": full_input_checked}
        (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("Standalone offline CLI validation passed", flush=True)


if __name__ == "__main__":
    main()
