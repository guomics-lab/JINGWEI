"""Build the native executable on its target OS from jingwei_benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parents[2]
APPLICATION = BUILD.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=APPLICATION / "dist/windows-x86_64")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows builds require Windows; use build_posix.py on Linux/macOS")
    if not (Path(sys.prefix) / "conda-meta").is_dir() or Path(sys.prefix).name != "jingwei_benchmark":
        parser.error("build inside the dedicated jingwei_benchmark conda environment")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (APPLICATION / "LICENSE.txt").is_file():
        parser.error("missing benchmark application LICENSE")
    from jingwei_benchmark.methods import availability, find_method_source
    from jingwei_benchmark import __version__
    from jingwei_benchmark.cli import DEFAULT_TIMEOUT_SECONDS
    states = availability(source_dir=find_method_source("NAguideR"), dreamai_source=find_method_source("DreamAI"))
    if missing := [name for name, state in states.items() if not state["available"]]:
        parser.error("Build environment is incomplete: " + ", ".join(missing))
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(output),
                    "--workpath", str(ROOT / "build/benchmark-windows"), str(BUILD / "windows.spec")],
                   env=env, cwd=ROOT, check=True)
    artifact = output / "jingwei/jingwei.exe"
    console = output / "jingwei/jingwei-cli.exe"
    distribution = artifact.parent
    for name in ("README.md", "PERFORMANCE.md", "SE_TESTS.md", "AE_ARCHITECTURE.md", f"RELEASE-{__version__}.md"):
        shutil.copyfile(APPLICATION / name, distribution / name)
    shutil.copyfile(APPLICATION / "LICENSE.txt", distribution / "LICENSE.txt")
    for name in ("NAguideR", "DreamAI"):
        source = Path(find_method_source(name))
        for file in source.iterdir():
            if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING")):
                target = distribution / "licenses" / name / file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "fastai", "pimms-learn", "plum-dispatch", "scikit-learn", "PySide6-Essentials", "pyinstaller", "pandas", "numpy", "scipy", "anndata")}
    with artifact.open("rb") as handle:
        executable_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    from jingwei_benchmark import __version__
    info = {"app_version": __version__, "platform": sys.platform, "python": sys.version, "versions": versions,
            "app_name": "jingwei", "desktop_executable": "jingwei.exe", "console_executable": "jingwei-cli.exe",
            "startup_environment_check": "before-main-window", "run_settings": ["command", "mechanism", "mask_rate"],
            "bundled_windows_r": True,
            "desktop_languages": ["zh", "en"],
            "app_user_model_id": "JINGWEI.Benchmark.Desktop",
            "default_input": "data/Alzheimer.csv",
            "default_jingwei_engine": "torch",
            "default_neural_training_policy": "fixed",
            "default_neural_epochs": 500,
            "neural_early_stopping": False,
            "default_dmf_architecture": "bilinear-residual-v1",
            "dmf_interaction_rank": 8,
            "pimms_cf_entry_batch_size": 4096,
            "default_baseline_engine": "optimized",
            "default_benchmark_methods": "all",
            "default_benchmark_repeats": 1, "default_gui_protocol": "fair",
            "mask_protocol_version": 4, "platform_mask_count": 1,
            "training_protocol": "one fixed JINGWEI-only denoising mask; known targets retained",
            "imputation_training_mask": "jingwei-only; selected mechanism and rate; original targets retained",
            "default_dcae_architecture": "global-ae-v1",
            "jingwei_mnar_mechanism": "mnar-jingwei-v2",
            "mask_classification_loss": "full input mask BCE; reconstruction on available targets only",
            "notebook_source_revision": "98686b54fdb9918a4d4cc9d99ac4880a6dadcefe",
            "default_timeout_s": DEFAULT_TIMEOUT_SECONDS,
            "default_mice_backend": "python-linear-mice-pmm",
            "console_sha256": hashlib.sha256(console.read_bytes()).hexdigest(),
            "macos_runtime_validated": False,
            "exe_sha256": executable_hash}
    for target in (output / "build-info.json", distribution / "build-info.json"):
        target.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Native desktop build: {distribution}")


if __name__ == "__main__":
    main()
