"""Build an offline CLI distribution containing Python, PyTorch, R and all methods."""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

BUILD = Path(__file__).resolve().parent
APPLICATION = BUILD.parent
ROOT = BUILD.parents[2]


def release_member(member):
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mode = 0o755 if member.isdir() or Path(member.name).name == "jingwei" else 0o644
    return member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        parser.error("build Linux/macOS CLI releases on their target OS")
    if not (APPLICATION / "LICENSE.txt").is_file():
        parser.error("missing benchmark application LICENSE")
    prefix = Path(sys.prefix)
    if prefix.name != "jingwei_benchmark" or not (prefix / "conda-meta").is_dir():
        parser.error("use the dedicated jingwei_benchmark conda build environment")
    from conda_pack import CondaEnv, File
    from jingwei_benchmark.methods import availability, find_method_source
    from jingwei_benchmark import __version__
    from jingwei_benchmark.cli import DEFAULT_TIMEOUT_SECONDS
    sources = {name: find_method_source(name) for name in ("NAguideR", "DreamAI")}
    rscript = prefix / "bin/Rscript"
    rlibrary = prefix / "lib/R/library"
    if not rscript.is_file() or not rlibrary.is_dir():
        parser.error("the build environment must contain R and all R backends")
    states = availability(str(rscript), str(rlibrary), sources["NAguideR"], sources["DreamAI"])
    if missing := [name for name, state in states.items() if not state["available"]]:
        parser.error("incomplete build environment: " + ", ".join(missing))
    system, machine = platform.system(), platform.machine()
    target = ("macos" if sys.platform == "darwin" else "linux") + "-" + machine
    output = (args.output or APPLICATION / "dist" / target).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "runtime.tar.gz"
    if archive.exists():
        parser.error(f"output already contains a runtime; choose a fresh output: {output}")
    env = CondaEnv.from_prefix(str(prefix), ignore_editable_packages=True)
    # No Qt or unrelated web-server runtime is shipped in the CLI application.
    excluded = ["*/__pycache__/*", "*/PySide6/*", "*/shiboken6/*",
                "*/PySide6*.dist-info/*", "*/pyside6*.dist-info/*", "*/shiboken6*.dist-info/*",
                "bin/pyside6*", "*/__editable__*", "*/jingwei-*.dist-info/*",
                "*/jingwei_benchmark-*.dist-info/*", "*/jingwei_benchmark/*", "*/jingwei/*",
                "*/fastapi/*", "*/uvicorn/*", "*/starlette/*",
                "*/fastapi-*.dist-info/*", "*/uvicorn-*.dist-info/*", "*/starlette-*.dist-info/*"]
    for pattern in excluded:
        env = env.exclude(pattern)
    purelib = Path(sysconfig.get_path("purelib")).relative_to(prefix)
    extra = [File(str(ROOT / "data/Alzheimer.csv"), "share/jingwei/data/Alzheimer.csv", is_conda=False)]
    with tempfile.TemporaryDirectory(prefix="jingwei-release-metadata-") as temporary:
        staging = Path(temporary)
        # Convert the private editable application into real files; its native
        # network adapters are application-local, separate from the public package.
        for name in ("jingwei_benchmark",):
            spec = importlib.util.find_spec(name)
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError(f"missing installed application dependency: {name}")
            folder = Path(next(iter(spec.submodule_search_locations)))
            for file in folder.rglob("*"):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    # The POSIX product has no GUI modules or assets.
                    if name == "jingwei_benchmark" and (file.name.startswith(("gui", "desktop", "console_entry")) or file.stem == "windows_identity" or "assets" in file.parts):
                        continue
                    extra.append(File(str(file), str(purelib / name / file.relative_to(folder)), is_conda=False))
            dist_name = name.replace("_", "-")
            version = metadata.version(dist_name)
            info = staging / f"{name}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n", encoding="utf-8")
            (info / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: JINGWEI standalone builder\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8")
            for file in info.iterdir():
                extra.append(File(str(file), str(purelib / info.name / file.name), is_conda=False))
        required = {"NAguideR": "inst/NAguideRapp/Trunc_KNN/Imput_funcs.r", "DreamAI": "Code/R/RegImpute.R"}
        for name, relative in required.items():
            file = Path(sources[name]) / relative
            extra.append(File(str(file), f"share/jingwei/.method-sources/{name}/{relative}", is_conda=False))
            for file in Path(sources[name]).glob("*"):
                if file.is_file() and file.name.upper().startswith(("LICENSE", "COPYING")):
                    extra.append(File(str(file), f"share/jingwei/licenses/{name}/{file.name}", is_conda=False))
        files = [*env.files, *extra]
        if any("/PySide6/" in f.target or "/shiboken6/" in f.target for f in files):
            raise RuntimeError("Qt unexpectedly entered the POSIX CLI runtime")
        print(f"Packing {len(files)} runtime files for {target}; this includes all R packages", flush=True)
        CondaEnv(str(prefix), files).pack(str(archive), format="tar.gz", compress_level=2, n_threads=4, verbose=True)
    with archive.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    launcher = (BUILD / "launcher.sh.in").read_text(encoding="utf-8")
    for key, value in {"SYSTEM": system, "MACHINE": machine, "SHA256": sha,
                       "RUNTIME_ID": target + "-" + sha[:20]}.items():
        launcher = launcher.replace("@" + key + "@", value)
    executable = output / "jingwei"
    executable.write_text(launcher, encoding="utf-8", newline="\n")
    executable.chmod(0o755)
    shutil.copyfile(BUILD / "fix_r_runtime.py", output / "fix_r_runtime.py")
    for name in ("README.md", "PERFORMANCE.md", "SE_TESTS.md", f"RELEASE-{__version__}.md"):
        shutil.copyfile(APPLICATION / name, output / name)
    shutil.copyfile(APPLICATION / "LICENSE.txt", output / "LICENSE.txt")
    versions = {name: metadata.version(name) for name in ("torch", "fastai", "pimms-learn", "scikit-learn", "numpy", "scipy", "pandas", "anndata")}
    r_version = subprocess.run([str(rscript), "--version"], capture_output=True, text=True, check=True)
    from jingwei_benchmark import __version__
    info = {"app_version": __version__, "app_name": "jingwei", "console_executable": "jingwei",
            "platform": target, "python": sys.version, "versions": versions,
            "runtime_sha256": sha, "bundled_r": True, "r_version": r_version.stdout.strip(),
            "gui_included": False, "build_libc": platform.libc_ver(), "method_count": len(states),
            "default_baseline_engine": "optimized",
            "default_neural_training_policy": "convergence",
            "neural_safety_epoch_cap": 5000,
            "neural_validation_fraction": .1,
            "neural_patience": 30,
            "pimms_cf_entry_batch_size": 4096,
            "default_benchmark_methods": "all",
            "default_benchmark_repeats": 1,
            "jingwei_mnar_mechanism": "mnar-jingwei-v2",
            "mask_classification_loss": "full input mask BCE; reconstruction on available targets only",
            "imputation_training_mask": "jingwei-only; selected mechanism and rate; original targets retained",
            "notebook_source_revision": "98686b54fdb9918a4d4cc9d99ac4880a6dadcefe",
            "default_timeout_s": DEFAULT_TIMEOUT_SECONDS,
            "default_mice_backend": "python-linear-mice-pmm",
            "macos_runtime_validated": False}
    (output / "build-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # The runtime is already compressed; avoid recompressing several GB twice.
    release = output.parent / f"jingwei-{target}.tar"
    with tarfile.open(release, "w") as bundle:
        bundle.add(output, arcname=f"jingwei-{target}", filter=release_member)
    print(f"Standalone CLI: {executable}\nRelease archive: {release}", flush=True)


if __name__ == "__main__":
    main()
