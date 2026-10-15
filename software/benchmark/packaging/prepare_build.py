"""Provision Python and original method sources INSIDE jingwei_benchmark."""

import argparse
import io
import platform
import os
import shutil
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parents[2]
SOURCES = {
    "NAguideR": ("wangshisheng/NAguideR", "ae26a5344c5a042e066ea3e127132a6be4bfb7a4"),
    "DreamAI": ("WangLab-MSSM/DreamAI", "2cc86d1487227e8215a716459a984b49c4aaac84"),
}


def run(*args, env=None):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True, env=env)


def sources():
    for name, (repo, commit) in SOURCES.items():
        target = ROOT / ".method-sources" / name
        if target.exists():
            print(f"Using existing method source: {target}", flush=True)
            continue
        url = f"https://codeload.github.com/{repo}/zip/{commit}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.infolist():
            parts = Path(member.filename).parts[1:]
            if not parts:
                continue
            path = target.joinpath(*parts).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("unsafe source archive member")
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))
        (target / "JINGWEI_SOURCE_COMMIT").write_text(commit + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-build", choices=("cpu",) if sys.platform == "darwin"
                        else ("cpu", "cu128"), default="cpu")
    parser.add_argument("--rscript", help="Windows Rscript path, or override conda R")
    parser.add_argument("--r-library", help="optional platform-specific R library")
    parser.add_argument("--skip-r", action="store_true", help="install Python methods only")
    args = parser.parse_args()
    if not (Path(sys.prefix) / "conda-meta").is_dir() or Path(sys.prefix).name != "jingwei_benchmark":
        parser.error("activate the dedicated conda environment jingwei_benchmark first")
    wheel_args = [] if sys.platform == "darwin" else [
        "--index-url", f"https://download.pytorch.org/whl/{args.torch_build}"]
    intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    suffix = "" if sys.platform == "darwin" else "+" + args.torch_build
    if intel_mac:
        run(sys.executable, "-m", "pip", "install", "numpy>=1.26,<2")
    run(sys.executable, "-m", "pip", "install",
        "torch==2.2.2" if intel_mac else "torch==2.7.1" + suffix,
        "torchvision==0.17.2" if intel_mac else "torchvision==0.22.1" + suffix, *wheel_args)
    run(sys.executable, "-m", "pip", "install", "-r", BUILD / "requirements-build.txt")
    run(sys.executable, "-m", "pip", "install", "--no-deps", "-e", BUILD.parent)
    sources()
    if not args.skip_r:
        from jingwei_benchmark.methods import find_rscript, r_environment
        rscript = find_rscript(args.rscript)
        if not rscript:
            parser.error("Rscript missing: install conda POSIX R profile or supply --rscript on Windows")
        configured = {}
        if args.rscript:
            configured["JINGWEI_RSCRIPT"] = rscript
        if args.r_library:
            args.r_library = str(Path(args.r_library).resolve())
            configured["JINGWEI_R_LIBRARY"] = args.r_library
        if configured:
            conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
            if not conda:
                parser.error("activate conda before saving custom R backend paths")
            run(conda, "env", "config", "vars", "set", "--prefix", sys.prefix,
                *(f"{key}={value}" for key, value in configured.items()))
            os.environ.update(configured)
        command = [rscript, "--vanilla", BUILD / "install_backends.R", ROOT]
        if args.r_library:
            command.append(args.r_library)
        run(*command, env=r_environment(args.r_library))
    run(sys.executable, "-m", "pip", "check")
    run(sys.executable, "-m", "jingwei_benchmark", "methods")
    from jingwei_benchmark.methods import availability, find_method_source
    states = availability(args.rscript, args.r_library, find_method_source("NAguideR"),
                          find_method_source("DreamAI"))
    missing = [name for name, state in states.items() if not state["available"]
               and not (args.skip_r and state["backend"] == "r")]
    if missing:
        raise RuntimeError("Backend installation incomplete: " + ", ".join(missing))


if __name__ == "__main__":
    main()
