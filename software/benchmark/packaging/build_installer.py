"""Wrap the Windows offline distribution in a per-user Inno Setup installer."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

BUILD = Path(__file__).resolve().parent
APPLICATION = BUILD.parent
ROOT = BUILD.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=APPLICATION / "dist/windows-x86_64/JINGWEI Benchmark")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("build the Windows installer on Windows")
    from jingwei_benchmark import __version__
    source = args.source.resolve()
    output = (args.output or APPLICATION / f"dist/releases/v{__version__}").resolve()
    info = json.loads((source / "build-info.json").read_text(encoding="utf-8"))
    if info["app_version"] != __version__:
        parser.error("rebuild the Windows application before creating the installer")
    for name in ("JINGWEI Benchmark.exe", "jingwei-benchmark.exe", "_internal"):
        if not (source / name).exists():
            parser.error(f"missing application component: {name}")
    from PIL import Image
    with Image.open(ROOT / "figures/Jingwei_logo.png") as logo:
        logo.convert("RGBA").save(source / "jingwei.ico", format="ICO", sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)])
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.iscc.resolve()), f"/DSourceDir={source}", f"/DReleaseDir={output}", f"/DAppVersion={__version__}",
                    str(BUILD / "windows-installer.iss")], check=True)
    artifacts = list(output.glob(f"JINGWEI-Benchmark-{__version__}-windows-x86_64-setup*"))
    if not artifacts or any(p.stat().st_size >= 2 * 1024**3 for p in artifacts):
        raise RuntimeError("installer missing or exceeds GitHub's per-asset size limit")
    print("Installer files:", *(str(p) for p in artifacts), sep="\n")


if __name__ == "__main__":
    main()
