"""Wrap the Windows offline distribution in a per-user Inno Setup installer."""

import argparse
import hashlib
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
    parser.add_argument("--source", type=Path, default=APPLICATION / "dist/windows-x86_64/jingwei")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single-file", dest="single_file", action="store_true", default=True,
                      help="complete offline EXE (default); no payload downloads during installation")
    mode.add_argument("--split", dest="single_file", action="store_false",
                      help="legacy EXE/BIN payload for the optional download installer")
    parser.add_argument("--allow-large-file", action="store_true",
                        help="allow a local offline EXE over GitHub's 2 GiB asset limit")
    parser.add_argument("--dictionary-mib", type=int, choices=(64, 128, 256, 512, 1024), default=512,
                        help="offline LZMA2 dictionary size in MiB; larger values use more builder/install RAM")
    parser.add_argument("--compression-block-threads", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("build the Windows installer on Windows")
    from jingwei_benchmark import __version__
    source = args.source.resolve()
    output = (args.output or APPLICATION / f"dist/releases/v{__version__}").resolve()
    info = json.loads((source / "build-info.json").read_text(encoding="utf-8"))
    if info["app_version"] != __version__:
        parser.error("rebuild the Windows application before creating the installer")
    for name in ("jingwei.exe", "jingwei-cli.exe", "_internal"):
        if not (source / name).exists():
            parser.error(f"missing application component: {name}")
    from PIL import Image
    with Image.open(ROOT / "figures/Jingwei_logo.png") as logo:
        logo.convert("RGBA").save(source / "jingwei.ico", format="ICO", sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)])
    output.mkdir(parents=True, exist_ok=True)
    options = (["/DSingleFileInstaller", f"/DInstallerDictionaryKB={args.dictionary_mib * 1024}",
                f"/DInstallerBlockThreads={args.compression_block_threads}"] if args.single_file else [])
    subprocess.run([str(args.iscc.resolve()), *options, f"/DSourceDir={source}", f"/DReleaseDir={output}", f"/DAppVersion={__version__}",
                    str(BUILD / "windows-installer.iss")], check=True)
    prefix = f"jingwei-{__version__}-windows-x86_64-" + ("offline" if args.single_file else "setup")
    artifacts = [output / (prefix + ".exe")] if args.single_file else sorted(output.glob(prefix + "*"))
    if not artifacts or not all(p.is_file() for p in artifacts):
        raise RuntimeError("installer output is missing")
    if not args.allow_large_file and any(p.stat().st_size >= 2 * 1024**3 for p in artifacts):
        raise RuntimeError("compiled installer exceeds GitHub's 2 GiB per-asset limit; "
                           "the local output is retained but must not be uploaded there")
    if args.single_file:
        installer = artifacts[0]
        with installer.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        (output / "windows-offline-installer.json").write_text(json.dumps({
            "version": __version__, "installer": installer.name,
            "size": installer.stat().st_size, "sha256": sha,
            "installed_size": sum(p.stat().st_size for p in source.rglob("*") if p.is_file()),
            "offline": True, "payload_downloads": False,
            "destination_editable": True, "destination_browse": True,
            "github_asset_size_valid": installer.stat().st_size < 2 * 1024**3,
            "compression": {"method": "lzma2/ultra64", "dictionary_mib": args.dictionary_mib,
                            "block_threads": args.compression_block_threads},
        }, indent=2), encoding="utf-8")
    print("Installer files:", *(str(p) for p in artifacts), sep="\n")


if __name__ == "__main__":
    main()
