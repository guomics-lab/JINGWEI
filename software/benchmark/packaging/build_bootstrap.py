"""Build the single-download Windows installer, pinning every payload checksum."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse

BUILD = Path(__file__).resolve().parent
APPLICATION = BUILD.parent
ROOT = BUILD.parents[2]


def main():
    from jingwei_benchmark import __version__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", required=True, type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--icon", required=True, type=Path)
    parser.add_argument("--base-url", default=f"https://github.com/guomics-lab/JINGWEI/releases/download/v{__version__}")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-server", action="store_true", help="allow a loopback HTTP fixture; never publish this build")
    args = parser.parse_args()
    url = urlparse(args.base_url)
    if not (url.scheme == "https" and url.hostname == "github.com"):
        if not (args.test_server and url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1")):
            parser.error("production payloads must use HTTPS on github.com")
    payload = (args.payload or APPLICATION / f"dist/releases/v{__version__}").resolve()
    output = (args.output or payload).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = ROOT / "build" / ("benchmark-bootstrap-test" if args.test_server else "benchmark-bootstrap")
    staging.mkdir(parents=True, exist_ok=True)
    prefix = f"JINGWEI-Benchmark-{__version__}-windows-x86_64-setup"
    files = sorted(payload.glob(prefix + "-*.bin")) + [payload / (prefix + ".exe")]
    if len(files) < 2 or not all(p.is_file() for p in files):
        parser.error("build the offline installer first")
    manifest, entries, checks, cleanup = [], [], [], []
    for file in files:
        with file.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        size = file.stat().st_size
        source = args.base_url.rstrip("/") + "/" + file.name
        manifest.append({"name": file.name, "size": size, "sha256": sha, "url": source})
        entries.append(f'Source: "{source}"; DestDir: "{{code:DownloadCache}}"; DestName: "{file.name}"; '
                       f'ExternalSize: {size}; Hash: "{sha}"; Flags: external download ignoreversion; '
                       f"Check: NeedDownload('{file.name}', '{sha}')")
        checks.append(f"  CheckPayload('{file.name}', '{sha}');")
        cleanup.append(f"  DeleteFile(AddBackslash(DownloadCache('')) + '{file.name}');")
    include, verify = staging / "payload.iss", staging / "verify.iss"
    include.write_text("\n".join(entries) + "\n", encoding="utf-8")
    verify.write_text("procedure VerifyPayload;\nbegin\n" + "\n".join(checks) + "\nend;\n\n"
                      "procedure DeletePayload;\nbegin\n" + "\n".join(cleanup) + "\nend;\n", encoding="utf-8")
    subprocess.run([str(args.iscc.resolve()), f"/DAppVersion={__version__}", f"/DReleaseDir={output}",
                    f"/DAppIcon={args.icon.resolve()}", f"/DPayloadExe={prefix}.exe",
                    f"/DPayloadInclude={include}", f"/DVerifyInclude={verify}", str(BUILD / "windows-bootstrap.iss")], check=True)
    (output / "windows-installer-manifest.json").write_text(
        json.dumps({"version": __version__, "test_only": args.test_server, "files": manifest}, indent=2), encoding="utf-8")
    print("Windows single-download installer:", output / "JINGWEI-Benchmark-Windows-Setup.exe")


if __name__ == "__main__":
    main()
