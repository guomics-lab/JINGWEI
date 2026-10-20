"""Split the offline Linux archive into GitHub-sized assets and write checksums."""

import argparse
import hashlib
from pathlib import Path
import shutil

BUILD = Path(__file__).resolve().parent
APPLICATION = BUILD.parent
CHUNK_SIZE = 1_900_000_000


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    from jingwei_benchmark import __version__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=APPLICATION / "dist/jingwei-linux-x86_64.tar")
    parser.add_argument("--output", type=Path, default=APPLICATION / f"dist/releases/v{__version__}")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"jingwei-{__version__}-linux-x86_64"
    parts = []
    whole = hashlib.sha256()
    snapshot = args.archive.stat()
    with args.archive.open("rb") as source:
        number = 1
        while block := source.read(8 * 1024**2):
            part = output / f"{prefix}.tar.part{number:03}"
            checksum = hashlib.sha256()
            written = 0
            with part.open("wb") as target:
                while block:
                    target.write(block)
                    whole.update(block)
                    checksum.update(block)
                    written += len(block)
                    if written == CHUNK_SIZE:
                        break
                    block = source.read(min(8 * 1024**2, CHUNK_SIZE - written))
            parts.append((part.name, checksum.hexdigest()))
            number += 1
    current = args.archive.stat()
    if (current.st_size, current.st_mtime_ns) != (snapshot.st_size, snapshot.st_mtime_ns):
        raise RuntimeError("Archive changed during splitting; finish copying/building it before preparing release assets")
    (output / f"{prefix}.SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in parts), encoding="utf-8", newline="\n")
    installer = (BUILD / "install-linux.sh.in").read_text(encoding="utf-8")
    installer = installer.replace("@PART_FILES@", " ".join(f'"{name}"' for name, _ in parts))
    installer = installer.replace("@ARCHIVE_SHA256@", whole.hexdigest())
    installer = installer.replace("@VERSION@", __version__)
    (output / "install-linux.sh").write_text(installer, encoding="utf-8", newline="\n")
    guide = APPLICATION / f"RELEASE-{__version__}.md"
    shutil.copyfile(guide if guide.exists() else APPLICATION / "RELEASE-1.0.md", output / f"INSTALL-{__version__}.md")
    shutil.copyfile(APPLICATION / "LICENSE.txt", output / "LICENSE.txt")
    for name in ("SE_TESTS.md",):
        shutil.copyfile(APPLICATION / name, output / name)
    print("Linux parts:", *(name for name, _ in parts), sep="\n")


if __name__ == "__main__":
    main()
