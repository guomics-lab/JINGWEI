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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=APPLICATION / "dist/jingwei-benchmark-linux-x86_64.tar")
    parser.add_argument("--output", type=Path, default=APPLICATION / "dist/releases/v1.0.0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = "jingwei-benchmark-1.0.0-linux-x86_64"
    parts = []
    whole = hashlib.sha256()
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
    (output / f"{prefix}.SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in parts), encoding="utf-8", newline="\n")
    installer = (BUILD / "install-linux.sh.in").read_text(encoding="utf-8")
    installer = installer.replace("@PART_FILES@", " ".join(f'"{name}"' for name, _ in parts))
    installer = installer.replace("@ARCHIVE_SHA256@", whole.hexdigest())
    (output / "install-linux.sh").write_text(installer, encoding="utf-8", newline="\n")
    shutil.copyfile(APPLICATION / "RELEASE-1.0.md", output / "INSTALL-1.0.md")
    shutil.copyfile(APPLICATION / "LICENSE", output / "LICENSE.txt")
    print("Linux parts:", *(name for name, _ in parts), sep="\n")


if __name__ == "__main__":
    main()
