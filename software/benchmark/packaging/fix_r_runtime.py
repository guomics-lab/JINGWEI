"""Quote conda R wrapper paths after offline prefix relocation (including spaces)."""

import os
from pathlib import Path
import re
import shlex
import sys
import tempfile


def repair(prefix):
    prefix = Path(prefix).resolve()
    paths = {
        "R_HOME_DIR": prefix / "lib/R",
        "R_SHARE_DIR": prefix / "lib/R/share",
        "R_INCLUDE_DIR": prefix / "lib/R/include",
        "R_DOC_DIR": prefix / "lib/R/doc",
        "SED": prefix / "bin/sed",
    }
    seen = set()
    for candidate in (prefix / "bin/R", prefix / "lib/R/bin/R"):
        if not candidate.is_file():
            continue
        path = candidate.resolve()
        if not path.is_relative_to(prefix):
            raise ValueError("R wrapper must stay inside the bundled runtime")
        if path in seen:
            continue
        seen.add(path)
        original = path.read_text(encoding="utf-8")
        lines = []
        for line in original.splitlines(keepends=True):
            key, _, value = line.rstrip("\r\n").partition("=")
            if key in paths:
                line = key + "=" + shlex.quote(str(paths[key])) + "\n"
            else:
                line = re.sub(r'''(?<!["'])\$\{SED\}(?!["'])''', '"${SED}"', line)
            lines.append(line)
        fixed = "".join(lines)
        if fixed != original:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".jingwei-", delete=False) as handle:
                handle.write(fixed)
                temporary = Path(handle.name)
            try:
                temporary.chmod(path.stat().st_mode)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    repair(sys.argv[1])
