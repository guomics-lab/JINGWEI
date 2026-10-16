"""Atomic JSON updates that tolerate transient Windows reader/antivirus locks."""

import json
import os
from pathlib import Path
import tempfile
import time


def write_json(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        for attempt in range(7):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 6:
                    raise
                time.sleep(.01 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)
