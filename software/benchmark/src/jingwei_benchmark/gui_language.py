"""Use the installation language once, then remember the user's selection."""

import hashlib
from pathlib import Path
import sys

from .atomic import write_json


def _preferences(workspace):
    import json
    try:
        value = json.loads((Path(workspace) / "ui.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _installation():
    from .runtime import resource_root
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else resource_root()
    path = root / "installation-language.txt"
    try:
        content = path.read_bytes()
        name = content.decode("utf-8-sig").splitlines()[0].strip()
        language = {"english": "en", "chinesesimp": "zh"}.get(name)
        if language:
            stamp = hashlib.sha256(content + str(path.stat().st_mtime_ns).encode("ascii")
                                   + str(path.resolve()).encode("utf-8")).hexdigest()
            return language, stamp
    except (OSError, ValueError, IndexError):
        pass
    return None


def initial_language(workspace):
    preferences = _preferences(workspace)
    installed = _installation()
    if installed and preferences.get("installation_language") != installed[1]:
        preferences.update(language=installed[0], installation_language=installed[1])
        write_json(Path(workspace) / "ui.json", preferences)
    language = preferences.get("language", "zh")
    return language if language in ("zh", "en") else "zh"


def remember_language(workspace, language):
    if language not in ("zh", "en"):
        raise ValueError("unsupported desktop language")
    preferences = _preferences(workspace)
    preferences["language"] = language
    installed = _installation()
    if installed:
        preferences["installation_language"] = installed[1]
    write_json(Path(workspace) / "ui.json", preferences)
