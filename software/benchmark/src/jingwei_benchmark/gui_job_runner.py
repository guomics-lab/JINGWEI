"""Run one durable desktop job without loading Qt or receiving GUI events."""

import sys
from pathlib import Path
import traceback

from .gui_jobs import read_json, write_json
from .cli import main as cli_main


def main():
    from .runtime import prepare_standard_streams
    from .process_guard import finalize_worker_exit, start_parent_watchdog
    prepare_standard_streams()
    start_parent_watchdog()
    request_file = Path(sys.argv[1])
    try:
        request = read_json(request_file)
        if request.get("schema_version") != 1:
            raise ValueError("Unsupported desktop job schema")
        code = cli_main(request["argv"])
    except BaseException:
        traceback.print_exc()
        code = 1
    write_json(request_file.parent / "completion.json", {"return_code": code})
    return finalize_worker_exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
