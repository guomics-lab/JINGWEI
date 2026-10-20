"""Entry point for windowed executables, including their isolated workers."""

import multiprocessing
import os
import sys
from pathlib import Path


def main(*, desktop=None):
    if desktop is None:
        desktop = not (getattr(sys, "frozen", False) and Path(sys.executable).stem.lower() in ("jingwei-benchmark", "jingwei-cli"))
    multiprocessing.freeze_support()
    from jingwei_benchmark.runtime import prepare_standard_streams
    prepare_standard_streams()
    if len(sys.argv) >= 3 and sys.argv[1] == "--jingwei-internal":
        module, argv = sys.argv[2], sys.argv[3:]
        if module == "cli":
            from jingwei_benchmark.cli import main as cli_main
            return cli_main(argv)
        if module in ("worker", "gui_job_runner", "gui_environment", "desktop_smoke"):
            sys.argv = [module, *argv]
            if module == "worker":
                from jingwei_benchmark.worker import main as run
            elif module == "gui_job_runner":
                from jingwei_benchmark.gui_job_runner import main as run
            elif module == "gui_environment":
                from jingwei_benchmark.gui_environment import main as run
            else:
                from jingwei_benchmark.desktop_smoke import main as run
            return run()
        raise ValueError("unknown internal entry point")
    if not desktop:
        from jingwei_benchmark.cli import main as cli_main
        return cli_main(sys.argv[1:])
    if sys.platform != "win32":
        raise RuntimeError("The benchmark GUI is available on Windows only; use jingwei-benchmark CLI")
    from jingwei_benchmark.gui import launch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace")
    args = parser.parse_args()
    return launch(args.workspace)


if __name__ == "__main__":
    raise SystemExit(main())
