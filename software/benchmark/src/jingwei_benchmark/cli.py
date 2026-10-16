"""Command-line entry point for inspection, imputation and benchmarking."""

import argparse
import json
from pathlib import Path
import sys
from . import __version__
from .data_io import load_matrix
from .methods import METHODS, availability, find_method_source
from .protocol import MECHANISMS
from .platforms import add_device_arguments
from .runtime import default_input_path

DEFAULT_TIMEOUT_SECONDS = 30 * 60


def parser():
    root = argparse.ArgumentParser(
        prog="jingwei-benchmark",
        description="JINGWEI Benchmark: compare proteomics imputation methods",
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = root.add_subparsers(dest="command", required=True)
    method = sub.add_parser(
        "methods", help="list methods and installed backend dependencies"
    )
    method.add_argument("--json", action="store_true")
    if sys.platform == "win32":
        gui = sub.add_parser("gui", help="launch the native Windows benchmark application")
        gui.add_argument("--workspace", type=Path, help="directory for persistent GUI jobs")
    for command in ("inspect", "impute", "benchmark"):
        p = sub.add_parser(command)
        default_input = default_input_path()
        p.add_argument("--input", type=Path, default=default_input, required=default_input is None,
                       help="CSV/h5ad input (default: bundled data/Alzheimer.csv)")
        p.add_argument(
            "--orientation",
            choices=("samples-by-features", "features-by-samples"),
            default="samples-by-features",
        )
        p.add_argument("--layer")
        p.add_argument(
            "--index-column",
            choices=("auto", "first", "none"),
            default="auto",
            help="infer ID column by type; use first for numeric IDs or none for expression data",
        )
        p.add_argument(
            "--header",
            choices=("auto", "yes", "no"),
            default="auto",
            help="infer header by type, or specify explicitly",
        )
        p.add_argument(
            "--missing-values", choices=("nan", "zero", "nonpositive"), default="nan"
        )
        p.add_argument("--scale", choices=("raw", "log2"), default="raw")
        p.add_argument("--transform", choices=("auto", "none", "log2"), default="auto")
        p.add_argument("--max-dense-mb", type=int, default=2048)
        if command == "inspect":
            continue
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--cancel-file", type=Path, help=argparse.SUPPRESS)
        p.add_argument(
            "--output-scale",
            choices=("input", "working"),
            default="working" if command == "benchmark" else "input",
        )
        p.add_argument(
            "--methods",
            nargs="+",
            default=["all"] if command == "benchmark" else ["jingwei-dmf", "jingwei-dcae", "mean", "median", "knn"],
            help="method IDs or all (benchmark default: all 34 methods)",
        )
        p.add_argument(
            "--method-config",
            type=Path,
            help="JSON object of per-method parameter overrides",
        )
        add_device_arguments(p)
        p.add_argument("--mechanism", choices=MECHANISMS, default="mcar")
        p.add_argument("--training-mechanism", choices=MECHANISMS)
        p.add_argument("--mask-rate", type=float, default=0.2)
        p.add_argument("--training-mask-rate", type=float, default=0.2)
        p.add_argument(
            "--paper-probabilities",
            action="store_true",
            help="use uncalibrated MNAR probabilities, recording actual rates",
        )
        p.add_argument(
            "--mask-level", choices=("entry", "feature-mean"), default="entry"
        )
        p.add_argument("--logistic-slope", type=float, default=-1.0)
        p.add_argument("--low-quantile", type=float, default=0.2)
        p.add_argument("--high-quantile", type=float, default=0.6)
        p.add_argument("--low-probability", type=float, default=0.9)
        p.add_argument("--high-probability", type=float, default=0.2)
        p.add_argument("--threshold-probability", type=float, default=0.05)
        p.add_argument("--allow-empty-marginals", action="store_true")
        p.add_argument("--repeats", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--baseline-engine", choices=("optimized", "native"), default="optimized",
                       help="execution path for comparison methods; native retains upstream data loading and MICE checks")
        p.add_argument("--mice-backend", choices=("python", "r"), default="python",
                       help="MICE-PMM implementation: Python linear ridge/PMM (default), or original R mice")
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                       help="maximum seconds per method/repeat (default: 1800; 30 min)")
        p.add_argument("--max-epochs", type=int, default=50)
        p.add_argument("--jingwei-preset", choices=("standard", "original-benchmark"), default="standard",
                       help="original-benchmark restores original JINGWEI model/training settings; overrides common training flags; --method-config takes precedence")
        p.add_argument("--training-engine", choices=("torch", "lightning"), default="torch",
                       help="JINGWEI native PyTorch execution, or original Lightning runner for comparison")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--latent-dim", type=int, default=64)
        p.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
        p.add_argument("--learning-rate", type=float, default=0.001)
        p.add_argument("--mask-weight", type=float, default=0.5)
        p.add_argument(
            "--w2-features",
            nargs="+",
            help="explicit common features for complete-reference sample W2",
        )
    for p in (method, sub.choices["impute"], sub.choices["benchmark"]):
        p.add_argument("--rscript")
        p.add_argument("--r-library")
        p.add_argument(
            "--naguider-source",
            default=find_method_source("NAguideR"),
            help="checkout for the original trKNN script",
        )
        p.add_argument(
            "--dreamai-source",
            default=find_method_source("DreamAI"),
            help="DreamAI checkout or original RegImpute.R",
        )
    method.add_argument("--mice-backend", choices=("python", "r"), default="python")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import launch
            return launch(args.workspace)
        if args.command == "methods":
            states = availability(
                args.rscript, args.r_library, args.naguider_source, args.dreamai_source,
                mice_backend=args.mice_backend,
            )
            if args.json:
                print(json.dumps(states, indent=2))
            else:
                print(f"{'METHOD':<22} {'BACKEND':<9} {'GPU':<5} STATUS")
                for name, state in states.items():
                    print(
                        f"{name:<22} {state['backend']:<9} {str(state['gpu']):<5} "
                        + (
                            "dependencies installed"
                            if state["available"]
                            else state["reason"]
                        )
                    )
            return 0
        if args.command == "inspect":
            data = load_matrix(
                args.input,
                orientation=args.orientation,
                layer=args.layer,
                index_column=args.index_column,
                header=args.header,
                missing_values=args.missing_values,
                scale=args.scale,
                transform=args.transform,
                max_dense_mb=args.max_dense_mb,
            )
            print(
                json.dumps(
                    {
                        "samples": len(data.frame),
                        "features": data.frame.shape[1],
                        "missing_rate": float(1 - data.observed.mean()),
                        "complete_reference_rows": int(data.observed.all(axis=1).sum()),
                        "working_scale": data.scale,
                        "csv_detection": data.csv_detection,
                    },
                    indent=2,
                )
            )
            return 0
        import math
        if not math.isfinite(args.timeout):
            raise ValueError("--timeout must be positive and finite")
        for name in (
            "repeats",
            "threads",
            "max_epochs",
            "batch_size",
            "latent_dim",
            "timeout",
        ):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_','-')} must be positive")
        if args.seed < 0 or any(d <= 0 for d in args.hidden_dims):
            raise ValueError("seed must be nonnegative and hidden dimensions positive")
        from .benchmark import run_experiment

        return run_experiment(args)
    except (ValueError, OSError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print(f"jingwei-benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
