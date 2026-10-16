"""Isolated method process, receiving no evaluation truth values."""

import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    from .runtime import prepare_standard_streams
    prepare_standard_streams()
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    folder = Path(payload["folder"])
    started = time.perf_counter()
    metadata = {"status": "failed"}
    try:
        print(f"Starting {payload['method']} | device={payload['device']} | seed={payload['seed']}", flush=True)
        from .platforms import validate_device
        validate_device(payload["device"], payload.get("gpu_id"))
        threads = payload.get("threads", 4)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "R_RANGER_NUM_THREADS"):
            os.environ[key] = str(threads)
        os.environ.setdefault("MPLBACKEND", "Agg")
        if payload.get("gpu_id") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu_id"])
        import numpy as np
        from threadpoolctl import threadpool_limits
        from .methods import run_method

        with np.load(payload["input_file"], allow_pickle=False) as data:
            matrix, training_hidden = data["values"], data["training_hidden"]
        print(f"Input: {matrix.shape[0]} samples x {matrix.shape[1]} proteins | "
              f"observed={int(np.isfinite(matrix).sum())} | training masked={int(training_hidden.sum())}", flush=True)
        if payload["method"].startswith(("jingwei-", "pimms-")):
            import torch

            torch.set_num_threads(threads)
        ready = time.perf_counter()
        with threadpool_limits(limits=threads):
            parameters = {**payload["parameters"], "threads": threads}
            # Legacy job JSON predates the Python backend. Replaying it must
            # retain its original R method, including already-running batches.
            if payload["method"] == "mice-pmm" and "mice_backend" not in parameters:
                parameters["mice_backend"] = "r"
            prediction, metadata = run_method(
                payload["method"],
                matrix,
                training_hidden,
                parameters,
                device=payload["device"],
                seed=payload["seed"],
                folder=folder,
                rscript=payload.get("rscript"),
                r_library=payload.get("r_library"),
                source_dir=payload.get("source_dir"),
                dreamai_source=payload.get("dreamai_source"),
            )
        computed = time.perf_counter()
        temporary = folder / "predictions.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, values=prediction)
        temporary.replace(folder / "predictions.npz")
        metadata["status"] = "completed"
        metadata.update(worker_setup_s=ready-started, backend_s=computed-ready,
                        prediction_write_s=time.perf_counter()-computed, cpu_thread_budget=threads)
        print(f"Completed {payload['method']} | backend={computed-ready:.3f}s | total={time.perf_counter()-started:.3f}s", flush=True)
    except Exception as exc:
        traceback.print_exc()
        metadata = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if hasattr(exc, "failure_kind"):
            metadata["failure_kind"] = exc.failure_kind
    metadata["time_s"] = time.perf_counter() - started
    from .atomic import write_json
    write_json(folder / "worker_result.json", metadata)
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
