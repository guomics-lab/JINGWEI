"""Reusable benchmark orchestration for the CLI and future GUI."""

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from .data_io import load_matrix, write_predictions
from .methods import METHODS, availability, canonical_name
from .metrics import imputation_metrics, wasserstein_distance_2
from .protocol import ORIGINAL_MECHANISM, protocol_masks, uses_mask_rate
from .platforms import resolve_device
from .runtime import python_command
from .presets import preset_parameters
from .atomic import write_json as dump_json


def _stop_process(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_worker(payload, timeout, cancel_file=None):
    folder = Path(payload["folder"])
    payload_file = folder / "job.json"
    dump_json(payload_file, payload)
    started = time.perf_counter()
    with (folder / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            python_command("worker", payload_file),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                 "PYTHONUNBUFFERED": "1",
                 "JINGWEI_PROCESS_LOG": str((folder / "run.log").resolve())},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if cancel_file and Path(cancel_file).exists():
                    _stop_process(process)
                    for name in ("predictions.npz", "worker_result.json"):
                        late = folder / name
                        if late.is_file():
                            late.rename(folder / ("cancelled_" + name))
                    return {"status": "cancelled", "error": "cancelled by user",
                            "time_s": time.perf_counter() - started}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                try:
                    process.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            _stop_process(process)
            # A worker can finish during Windows process-tree termination.
            # Preserve late artifacts for inspection, but never accept them as results.
            for name in ("predictions.npz", "worker_result.json"):
                late = folder / name
                if late.is_file():
                    late.rename(folder / ("late_" + name))
            return {
                "status": "timeout",
                "error": f"method exceeded {timeout}s",
                "time_s": time.perf_counter() - started,
            }
        except BaseException:
            _stop_process(process)
            raise
    result_file = folder / "worker_result.json"
    if not result_file.is_file():
        return {
            "status": "failed",
            "error": f"worker exited {process.returncode}; see run.log",
            "time_s": time.perf_counter() - started,
        }
    return json.loads(result_file.read_text(encoding="utf-8"))


def _report(folder, results, methods, repeats, benchmark, notebook=False):
    frame = pd.DataFrame(results)
    frame.to_csv(folder / "metrics.csv", index=False)
    dump_json(folder / "metrics.json", results)
    summaries = []
    metric_names = ("MSE", "MAE", "RMSE", "NRMSE", "Pearson", "Spearman", "W2", "time_s")
    for method in methods:
        selected = frame[frame.method == method]
        successful = selected[selected.status == "completed"]
        row = {
            "method": method,
            "completed": int(len(successful)),
            "failed": int((selected.status == "failed").sum()),
            "skipped": int((selected.status == "skipped").sum()),
            "timeout": int((selected.status == "timeout").sum()),
            "cancelled": int((selected.status == "cancelled").sum()),
        }
        for metric in metric_names:
            values = pd.to_numeric(
                successful.get(metric, pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(folder / "summary.csv", index=False)
    paired_repeats = []
    if benchmark:
        for repeat in range(repeats):
            selected = frame[frame.repeat == repeat]
            if len(selected) == len(methods) and (selected.status == "completed").all():
                paired_repeats.append(repeat)
        frame[frame.repeat.isin(paired_repeats)].to_csv(
            folder / "paired_metrics.csv", index=False
        )
    text = (
        "Scores describe artificial holdouts from observed entries under the selected mechanism. "
        "Summary means include successful runs only; counts show missing runs. "
        "paired_metrics.csv contains only repeats completed by every selected method. "
        "W2 compares multivariate samples on common complete reference rows; unavailable W2 is left blank. "
        "MSE/MAE/RMSE/NRMSE/correlations use exactly the evaluation-hidden entries. "
        "MSE is RMSE squared, in squared working-scale units."
    )
    if frame.get("n_excluded", pd.Series(dtype=float)).fillna(0).gt(0).any() and not notebook:
        text += (" Original-repository MNAR excludes the shared columns with no observations "
                 "after evaluation masking, as in the source notebook. n_held_out, n_evaluated "
                 "and n_excluded record this scope; excluded columns remain missing in exports.")
    if notebook:
        text = ("Historical notebook replay: first CSV column removed unconditionally; log2 of positive values; "
                "float32 statistics fitted before masking; one original get_mask draw, no second training mask. "
                "Scores exclude columns empty after masking and nonfinite predictions as in the old notebook. "
                "n_held_out, n_evaluated and n_excluded show the scoring subset. Additional metrics were not "
                "reported by the original notebook. Runtime and stochastic outputs depend on this machine and dependencies.")
    page = (
        "<!doctype html><html><meta charset='utf-8'><title>JINGWEI benchmark</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>JINGWEI results</h1><p>"
        + html.escape(text)
        + "</p><p>Paired repeats: "
        + str(len(paired_repeats))
        + "</p>"
        + summary.to_html(index=False, escape=True)
        + "<h2>Individual runs</h2>"
        + frame.to_html(index=False, escape=True)
        + "</html>"
    )
    (folder / "report.html").write_text(page, encoding="utf-8")


def run_experiment(args):
    methods = (
        list(METHODS)
        if args.methods == ["all"]
        else list(dict.fromkeys(canonical_name(n) for n in args.methods))
    )
    training_methods = [name for name in methods if name.startswith("jingwei-")] if args.command == "impute" else []
    if training_methods and uses_mask_rate(args.mechanism, args.paper_probabilities) and not 0 <= args.mask_rate < 1:
        raise ValueError("JINGWEI training mask rate must be in [0,1)")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory is not empty; choose a new --output path")
    overrides = (
        json.loads(args.method_config.read_text(encoding="utf-8-sig"))
        if args.method_config else {}
    )
    if not isinstance(overrides, dict) or any(not isinstance(v, dict) for v in overrides.values()):
        raise ValueError("--method-config must contain an object of parameter objects keyed by method")
    overrides = {canonical_name(k): v for k, v in overrides.items()}
    notebook = args.command == "benchmark" and getattr(args, "evaluation_protocol", "fair") == "original-notebook"
    if "jingwei-dcae" in methods:
        architecture = overrides.get("jingwei-dcae", {}).get("ae_architecture",
            getattr(args, "ae_architecture", None) or ("native" if notebook else "global"))
        if architecture not in ("native", "global"):
            raise ValueError("AE architecture must be native or global")
        if notebook and architecture != "native":
            raise ValueError("original-notebook requires the original AE; select the fair protocol to use the global AE")
    if notebook:
        from .notebook_protocol import load_notebook_matrix, NOTEBOOK_ORDER
        methods.sort(key=lambda method: NOTEBOOK_ORDER.index(method) if method in NOTEBOOK_ORDER else len(NOTEBOOK_ORDER))
        args.methods = methods
        data = load_notebook_matrix(args.input, seed=args.seed, max_dense_mb=args.max_dense_mb)
        args.repeats = 1
        print("Original notebook replay: first CSV column dropped; pre-mask float32 statistics; "
              "single get_mask draw; no second mask; original score filtering. Seed=" + str(args.seed), flush=True)
    else:
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
    # Historical CSVDataset masking uses working values (normally log2); custom
    # MNAR mechanisms retain their declared-input-scale contract.
    truth = data.frame.to_numpy(dtype=np.float32 if notebook else float)
    args.output.mkdir(parents=True, exist_ok=True)
    status = availability(
        args.rscript,
        args.r_library,
        args.naguider_source,
        args.dreamai_source,
        names=methods,
        mice_backend=getattr(args, "mice_backend", "python"),
        evaluation_protocol="original-notebook" if notebook else "fair",
    )
    device = resolve_device(args.device, args.gpu_id,
                            neural=any(METHODS[n].gpu for n in methods))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    versions = {}
    for package in (
        "jingwei",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "fastai",
        "pytorch-lightning",
        "pimms-learn",
        "anndata",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    with args.input.open("rb") as input_stream:
        fingerprint = hashlib.file_digest(input_stream, "sha256").hexdigest()
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "config": config,
        "input_sha256": fingerprint,
        "shape": list(truth.shape),
        "sample_ids": list(data.frame.index),
        "feature_ids": list(data.frame.columns),
        "observed_count": int(data.observed.sum()),
        "evaluation_scale": data.scale,
        "input_scale": args.scale,
        "mask_protocol_version": 3,
        "effective_protocol": ({"name": "original-notebook", "source_revision": "98686b54fdb9918a4d4cc9d99ac4880a6dadcefe",
            "seed": args.seed, "repeats": 1, "input": "raw CSV; first column discarded",
            "preprocessing": "log2 positives; original float32 pre-mask mean/std",
            "evaluation": "original CSVDataset.get_mask, no calibration or marginal repair",
            "training": "single notebook mask, no second mask",
            "scoring": "exclude empty columns and nonfinite predictions; record excluded counts",
            "rng": "original method order; Python/NumPy/Torch streams carried between workers"} if notebook else
            {"name": "fair", "evaluation": args.mechanism,
             "training": "method-owned; no platform training mask",
             "platform_mask_count": 1,
             "normalization": "training-available observations only"} if args.command == "benchmark" else
            {"name": "impute", "evaluation": "not-applied", "platform_mask_count": 0,
             "training": "JINGWEI training corruption; original observations retained as targets",
             "training_methods": training_methods, "training_mechanism": args.mechanism if training_methods else None,
             "normalization": "input observations only"}),
        "device": device,
        "csv_detection": data.csv_detection,
        "platform": platform.platform(),
        "python": sys.version,
        "versions": versions,
        "method_availability": {n: status[n] for n in methods},
        "runs": [],
    }
    dump_json(args.output / "manifest.json", manifest)
    results = []
    repeats = args.repeats if args.command == "benchmark" else 1
    for repeat, seed_sequence in enumerate(
        np.random.SeedSequence(args.seed).spawn(repeats)
    ):
        mask_sequence, model_sequence = seed_sequence.spawn(2)
        mask_seed = int(mask_sequence.generate_state(1)[0] % (2**31 - 1))
        model_seed = int(model_sequence.generate_state(1)[0] % (2**31 - 1))
        if notebook:
            from .notebook_protocol import notebook_masks
            masks, mask_metadata = notebook_masks(data)
            mask_seed = model_seed = args.seed
        else:
            masks, mask_metadata = protocol_masks(
            data.raw_values,
            data.observed,
            working_values=truth,
            mechanism=args.mechanism,
            evaluation_rate=(
                (args.mask_rate if uses_mask_rate(args.mechanism, args.paper_probabilities) else None)
                if args.command == "benchmark"
                else 0
            ),
            training_rate=(args.mask_rate if uses_mask_rate(args.mechanism, args.paper_probabilities) else None) if training_methods else 0,
            seed=mask_seed,
            preserve_marginals=not args.allow_empty_marginals,
            level=args.mask_level,
            slope=args.logistic_slope,
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
            low_probability=args.low_probability,
            high_probability=args.high_probability,
            threshold_probability=args.threshold_probability,
        )
        if args.command == "benchmark" and not masks.evaluation_hidden.any():
            raise ValueError(
                "evaluation mask has no held-out entries; increase rate or use a larger input"
            )
        repeat_dir = args.output / f"repeat_{repeat:03d}"
        print(f"Mask | observed={int(masks.observed.sum())} | evaluation={int(masks.evaluation_hidden.sum())} | "
              f"available={int(masks.available.sum())} | training={int(masks.training_hidden.sum())} | "
              f"empty features={int((~masks.available.any(axis=0)).sum())}", flush=True)
        repeat_dir.mkdir()
        original_empty_policy = not notebook and args.mechanism == ORIGINAL_MECHANISM
        valid_columns = masks.available.any(axis=0) if original_empty_policy else np.ones(truth.shape[1], dtype=bool)
        score_scope = masks.evaluation_hidden & valid_columns[None, :]
        if args.command == "benchmark" and not score_scope.any():
            raise ValueError("no evaluable holdouts remain on features observed after evaluation masking")
        if original_empty_policy:
            dump_json(repeat_dir / "feature_scope.json", {
                "policy": "original notebook: exclude features empty after evaluation masking",
                "excluded_feature_ids": list(data.frame.columns[~valid_columns]),
                "empty_after_evaluation": int((~valid_columns).sum()),
                "empty_after_training": int((~masks.training_visible.any(axis=0)).sum()),
                "n_held_out": int(masks.evaluation_hidden.sum()),
                "n_evaluated": int(score_scope.sum()),
                "n_excluded": int((masks.evaluation_hidden & ~valid_columns[None, :]).sum()),
                "export": "excluded features are NaN; they are not estimated or scored",
            })
            if (~valid_columns).any():
                print(f"Feature scope | excluded empty features={int((~valid_columns).sum())} | "
                      f"evaluated={int(score_scope.sum())} | excluded holdouts="
                      f"{int(masks.evaluation_hidden.sum()-score_scope.sum())}", flush=True)
        np.savez_compressed(
            repeat_dir / "masks.npz",
            observed=masks.observed,
            evaluation_hidden=masks.evaluation_hidden,
            training_hidden=masks.training_hidden,
        )
        # Benchmark workers receive only the common test-corrupted data. In
        # imputation mode the common matrix is the original input; only JINGWEI
        # receives the separate training-corruption mask, with targets retained.
        legacy = data.notebook_reference if notebook else None
        extra = ({"notebook_normalized": np.where(masks.available, legacy["normalized"], 0).astype("float32"),
                  "notebook_mean": legacy["mean"], "notebook_std": legacy["std"],
                  "notebook_rng_state": legacy["rng_state"]} if notebook else {})
        np.savez(
            repeat_dir / "method_input.npz",
            values=np.where(masks.available, truth, np.nan) if notebook else masks.method_input(truth),
            training_hidden=np.zeros_like(masks.training_hidden),
            **extra,
        )
        if training_methods:
            np.savez(repeat_dir / "jingwei_training_input.npz",
                     values=masks.method_input(truth), training_hidden=masks.training_hidden)
        dump_json(
            repeat_dir / "mask_config.json",
            {**mask_metadata, "mask_seed": mask_seed, "model_seed": model_seed,
             "training_methods": training_methods},
        )
        notebook_rng_input = None
        for method in methods:
            folder = repeat_dir / method
            folder.mkdir()
            row = {
                "repeat": repeat,
                "method": method,
                "seed": model_seed,
                "evaluation_rate": mask_metadata["evaluation"]["actual_rate"],
                "training_rate": mask_metadata["training"]["actual_rate"] if method in training_methods else 0.0,
                "evaluation_scale": data.scale,
                "zero_cols": int((~valid_columns).sum()) if original_empty_policy else 0,
            }
            cancel_file = getattr(args, "cancel_file", None)
            if cancel_file and Path(cancel_file).exists():
                result = {"status": "cancelled", "error": "cancelled by user"}
            elif not status[method]["available"]:
                result = {"status": "skipped", "error": status[method]["reason"]}
            else:
                params = {
                    "training_engine": getattr(args, "training_engine", "torch"),
                    "baseline_engine": getattr(args, "baseline_engine", "optimized"),
                    "mice_backend": getattr(args, "mice_backend", "python"),
                    "threads": args.threads,
                    "max_epochs": args.max_epochs,
                    "batch_size": args.batch_size,
                    "latent_dim": args.latent_dim,
                    "hidden_dims": args.hidden_dims,
                    "learning_rate": args.learning_rate,
                    "mask_weight": args.mask_weight,
                    "ae_architecture": getattr(args, "ae_architecture", None) or ("native" if notebook else "global"),
                    "evaluation_protocol": "original-notebook" if notebook else "fair",
                    **preset_parameters("original-benchmark" if notebook else getattr(args, "jingwei_preset", "standard"), method),
                    **overrides.get(method, {}),
                }
                payload = {
                    "method": method,
                    "input_file": str((repeat_dir / ("jingwei_training_input.npz" if method in training_methods else "method_input.npz")).resolve()),
                    "folder": str(folder.resolve()),
                    "parameters": params,
                    "device": device if METHODS[method].gpu else "cpu",
                    "seed": model_seed,
                    "threads": args.threads,
                    "gpu_id": args.gpu_id if METHODS[method].gpu else None,
                    "rscript": args.rscript,
                    "r_library": args.r_library,
                    "source_dir": args.naguider_source,
                    "dreamai_source": args.dreamai_source,
                    "notebook_rng_input": notebook_rng_input,
                    "exclude_empty_features": original_empty_policy,
                }
                print(f"[{repeat+1}/{repeats}] {method}", flush=True)
                result = _run_worker(payload, args.timeout, cancel_file)
                if notebook and (folder / 'notebook_rng.npz').is_file() and (folder / 'notebook_rng.json').is_file():
                    notebook_rng_input = str(folder.resolve())
                if result["status"] == "completed":
                    try:
                        prediction = np.load(
                            folder / "predictions.npz", allow_pickle=False
                        )["values"]
                        if args.command == "benchmark":
                            if notebook:
                                from .notebook_protocol import notebook_score_mask
                                score_mask = notebook_score_mask(truth, prediction, masks)
                            else:
                                score_mask = score_scope
                            result["n_held_out"] = int(masks.evaluation_hidden.sum())
                            result["n_excluded"] = int(masks.evaluation_hidden.sum() - score_mask.sum())
                            result.update(
                                imputation_metrics(
                                    truth, prediction, score_mask
                                )
                            )
                            if notebook:
                                from sklearn.metrics import mean_squared_error
                                result["MSE"] = float(mean_squared_error(truth[score_mask], prediction[score_mask]))
                            actual, predicted = (
                                truth[score_mask],
                                prediction[score_mask],
                            )
                            visible_sd = float(np.std(truth[masks.available]))
                            result["NRMSE"] = (
                                result["RMSE"] / visible_sd if visible_sd else None
                            )
                            varied = (
                                len(actual) > 1
                                and np.ptp(actual) > 0
                                and np.ptp(predicted) > 0
                            )
                            result["Pearson"] = (
                                float(pearsonr(actual, predicted).statistic)
                                if varied
                                else None
                            )
                            result["Spearman"] = (
                                float(spearmanr(actual, predicted).statistic)
                                if varied
                                else None
                            )
                            for metric in ("Pearson", "Spearman"):
                                if result[metric] is not None and not np.isfinite(result[metric]):
                                    result[metric] = None
                            selected_features = (
                                list(data.frame.columns)
                                if args.w2_features is None
                                else args.w2_features
                            )
                            if not selected_features or any(
                                f not in data.frame.columns for f in selected_features
                            ):
                                raise ValueError(
                                    "unknown or empty --w2-features selection"
                                )
                            positions = data.frame.columns.get_indexer(
                                selected_features
                            )
                            result["W2_excluded_empty_features"] = int((~valid_columns[positions]).sum())
                            positions = positions[valid_columns[positions]]
                            complete_rows = data.observed[:, positions].all(axis=1)
                            if len(positions) == 0:
                                complete_rows[:] = False
                            result["W2_n_samples"] = int(complete_rows.sum())
                            result["W2_n_features"] = len(positions)
                            result["W2_n_evaluated"] = int(
                                masks.evaluation_hidden[
                                    np.ix_(complete_rows, positions)
                                ].sum()
                            )
                            finite_w2 = np.isfinite(prediction[np.ix_(complete_rows, positions)]).all()
                            result["W2"] = (
                                wasserstein_distance_2(
                                    truth[np.ix_(complete_rows, positions)],
                                    prediction[np.ix_(complete_rows, positions)],
                                )
                                if complete_rows.any() and finite_w2
                                else None
                            )
                            result["W2_note"] = (
                                None
                                if complete_rows.any()
                                else "no complete reference samples on selected features"
                            )
                            if not finite_w2:
                                result["W2_note"] = "nonfinite predictions on the complete reference matrix"
                            if complete_rows.any() and result["W2_n_evaluated"] == 0:
                                result["W2"] = None
                                result["W2_note"] = (
                                    "selected complete reference matrix contains no evaluation-hidden entries"
                                )
                        exported = prediction.copy()
                        if original_empty_policy:
                            exported[:, ~valid_columns] = np.nan
                        write_predictions(
                            data,
                            exported,
                            folder,
                            method=method,
                            input_scale=args.scale,
                            output_scale=args.output_scale,
                            allow_nonfinite=notebook,
                            excluded_features=~valid_columns if original_empty_policy else None,
                        )
                    except Exception as exc:
                        result.update(
                            status="failed", error=f"output/evaluation failed: {exc}"
                        )
            row.update(result)
            results.append(row)
            dump_json(folder / "result.json", row)
            manifest["runs"].append(row)
            dump_json(args.output / "manifest.json", manifest)
            _report(args.output, results, methods, repeats, args.command == "benchmark", notebook)
            print(
                f"  {row['status']}"
                + (f": {row['error']}" if row.get("error") else ""),
                flush=True,
            )
            if row["status"] == "cancelled":
                print(f"Results: {args.output.resolve()}")
                return 3
    print(f"Results: {args.output.resolve()}")
    return 0 if all(r["status"] == "completed" for r in results) else 2
