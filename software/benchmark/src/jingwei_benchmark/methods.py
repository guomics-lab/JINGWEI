"""Method registry and adapters. Official R and PIMMS backends stay optional."""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from .platforms import cuda_supported, platform_name, resolve_device, validate_device
from .runtime import resource_root


@dataclass(frozen=True)
class MethodSpec:
    name: str
    backend: str
    gpu: bool = False
    packages: tuple = ()


class MethodInputError(ValueError):
    """The native method cannot operate on this corrupted matrix."""

    failure_kind = "unsupported_input"


_NATIVE = (
    "zero",
    "minimum",
    "feature-minimum",
    "mean",
    "median",
    "sample-median",
    "perseus",
    "knn",
    "iterative-ridge",
    "iterative-rf",
    "nmf",
    "mice-pmm",
)
_R = {
    "bpca": ("pcaMethods",),
    "svd": ("pcaMethods",),
    "naguider-knn": ("impute",),
    "seqknn": ("SeqKnn",),
    "lls": ("pcaMethods",),
    "mle": ("norm",),
    "qrilc": ("imputeLCMD",),
    "mindet": ("imputeLCMD",),
    "minprob": ("imputeLCMD",),
    "impseq": ("rrcovNA",),
    "impseqrob": ("rrcovNA",),
    "irm": ("VIM",),
    "mice-norm": ("mice",),
    "mice-cart": ("mice",),
    "missforest": ("missForest",),
    "grr": ("glmnet",),
    "gms": ("GMSimpute",),
    "trknn": (),
}
METHODS = {n: MethodSpec(n, "python") for n in _NATIVE}
METHODS.update({n: MethodSpec(n, "r", packages=p) for n, p in _R.items()})
METHODS.update(
    {
        n: MethodSpec(n, "jingwei", True, ("torch", "pytorch_lightning"))
        for n in ("jingwei-dmf", "jingwei-dcae")
    }
)
METHODS.update(
    {
        n: MethodSpec(n, "pimms", True, ("pimmslearn",))
        for n in ("pimms-dae", "pimms-vae", "pimms-cf")
    }
)
ALIASES = {
    "dmf": "jingwei-dmf",
    "dcae": "jingwei-dcae",
    "rf": "missforest",
    "mice": "mice-pmm",
    "colmedian": "sample-median",
    "rowmedian": "median",
    "pi": "perseus",
    "svdmethod": "svd",
    "knnmethod": "naguider-knn",
    "qr": "qrilc",
}


def canonical_name(name):
    name = ALIASES.get(name.lower(), name.lower())
    if name not in METHODS:
        raise ValueError(f"unknown method {name}; use 'jingwei methods'")
    return name


def find_rscript(explicit=None):
    if explicit:
        path = Path(explicit)
        return str(path.resolve()) if path.is_file() else shutil.which(explicit)
    if getattr(sys, "frozen", False):
        bundled = resource_root() / ".r-runtime/bin/Rscript.exe"
        if bundled.is_file():
            return str(bundled.resolve())
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    root = resource_root()
    candidates = [os.environ.get("JINGWEI_RSCRIPT")]
    if platform_name() == "windows":
        candidates += [str(prefix / "Library/bin/Rscript.exe"),
                       str(prefix / "Scripts/Rscript.exe"),
                       shutil.which("Rscript"), str(root / ".r-runtime/bin/Rscript.exe")]
    else:
        candidates += [str(prefix / "bin/Rscript"), shutil.which("Rscript")]
        for home in (Path.home() / "miniconda3", Path.home() / "miniforge3", Path.home() / "anaconda3",
                     Path("/opt/miniconda3"), Path("/opt/anaconda3")):
            candidates.append(str(home / "envs/jingwei_benchmark/bin/Rscript"))
        if platform_name() == "macos":
            candidates += ["/Library/Frameworks/R.framework/Resources/bin/Rscript"]
    for path in candidates:
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    return None


def find_method_source(name):
    path = resource_root() / ".method-sources" / name
    return str(path) if path.is_dir() else None


def r_environment(library=None):
    env = os.environ.copy()
    if os.name == "nt":
        # UCRT R needs a UTF-8 Windows locale to resolve a bundled R_HOME and
        # package library beneath a Unicode installation directory.
        env["LC_ALL"] = "English_United States.utf8"
        if getattr(sys, "frozen", False) and (resource_root() / ".r-runtime/bin/Rscript.exe").is_file():
            env["R_HOME"] = str(resource_root() / ".r-runtime")
    root = resource_root() / ".r-library"
    requested = library or env.get("JINGWEI_R_LIBRARY")
    directory = Path(requested) if requested else root / platform_name()
    usable = directory.is_dir() and (
        platform_name() != "windows" or (directory / "jsonlite").is_dir()
    )
    if requested or usable:
        env["R_LIBS_USER"] = str(directory.resolve())
    elif platform_name() == "windows" and (root / "jsonlite").is_dir():
        env["R_LIBS_USER"] = str(root.resolve())
    return env


def availability(
    rscript=None, r_library=None, source_dir=None, dreamai_source=None, names=None,
    mice_backend="python",
    evaluation_protocol="fair",
):
    selected = dict(METHODS) if names is None else {name: METHODS[name] for name in names}
    if evaluation_protocol == "original-notebook":
        for name in ("missforest", "mice-pmm"):
            if name in selected:
                selected[name] = MethodSpec(name, "python")
    if mice_backend not in ("python", "r"):
        raise ValueError("MICE backend must be python or r")
    if mice_backend == "r" and "mice-pmm" in selected and evaluation_protocol != "original-notebook":
        selected["mice-pmm"] = MethodSpec("mice-pmm", "r", packages=("mice",))
    rscript = find_rscript(rscript)
    packages = sorted(
        {
            "jsonlite",
            "DreamAI",
            *(p for s in selected.values() if s.backend == "r" for p in s.packages),
        }
    )
    found = {}
    check_error = None
    bridge = Path(__file__).with_name("r_backend.R")
    if (
        rscript
        and bridge.is_file()
        and any(s.backend == "r" for s in selected.values())
    ):
        try:
            check = subprocess.run(
                [rscript, "--vanilla", str(bridge), "--check", *packages],
                env=r_environment(r_library), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=45,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if check.returncode == 0:
                for line in check.stdout.splitlines():
                    fields = line.split("\t")
                    if len(fields) >= 2:
                        found[fields[0]] = fields[1] == "TRUE"
            else:
                check_error = f"R dependency check failed (exit {check.returncode}): " + (check.stderr or check.stdout or "no output")[-2000:].strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            check_error = f"R dependency check failed: {exc}"
    result = {}
    for name, spec in selected.items():
        reason = None
        if spec.backend == "r":
            if not rscript:
                reason = "Rscript not found"
            elif not bridge.is_file():
                reason = "R bridge not installed"
            elif check_error:
                reason = check_error
            else:
                missing = [p for p in ("jsonlite", *spec.packages) if not found.get(p)]
                if missing:
                    reason = "missing R packages: " + ", ".join(missing)
                elif name == "trknn" and not source_dir:
                    reason = "requires --naguider-source with Trunc_KNN/Imput_funcs.r"
                elif name == "grr" and not found.get("DreamAI") and not dreamai_source:
                    reason = "requires DreamAI package or --dreamai-source for original RegImpute.R"
        else:
            missing = [p for p in spec.packages if find_spec(p) is None]
            if missing:
                reason = "missing Python packages: " + ", ".join(missing)
        result[name] = {
            "backend": spec.backend,
            "gpu": spec.gpu and cuda_supported(),
            "available": reason is None,
            "reason": reason,
        }
    return result


def _fill(matrix, statistic, axis):
    with np.errstate(all="ignore"):
        values = statistic(matrix, axis=axis)
    if not np.isfinite(values).all():
        raise ValueError("method cannot impute an entirely missing row/feature")
    return np.where(
        np.isfinite(matrix), matrix, values[None, :] if axis == 0 else values[:, None]
    )


def run_method(
    name,
    matrix,
    training_hidden,
    params,
    *,
    device,
    seed,
    folder,
    rscript=None,
    r_library=None,
    source_dir=None,
    dreamai_source=None,
):
    spec = METHODS[name]
    if name == "mice-pmm":
        mice_backend = params.get("mice_backend", "python")
        if mice_backend not in ("python", "r"):
            raise ValueError("MICE backend must be python or r")
        if mice_backend == "r":
            spec = MethodSpec(name, "r", packages=("mice",))
    validate_device(device)
    if spec.gpu:
        device = resolve_device(device)
    matrix = np.asarray(matrix, dtype=float)
    baseline_engine = params.get("baseline_engine", "optimized")
    if baseline_engine not in ("optimized", "native"):
        raise ValueError("baseline engine must be optimized or native")
    available = np.isfinite(matrix)
    if name == "mice-pmm" and mice_backend == "python" and not available.any(axis=0).all():
        raise MethodInputError("MICE PMM cannot impute an entirely missing feature without observed donors")
    if training_hidden.shape != matrix.shape or (training_hidden & ~available).any():
        raise ValueError("training mask must be a subset of remaining observations")
    if name in ("seqknn", "impseq", "mle"):
        if available.all():
            return matrix.copy(), {"device": "cpu", "baseline_engine": baseline_engine,
                "training_protocol": "no missing entries; input already complete"}
    if name in ("seqknn", "impseq"):
        # These sequential implementations bootstrap from complete feature
        # profiles. Inventing a mean-filled pool would change the method.
        neighbors = min(int(params.get("k", 10)), matrix.shape[1] - 1)
        complete = int(available.all(axis=0).sum())
        # SeqKnn filters unavailable neighbors internally, so k can exceed the
        # initial pool size. Two rows are still needed for its matrix handling;
        # impSeq also needs at least two rows for a finite sample covariance.
        if complete < 2 or (name == "seqknn" and neighbors < 1):
            label = "SeqKNN" if name == "seqknn" else "ImpSeq"
            raise MethodInputError(
                f"{label} requires at least 2 fully observed feature profiles "
                f"after evaluation masking; found {complete}. The shared mask is preserved."
            )
    if name == "mle":
        # norm::prelim.norm encodes NA patterns as signed 32-bit integers. With
        # gaps in later sample columns, integer overflow merges distinct
        # patterns; the package can return finite but invalid imputations.
        overflow = (len(matrix) > 1024 or
                    (len(matrix) > 31 and (~available[31:]).any()) or
                    (len(matrix) >= 31 and (~available[:31]).all(axis=0).any()))
        if overflow:
            raise MethodInputError(
                "MLE (norm) cannot safely encode this input's missingness patterns "
                f"across {len(matrix)} sample columns in signed 32-bit integers. "
                "No imputation score is produced."
            )
    rng = np.random.default_rng(seed)
    if spec.backend == "python":
        mice_metadata = {}
        if name == "mice-pmm":
            from .mice_engine import run_mice_pmm
            options = dict(params)
            options.setdefault("mice_solver", "cached" if baseline_engine == "optimized" else "reference")
            prediction, mice_metadata = run_mice_pmm(matrix, options, seed=seed, folder=folder)
        elif name == "zero":
            prediction = np.nan_to_num(matrix, nan=0.0)
        elif name == "minimum":
            prediction = np.where(available, matrix, np.nanmin(matrix))
        elif name in ("mean", "median", "sample-median", "feature-minimum"):
            stat = {
                "mean": np.nanmean,
                "median": np.nanmedian,
                "sample-median": np.nanmedian,
                "feature-minimum": np.nanmin,
            }[name]
            prediction = _fill(matrix, stat, 1 if name == "sample-median" else 0)
        elif name == "perseus":
            prediction = matrix.copy()
            for row in range(len(matrix)):
                values = matrix[row, available[row]]
                if values.size < 2 and (~available[row]).any():
                    raise ValueError(
                        "Perseus requires at least two visible features per sample"
                    )
                if values.size >= 2:
                    sd = values.std(ddof=1)
                    prediction[row, ~available[row]] = rng.normal(
                        values.mean() - params.get("downshift", 1.8) * sd,
                        params.get("width", 0.3) * sd,
                        (~available[row]).sum(),
                    )
        elif name == "knn":
            from sklearn.impute import KNNImputer

            prediction = KNNImputer(
                n_neighbors=params.get("k", 5), keep_empty_features=True
            ).fit_transform(matrix)
        elif name == "nmf":
            from .nmf_engine import run_nmf
            if not available.any(axis=0).all():
                raise MethodInputError("NMF requires at least one observation per feature")
            prediction = run_nmf(np.where(available, matrix, 0), available.astype(float))
        else:
            from sklearn.ensemble import RandomForestRegressor
            from .iterative_engine import make_imputer

            # On tiny forests, per-fit joblib pools cost more than tree growth.
            # Threads are a ceiling; tree count/predictors/RNG remain unchanged.
            forest_jobs = int(params.get("threads", 1))
            if baseline_engine == "optimized" and matrix.size < 10000:
                forest_jobs = 1
            estimator = (
                None
                if name == "iterative-ridge"
                else RandomForestRegressor(
                    n_estimators=params.get("trees", 100), random_state=seed,
                    n_jobs=forest_jobs
                )
            )
            imputer, discard_models = make_imputer(
                optimized=baseline_engine == "optimized",
                estimator=estimator,
                random_state=seed,
                max_iter=params.get("max_iter", 10),
                keep_empty_features=True,
            )
            prediction = imputer.fit_transform(matrix)
        metadata = {
            "device": "cpu",
            "training_protocol": "fit common evaluation-corrupted observations",
            **mice_metadata,
        }
        if name in ("iterative-rf", "iterative-ridge"):
            metadata.update(iterations_completed=int(imputer.n_iter_), max_iterations=params.get("max_iter", 10))
            metadata.update(feature_fits=len(imputer.imputation_sequence_),
                            retained_estimators=0 if discard_models else len(imputer.imputation_sequence_))
        if name == "iterative-rf":
            metadata.update(forest_jobs=forest_jobs, trees=params.get("trees", 100))
    elif spec.backend == "jingwei":
        engine = params.get("training_engine", "torch")
        if engine == "torch":
            from .torch_engine import run_jingwei_torch as run_jingwei
        elif engine == "lightning":
            from .lightning_engine import run_jingwei_lightning as run_jingwei
        else:
            raise ValueError(f"unknown training engine: {engine}")
        prediction, metadata = run_jingwei(name, matrix, training_hidden, params, device=device, seed=seed, folder=folder)
    elif spec.backend == "pimms":
        from .pimms_engine import run_pimms

        prediction, metadata = run_pimms(name, matrix, training_hidden, params, device=device, seed=seed, folder=folder)
    else:
        transfer = params.get("r_transfer_format", "binary")
        if transfer not in ("binary", "csv"):
            raise ValueError("R transfer format must be binary or csv")
        suffix = "bin" if transfer == "binary" else "csv"
        input_file, output_file = (
            Path(folder) / ("r_input." + suffix),
            Path(folder) / ("r_output." + suffix),
        )
        if transfer == "binary":
            np.asarray(matrix, dtype="<f8").tofile(input_file)
        else:
            pd.DataFrame(matrix).to_csv(input_file, index=False)
        parameter_file = Path(folder) / "r_parameters.json"
        parameter_file.write_text(json.dumps({**params, "matrix_shape": list(matrix.shape)}), encoding="utf-8")
        # R reads script expressions lazily. A per-run snapshot also preserves
        # the adapter used for replay and protects in-flight source runs from
        # an application update while a long R computation is executing.
        adapter = Path(folder) / "r_adapter"
        adapter.mkdir(exist_ok=True)
        for filename in ("r_backend.R", "r_acceleration.R"):
            shutil.copyfile(Path(__file__).with_name(filename), adapter / filename)
        command = [
            find_rscript(rscript),
            "--vanilla",
            str(adapter / "r_backend.R"),
            name,
            str(input_file),
            str(output_file),
            str(seed),
            str(parameter_file),
            str(source_dir or ""),
            str(dreamai_source or ""),
            transfer,
        ]
        completed = subprocess.run(
            command, env=r_environment(r_library), capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
        if completed.returncode:
            raise RuntimeError("R method failed: " + completed.stderr.strip()[:1500])
        if transfer == "binary":
            prediction = np.fromfile(output_file, dtype="<f8").reshape(matrix.shape)
        else:
            prediction = pd.read_csv(output_file).to_numpy(dtype=float)
        metadata = {
            "device": "cpu",
            "training_protocol": "official R method on common evaluation-corrupted observations",
            "r_transfer_format": transfer,
            "r_adapter_sha256": hashlib.sha256((adapter / "r_backend.R").read_bytes()).hexdigest(),
            "r_acceleration_sha256": hashlib.sha256((adapter / "r_acceleration.R").read_bytes()).hexdigest(),
        }
        detail_file = Path(str(output_file) + ".json")
        if detail_file.is_file():
            detail = json.loads(detail_file.read_text(encoding="utf-8"))
            metadata["r_acceleration"] = detail["acceleration"]
            metadata.update(detail.get("execution", {}))
    if spec.backend != "jingwei":
        metadata["baseline_engine"] = baseline_engine
    if name == "mice-pmm":
        metadata["mice_backend"] = mice_backend
    if not spec.gpu:
        # Completing an algorithm is not evidence that an iterative or
        # stochastic estimator has converged. Preserve that distinction.
        direct = name in ("zero", "minimum", "feature-minimum", "mean", "median", "sample-median", "knn", "perseus")
        metadata.setdefault("convergence_status", "not_applicable" if direct else "not_assessed")
        metadata.setdefault("converged", None)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != matrix.shape or not np.isfinite(prediction).all():
        raise ValueError(
            "method returned wrong shape or nonfinite predictions; no test entries were dropped"
        )
    prediction[available] = matrix[available]
    return prediction, metadata
