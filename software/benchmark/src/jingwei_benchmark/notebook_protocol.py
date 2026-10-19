"""Explicit historical benchmark replay; never used by the fair protocol.

The source notebook assumes a header and discards the first CSV column, logs
positive values, then lets CSVDataset fit float32 statistics before masking.
These historical choices are preserved for comparison, including the single
mask, mixed-scale thresholds and exclusion of empty protein columns in scoring.
"""

from pathlib import Path
import tempfile
import threading

import numpy as np
import pandas as pd

from .data_io import MatrixData
from .masking import BenchmarkMasks

SOURCE_REVISION = "98686b54fdb9918a4d4cc9d99ac4880a6dadcefe"
PROTOCOLS = ("fair", "original-notebook")
NOTEBOOK_ORDER = ("missforest", "pimms-vae", "knn", "nmf", "pimms-dae", "feature-minimum", "mice-pmm", "jingwei-dmf", "jingwei-dcae")
_RNG_LOCK = threading.Lock()


def _read_notebook_csv(path, max_dense_mb):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("original-notebook replay requires the original raw CSV with a header and first column; use fair for h5ad")
    if not np.isfinite(max_dense_mb) or max_dense_mb <= 0:
        raise ValueError("--max-dense-mb must be positive and finite")
    frame = pd.read_csv(path)
    if frame.shape[0] == 0 or frame.shape[1] < 2:
        raise ValueError("original-notebook replay needs rows and at least two CSV columns")
    if frame.shape[0] * (frame.shape[1] - 1) * 8 > max_dense_mb * 1024**2:
        raise ValueError("matrix exceeds --max-dense-mb")
    raw = frame.iloc[:, 1:].to_numpy(dtype=float)
    if np.isinf(raw).any():
        raise ValueError("input contains infinity")
    working = raw.copy()
    positive = working > 0
    working[positive] = np.log2(working[positive])
    return frame, raw, working


def _detection():
    return {"has_header": True, "has_id_column": True,
            "header_policy": "original-notebook", "index_policy": "first-unconditionally"}


def load_notebook_preview(path, *, max_dense_mb=2048):
    """Read the same input for Qt without importing Torch or drawing any mask."""
    frame, raw, working = _read_notebook_csv(path, max_dense_mb)
    values = np.nan_to_num(working, nan=0).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("original-notebook replay requires float32-finite values")
    observed = values != 0
    preview = pd.DataFrame(np.where(observed, values, np.nan), columns=frame.columns[1:])
    preview.index = preview.index.astype(str)
    return MatrixData(preview, observed, "log2", raw_values=raw, csv_detection=_detection())


def load_notebook_matrix(path, *, seed=0, max_dense_mb=2048):
    """Run benchmark.ipynb's preprocessing and original CSVDataset verbatim."""
    import torch
    from ._legacy.datasets import CSVDataset

    frame, raw, working = _read_notebook_csv(path, max_dense_mb)
    converted = pd.concat([frame.iloc[:, :1], pd.DataFrame(working, index=frame.index, columns=frame.columns[1:])], axis=1)
    # The CSV round trip is intentional: it is part of the original loader.
    with tempfile.TemporaryDirectory(prefix="jingwei-notebook-") as temp:
        source = Path(temp) / "working.csv"
        converted.to_csv(source, index=False)
        with _RNG_LOCK, torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            dataset = CSVDataset(source)
            rng_state = torch.get_rng_state().numpy().copy()
    truth = dataset.data_filled.numpy().copy()
    if not np.isfinite(truth).all():
        raise ValueError("original-notebook replay requires float32-finite values")
    observed = truth != 0  # Exactly benchmark.ipynb's initial_mask.
    visible = dataset.mask.numpy().astype(bool)
    evaluation = ~visible & observed
    masks = BenchmarkMasks(observed, evaluation, np.zeros_like(observed))
    values = pd.DataFrame(truth, columns=frame.columns[1:])
    values.index = values.index.astype(str)
    data = MatrixData(values, observed, "log2", raw_values=raw, csv_detection=_detection())
    data.notebook_reference = {
        "masks": masks, "normalized": dataset.data_normalized.numpy().copy(),
        "mean": dataset.mean.numpy().copy(), "std": dataset.std.numpy().copy(),
        "rng_state": rng_state, "seed": seed,
    }
    return data


def notebook_masks(data):
    reference = data.notebook_reference
    masks = reference["masks"]
    eligible = int(masks.observed.sum())
    hidden = int(masks.evaluation_hidden.sum())
    common = {"mechanism": "mnar-jingwei-original", "probability_mode": "original-notebook",
              "marginal_repairs": 0, "marginal_preservation": False,
              "requested_rate": None, "value_scale": "log2-float32",
              "source_revision": SOURCE_REVISION, "rng": "torch-cpu-global-stream",
              "random_seed": reference["seed"]}
    return masks, {
        "evaluation": {**common, "eligible_count": eligible, "hidden_count": hidden,
                       "actual_rate": hidden / eligible if eligible else 0.0,
                       "statistics_scope": "all initially observed entries, before evaluation masking",
                       "rule": "benchmark.ipynb -> src/datasets.py::CSVDataset.get_mask"},
        "training": {**common, "eligible_count": int(masks.available.sum()), "hidden_count": 0,
                     "actual_rate": 0.0, "rule": "no second mask in the original notebook"},
    }


def notebook_score_mask(truth, prediction, masks):
    """Exact old valid_mask: common nonempty columns, then finite predictions."""
    valid_columns = masks.available.any(axis=0)
    return (masks.evaluation_hidden & valid_columns[None, :]
            & np.isfinite(truth) & np.isfinite(prediction))
