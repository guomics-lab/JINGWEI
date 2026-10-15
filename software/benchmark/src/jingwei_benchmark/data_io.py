"""CSV and AnnData input with explicit orientation, missingness and scale."""

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class MatrixData:
    frame: pd.DataFrame
    observed: np.ndarray
    scale: str
    adata: object = None
    raw_values: object = None
    csv_detection: object = None


def _cell_kind(value):
    value = value.strip()
    if value.lower() in ("", "na", "nan", "n/a", "null", "none"):
        return "missing"
    try:
        float(value)
        return "numeric"
    except ValueError:
        return "text"


def detect_csv_layout(path, *, header="auto", index_column="auto"):
    """Infer headers/ID columns by type; numeric identifiers require an override."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, [])
        if not first:
            raise ValueError("CSV is empty")
        kinds = [_cell_kind(v) for v in first]
        labels = {
            "id",
            "sample",
            "sample_id",
            "sampleid",
            "sample id",
            "protein",
            "protein_id",
            "feature",
            "feature_id",
            "index",
        }
        inferred_header = any(k == "text" for k in kinds[1:]) or (
            kinds[0] == "text" and first[0].strip().lower() in labels
        )
        has_header = inferred_header if header == "auto" else header == "yes"
        counts = {"numeric": 0, "text": 0, "missing": 0}
        if not has_header:
            counts[kinds[0]] += 1
        for row in reader:
            if not row:
                continue
            if len(row) != len(first):
                raise ValueError("CSV rows have inconsistent numbers of columns")
            counts[_cell_kind(row[0])] += 1
    has_index = (
        counts["text"] > 0 if index_column == "auto" else index_column == "first"
    )
    return {
        "has_header": has_header,
        "has_id_column": has_index,
        "first_row_types": {kind: kinds.count(kind) for kind in counts},
        "first_column_types": counts,
        "header_policy": header,
        "index_policy": index_column,
    }


def load_matrix(
    path,
    *,
    orientation="samples-by-features",
    layer=None,
    index_column="auto",
    header="auto",
    missing_values="nan",
    scale="raw",
    transform="auto",
    max_dense_mb=2048,
):
    path = Path(path)
    adata = None
    detection = None
    if path.suffix.lower() == ".h5ad":
        if orientation != "samples-by-features":
            raise ValueError("AnnData requires samples in obs and features in var")
        import anndata as ad

        adata = ad.read_h5ad(path)
        matrix = adata.X if layer is None else adata.layers[layer]
        if np.prod(matrix.shape) * 8 > max_dense_mb * 1024**2:
            raise ValueError("matrix exceeds --max-dense-mb")
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        frame = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
    elif path.suffix.lower() == ".csv":
        if layer is not None:
            raise ValueError("--layer is only valid for h5ad")
        detection = detect_csv_layout(path, header=header, index_column=index_column)
        if detection["has_header"]:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                names = next(csv.reader(handle))
            feature_header = names[1:] if detection["has_id_column"] else names
            if len(feature_header) != len(set(feature_header)):
                raise ValueError("CSV contains duplicate column identifiers")
        frame = pd.read_csv(
            path,
            header=0 if detection["has_header"] else None,
            index_col=0 if detection["has_id_column"] else None,
        )
        if orientation == "features-by-samples":
            frame = frame.T
    else:
        raise ValueError("input must be .csv or .h5ad")
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(
            "input must be nonempty with unique sample and feature identifiers"
        )
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("identifiers are not unique as strings")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("input contains infinity")
    observed = np.isfinite(values)
    if missing_values == "zero":
        observed &= values != 0
    elif missing_values == "nonpositive":
        observed &= values > 0
    if not observed.any():
        raise ValueError("input has no observed values")
    values = np.where(observed, values, np.nan)
    raw_values = values.copy()
    do_log = transform == "log2" or (transform == "auto" and scale == "raw")
    if scale == "log2" and transform == "log2":
        raise ValueError("input is already log2")
    working_scale = scale
    if do_log:
        if (values[observed] <= 0).any():
            raise ValueError(
                "log2 requires positive observations; declare nonpositive missing or use --transform none"
            )
        values[observed] = np.log2(values[observed])
        working_scale = "log2"
    return MatrixData(
        pd.DataFrame(values, index=frame.index, columns=frame.columns),
        observed,
        working_scale,
        adata,
        raw_values,
        detection,
    )


def write_predictions(
    data, values, folder, *, method, input_scale, output_scale="input"
):
    folder = Path(folder)
    output_values = np.asarray(values, dtype=float).copy()
    if output_scale == "input" and input_scale == "raw" and data.scale == "log2":
        with np.errstate(over="ignore", invalid="ignore"):
            output_values = np.exp2(output_values)
    if not np.isfinite(output_values).all():
        raise ValueError("nonfinite predictions on output scale")
    csv_temporary = folder / "imputed.csv.tmp"
    pd.DataFrame(
        output_values, index=data.frame.index, columns=data.frame.columns
    ).to_csv(csv_temporary)
    csv_temporary.replace(folder / "imputed.csv")
    if data.adata is not None:
        result = data.adata.copy()
        result.layers["jingwei_imputed"] = output_values
        result.uns["jingwei"] = {
            "method": method,
            "output_scale": input_scale if output_scale == "input" else data.scale,
        }
        temporary = folder / "imputed.h5ad.tmp"
        result.write_h5ad(temporary)
        temporary.replace(folder / "imputed.h5ad")
