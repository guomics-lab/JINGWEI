"""Held-out reconstruction metrics and Wasserstein distances between samples."""

import numpy as np


def _sample(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sample")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.sort(values)


def _quantile_gaps(u_values, v_values):
    u = _sample(u_values, "u_values")
    v = _sample(v_values, "v_values")
    if u.size == v.size:
        return np.abs(u - v), np.full(u.size, 1.0 / u.size)
    # Empirical inverse CDFs are step functions: integrate each interval exactly.
    u_cdf = np.arange(1, u.size + 1, dtype=np.float64) / u.size
    v_cdf = np.arange(1, v.size + 1, dtype=np.float64) / v.size
    edges = np.unique(np.concatenate(([0.0], u_cdf, v_cdf)))
    widths = np.diff(edges)
    midpoints = edges[:-1] + widths / 2
    u_quantiles = u[np.searchsorted(u_cdf, midpoints, side="left")]
    v_quantiles = v[np.searchsorted(v_cdf, midpoints, side="left")]
    return np.abs(u_quantiles - v_quantiles), widths


def wasserstein_distance_1(u_values, v_values):
    """Exact W1 for two finite, uniformly weighted one-dimensional samples."""
    gaps, masses = _quantile_gaps(u_values, v_values)
    return float(np.dot(masses, gaps))


def wasserstein_distance_2_1d(u_values, v_values):
    """Return W2, not W2 squared, for unweighted 1-D empirical distributions.

    W2 = sqrt(integral_0^1 |F_u^-1(q) - F_v^-1(q)|^2 dq).
    Unequal sample counts and repeated values are supported without interpolating
    quantiles. For equal counts this is the RMSE between the sorted samples.
    This is not a multivariate distance and does not measure paired-entry error.
    """
    gaps, masses = _quantile_gaps(u_values, v_values)
    scale = float(gaps.max())
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.dot(masses, (gaps / scale) ** 2)))


def wasserstein_distance_2(reference_samples, predicted_samples):
    """Exact multivariate W2 between two uniformly weighted sample matrices.

    Rows are samples, columns are the same features in the same order. The
    ground cost is squared Euclidean distance between full sample vectors:
    W2 = sqrt(min_pi sum_ij pi_ij * ||reference_i - predicted_j||_2**2).

    Equal row counts use an exact minimum-cost assignment; unequal counts use
    balanced optimal transport via a sparse linear program. Rows are never
    flattened into individual feature values. Both inputs must be complete and
    finite: native missing entries cannot be used as reference ground truth.
    Feature units and scaling must match. Memory for pairwise costs is O(n*m).
    """
    from scipy.optimize import linear_sum_assignment, linprog
    from scipy.sparse import eye, kron, vstack
    from scipy.spatial.distance import cdist

    reference = np.asarray(reference_samples, dtype=np.float64)
    predicted = np.asarray(predicted_samples, dtype=np.float64)
    if reference.ndim != 2 or predicted.ndim != 2:
        raise ValueError("inputs must be 2-D matrices with samples in rows")
    n, features = reference.shape
    m, predicted_features = predicted.shape
    if n == 0 or m == 0 or features == 0 or features != predicted_features:
        raise ValueError("nonempty sample matrices must share the same feature count")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError(
            "sample W2 requires complete, finite matrices on a common feature set"
        )
    cost = cdist(reference, predicted, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError(
            "squared sample distances overflow; rescale features before computing W2"
        )
    cost_scale = float(cost.max())
    if cost_scale == 0:
        return 0.0
    if n == m:
        rows, columns = linear_sum_assignment(cost)
        squared_distance = float(cost[rows, columns].mean())
    else:
        source_constraints = kron(eye(n, format="csr"), np.ones((1, m)), format="csr")
        target_constraints = kron(np.ones((1, n)), eye(m, format="csr"), format="csr")
        # The final target marginal is implied by all source and other target marginals.
        constraints = vstack(
            (source_constraints, target_constraints[:-1]), format="csr"
        )
        marginals = np.concatenate((np.full(n, 1.0 / n), np.full(m - 1, 1.0 / m)))
        solution = linprog(
            (cost / cost_scale).ravel(),
            A_eq=constraints,
            b_eq=marginals,
            bounds=(0, None),
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(
                f"sample W2 optimal transport failed: {solution.message}"
            )
        squared_distance = float(solution.fun * cost_scale)
    return float(np.sqrt(max(0.0, squared_distance)))


def imputation_metrics(truth, predictions, evaluation_mask):
    """Score exactly the evaluation-hidden cells; never drop invalid predictions.

    MAE/RMSE compare the paired entries. Native missing and training-hidden cells
    are excluded. Missing or nonfinite predictions at test cells raise an error.
    Sample W2 is computed separately with wasserstein_distance_2, on complete
    matrices (or an explicitly selected common complete submatrix).
    """
    truth = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(evaluation_mask)
    if truth.ndim != 2 or predictions.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError(
            "truth, predictions and evaluation_mask must have matching 2-D shapes"
        )
    if mask.dtype != np.bool_:
        raise ValueError("evaluation_mask must be boolean (True means held out)")
    if not mask.any():
        raise ValueError("evaluation_mask contains no held-out entries")
    target = truth[mask]
    predicted = predictions[mask]
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError("all held-out targets and predictions must be finite")
    errors = np.abs(target - predicted)
    if not np.isfinite(errors).all():
        raise ValueError("held-out differences overflow; rescale before evaluation")
    scale = float(errors.max())
    rmse = float(scale * np.sqrt(np.mean((errors / scale) ** 2))) if scale else 0.0
    return {
        "n_evaluated": int(mask.sum()),
        "MAE": float(np.mean(errors)),
        "RMSE": rmse,
    }
