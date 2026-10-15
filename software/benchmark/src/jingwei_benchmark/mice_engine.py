"""Linear ridge MICE with posterior draws and type-1 predictive mean matching.

The conditional model has a flat intercept, beta|sigma2 ~ N(0, sigma2/lambda I)
and p(sigma2) proportional to 1/sigma2. This is a regularized linear MICE
implementation, not an exact port of R mice's predictor screening/variance draw.
All predictors and requested imputations/sweeps are retained. A sample-space
Gram cache avoids feature-space covariance matrices on wide proteomics data.
"""

from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def conditional_predictions(z, gram, target, observed, normal_predictors,
                            normal_observed, normal_intercept, chi_square,
                            ridge=1e-5):
    """Return posterior mean/drawn predictions and a conjugate variance draw.

    Random variates are explicit so a dense feature-space oracle can check the
    posterior independently, including p > n. Intercept uncertainty is sampled.
    """
    count = len(observed)
    penalty = float(ridge) * count
    if count < 2 or not np.isfinite(penalty) or penalty <= 0 or chi_square <= 0:
        raise ValueError("MICE requires two observed target values and positive ridge/variance draw")
    column = z[:, target]
    kernel = gram - np.outer(column, column)
    cross = kernel[:, observed].copy()
    training = cross[observed].copy()
    column_mean = training.mean(axis=0)
    grand_mean = float(column_mean.mean())
    cross -= cross.mean(axis=1, keepdims=True)
    cross -= column_mean[None, :]
    cross += grand_mean
    training -= training.mean(axis=1, keepdims=True)
    training -= column_mean[None, :]
    training += grand_mean
    training = (training + training.T) * 0.5
    training.flat[::count + 1] += penalty
    try:
        factor = cho_factor(training, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MICE conditional covariance is not positive definite; increase ridge") from exc

    response = column[observed]
    response_mean = float(response.mean())
    centered_response = response - response_mean
    # A prior coefficient draw projected to sample space. The target's own
    # coefficient is excluded, while retaining a fixed-size RNG draw per fit.
    prior = (z @ normal_predictors - column * normal_predictors[target]) / np.sqrt(penalty)
    prior -= prior[observed].mean()
    solutions = cho_solve(factor, np.column_stack((centered_response,
                         prior[observed] + normal_observed)), check_finite=False)
    mean = response_mean + cross @ solutions[:, 0]
    # Equivalent to yc'yc - yc'X (X'X+lambda I)^-1 X'yc, without cancellation.
    scale = penalty * float(centered_response @ solutions[:, 0])
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("MICE posterior variance is invalid")
    variance = scale / chi_square
    noise = prior - cross @ solutions[:, 1] + normal_intercept / np.sqrt(count)
    # PMM matches conditional means with parameter uncertainty. It does NOT
    # add independent response residuals to the matching predictions.
    drawn = mean + np.sqrt(variance) * noise
    return mean, drawn, variance


def _positive_integer(params, key, default):
    value = params.get(key, default)
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError(f"MICE {key} must be a positive integer")
    return int(value)


def run_mice_pmm(matrix, params, *, seed, folder=None):
    """Fit independent full conditional chains, save draws, return their mean."""
    started = time.perf_counter()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or np.isinf(matrix).any():
        raise ValueError("MICE expects a nonempty 2D matrix of finite values or NaN")
    m = _positive_integer(params, "m", 5)
    iterations = _positive_integer(params, "max_iter", 5)
    donors = _positive_integer(params, "donors", 5)
    ridge = float(params.get("ridge", 1e-5))
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("MICE ridge must be finite and positive")
    solver = params.get("mice_solver", "cached")
    if solver not in ("cached", "reference"):
        raise ValueError("MICE solver must be cached or reference")
    observed = np.isfinite(matrix)
    counts = observed.sum(axis=0)
    targets = np.flatnonzero(~observed.all(axis=0))
    if (counts == 0).any():
        raise ValueError("MICE PMM cannot impute an entirely missing feature without observed donors")
    center = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    constant = np.nanmax(matrix, axis=0) == np.nanmin(matrix, axis=0)
    scale[constant] = 1.0
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("MICE could not standardize the observed values safely")
    standardized = (matrix - center) / scale
    if not np.isfinite(standardized[observed]).all():
        raise ValueError("MICE could not standardize the observed values")
    observed_rows = {j: np.flatnonzero(observed[:, j]) for j in targets}
    missing_rows = {j: np.flatnonzero(~observed[:, j]) for j in targets}
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(m)]
    draws = np.empty((m, *matrix.shape), dtype=np.float64)
    trace = []
    for chain, chain_seed in enumerate(seeds):
        rng = np.random.default_rng(chain_seed)
        values = matrix.copy()
        z = standardized.copy()
        for j in targets:
            selected = rng.choice(observed_rows[j], size=len(missing_rows[j]))
            values[missing_rows[j], j] = matrix[selected, j]
            z[missing_rows[j], j] = standardized[selected, j]
        for iteration in range(iterations):
            gram = z @ z.T
            for j in targets:
                rows, missing = observed_rows[j], missing_rows[j]
                # With one distinct observed value, every admissible PMM donor
                # has that value. No variance model can be identified or needs
                # to be invented. Initialization already filled that sole value.
                if constant[j]:
                    continue
                old = z[:, j].copy()
                # Reference recomputes the Gram each conditional, providing a
                # replay path for validating the incremental execution cache.
                if solver == "reference":
                    gram = z @ z.T
                mean, drawn, _ = conditional_predictions(z, gram, j, rows,
                    rng.standard_normal(matrix.shape[1]), rng.standard_normal(len(rows)),
                    rng.standard_normal(), rng.chisquare(len(rows) - 1), ridge)
                order = rng.permutation(len(rows))
                distance = np.abs(mean[rows[order], None] - drawn[missing][None, :])
                nearest = np.argsort(distance, axis=0, kind="stable")[:min(donors, len(rows))]
                selected = rows[order[nearest[rng.integers(len(nearest), size=len(missing)),
                                             np.arange(len(missing))]]]
                values[missing, j] = matrix[selected, j]
                z[missing, j] = standardized[selected, j]
                if solver == "cached":
                    gram -= np.outer(old, old)
                    gram += np.outer(z[:, j], z[:, j])
            trace.append({"chain": chain, "iteration": iteration + 1,
                          "elapsed_s": time.perf_counter() - started})
            print(f"MICE {chain+1}/{m}, iteration {iteration+1}/{iterations}", flush=True)
        draws[chain] = values
    prediction = draws.mean(axis=0)
    prediction[observed] = matrix[observed]
    metadata = {"implementation": "python-linear-mice-pmm", "implementation_version": 1,
        "device": "cpu", "solver": solver, "mice_imputations": m,
        "mice_iterations": iterations, "donors": donors, "ridge": ridge,
        "ridge_penalty": "ridge * observed target count on fixed standardized predictors",
        "posterior": "conjugate ridge beta prior, flat intercept, scale prior 1/sigma2; df=n_observed-1",
        "matching": "type-1 PMM; randomly selected original observed donors",
        "all_predictors": True, "feature_fits": m * iterations * int((~constant[targets]).sum()),
        "conditional_updates": m * iterations * len(targets),
        "constant_donor_features": np.flatnonzero(constant & ~observed.all(axis=0)).tolist(),
        "single_observation_features": np.flatnonzero(counts == 1).tolist(),
        "chain_seeds": seeds, "iteration_trace": trace,
        "aggregation": "pointwise mean of completed matrices; no inferential pooling",
        "training_protocol": "fit common evaluation-corrupted observations",
        "fit_s": time.perf_counter() - started}
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "mice_draws.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, values=draws, center=center, scale=scale, observed_counts=counts)
        temporary.replace(folder / "mice_draws.npz")
        metadata["mice_draws_file"] = "mice_draws.npz"
    return prediction, metadata
