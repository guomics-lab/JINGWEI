"""MNAR probabilities and sequential evaluation/training masks."""

import numpy as np
from scipy.special import expit
from .masking import BenchmarkMasks, sample_hidden_mask

ORIGINAL_MECHANISM = "mnar-jingwei-original"
MECHANISMS = ("mcar", "mnar-logistic", "mnar-threshold", "mnar-jingwei", ORIGINAL_MECHANISM)


def uses_mask_rate(mechanism, uncalibrated=False):
    """Whether the requested rate controls this stage (shared by CLI and GUI)."""
    return mechanism == "mcar" or (mechanism != ORIGINAL_MECHANISM and not uncalibrated)


def _original_probabilities(values, eligible):
    """Replay historical CSVDataset.get_mask, including its mixed-scale comparison.

    This is deliberately NOT the corrected z-score rule or a quantile rule.
    Statistics are fitted only to this stage's eligible entries, so evaluation
    truth remains excluded when constructing the second, training mask.
    """
    import torch

    # pandas -> .values -> torch.tensor in the historical CSV loader preserves
    # column-major strides. This also determines rand_like's per-cell sequence.
    data = torch.from_numpy(np.array(np.where(eligible, values, np.nan), dtype=np.float32, order="F"))
    if not torch.isfinite(data[torch.from_numpy(eligible)]).all():
        raise ValueError("original-repository masking requires float32-finite values")
    mean = torch.nanmean(data, dim=0, keepdim=True)
    std = torch.sqrt(torch.nanmean((data - mean) ** 2, dim=0, keepdim=True))
    mean[torch.isnan(mean)] = 0.0
    std[torch.isnan(std)] = 1.0
    std[std < 1e-8] = 1e-8
    normalized = (data - mean) / std
    p = torch.zeros_like(data)
    p[normalized < mean - 3 * std] = 0.9
    p[(mean - 3 * std < normalized) & (normalized < mean - 2 * std)] = 0.5
    p[(mean - 2 * std < normalized) & (normalized < mean - std)] = 0.3
    return np.where(eligible, p.numpy(), 0.0)


def missing_probabilities(
    values,
    eligible,
    mechanism,
    *,
    level="entry",
    slope=-1.0,
    low_quantile=0.2,
    high_quantile=0.6,
    low_probability=0.9,
    high_probability=0.2,
    threshold_probability=0.05,
):
    values, eligible = np.asarray(values, dtype=float), np.asarray(eligible, dtype=bool)
    if values.ndim != 2 or values.shape != eligible.shape or not np.isfinite(values[eligible]).all():
        raise ValueError("masking values must be finite at eligible entries")
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    if mechanism == ORIGINAL_MECHANISM:
        return _original_probabilities(values, eligible)
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    if any(
        not np.isfinite(p) or not 0 <= p <= 1
        for p in (low_probability, high_probability, threshold_probability)
    ):
        raise ValueError("probabilities must be finite and in [0,1]")
    if low_probability < high_probability or not np.isfinite(slope) or slope > 0:
        raise ValueError("MNAR probabilities must decrease with intensity")
    if not eligible.any():
        return np.zeros(values.shape)
    if level == "entry":
        intensities, population = values, values[eligible]
    elif level == "feature-mean":
        counts = eligible.sum(axis=0)
        means = np.divide(
            np.where(eligible, values, 0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1]),
            where=counts > 0,
        )
        intensities, population = (
            np.broadcast_to(means, values.shape),
            means[counts > 0],
        )
    else:
        raise ValueError("mask level must be entry or feature-mean")
    low, high = np.quantile(population, [low_quantile, high_quantile])
    if mechanism == "mcar":
        probabilities = np.full(values.shape, 0.2)
    elif mechanism == "mnar-logistic":
        probabilities = expit(
            slope
            * (intensities - np.mean(population))
            / (float(np.std(population)) or 1.0)
        )
    elif mechanism == "mnar-threshold":
        probabilities = np.where(
            intensities <= low, low_probability, threshold_probability
        )
    else:
        transition = low_probability + (high_probability - low_probability) * (
            intensities - low
        ) / max(high - low, 1e-12)
        probabilities = np.where(
            intensities <= low,
            low_probability,
            np.where(intensities >= high, 0.0, transition),
        )
    return np.where(eligible, probabilities, 0.0)


def calibrate_probabilities(probabilities, eligible, rate):
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("mask rate must be in [0,1]")
    p = probabilities[eligible]
    if not p.size or rate == 0:
        return np.zeros_like(probabilities)
    supported_rate = np.mean(p > 0)
    if rate > supported_rate + 1e-12:
        raise ValueError(
            f"requested rate exceeds mechanism support ({supported_rate:.3f})"
        )
    if np.isclose(rate, supported_rate, rtol=0, atol=1e-14):
        return (probabilities > 0).astype(float)
    lower, upper = 0.0, 1.0
    while np.minimum(upper * p, 1).mean() < rate:
        upper *= 2
    for _ in range(60):
        middle = (lower + upper) / 2
        if np.minimum(middle * p, 1).mean() < rate:
            lower = middle
        else:
            upper = middle
    return np.minimum(probabilities * upper, 1)


def _retain_marginals(eligible, hidden, rng):
    repaired = hidden.copy()
    for row in np.flatnonzero(
        eligible.any(axis=1) & ~(eligible & ~repaired).any(axis=1)
    ):
        repaired[row, rng.choice(np.flatnonzero(eligible[row] & repaired[row]))] = False
    for column in np.flatnonzero(
        eligible.any(axis=0) & ~(eligible & ~repaired).any(axis=0)
    ):
        repaired[
            rng.choice(np.flatnonzero(eligible[:, column] & repaired[:, column])),
            column,
        ] = False
    return repaired, int(np.count_nonzero(hidden & ~repaired))


def protocol_masks(
    values,
    observed,
    *,
    mechanism="mcar",
    training_mechanism=None,
    evaluation_rate=0.2,
    training_rate=0.2,
    seed=42,
    preserve_marginals=True,
    working_values=None,
    **parameters,
):
    values, observed = np.asarray(values, dtype=float), np.asarray(observed)
    if observed.dtype != np.bool_ or observed.ndim != 2 or observed.shape != values.shape:
        raise ValueError("observed must be a 2-D boolean mask matching values")
    working_values = values if working_values is None else np.asarray(working_values, dtype=float)
    if working_values.shape != values.shape:
        raise ValueError("working masking values must match values")
    evaluation_seed, training_seed = np.random.SeedSequence(seed).spawn(2)
    metadata = {}

    def draw(eligible, name, rate, stage_seed, stage):
        rng = np.random.default_rng(stage_seed)
        original = name == ORIGINAL_MECHANISM
        random_seed = None
        expected_rate = None
        if name == "mcar":
            hidden = sample_hidden_mask(eligible, 0.2 if rate is None else rate, rng)
        elif original:
            import torch

            probabilities = missing_probabilities(working_values, eligible, name)
            random_seed = int(stage_seed.generate_state(1)[0])
            generator = torch.Generator().manual_seed(random_seed)
            random_values = torch.empty_strided(eligible.shape, (1, eligible.shape[0]), dtype=torch.float32).uniform_(generator=generator).numpy()
            hidden = eligible & (random_values < probabilities)
            if rate == 0:
                hidden[:] = False
            else:
                rate = None  # Original probabilities are never rate-calibrated.
            expected_rate = float(probabilities[eligible].mean()) if eligible.any() else 0.0
        else:
            probabilities = missing_probabilities(values, eligible, name, **parameters)
            if rate is not None:
                probabilities = calibrate_probabilities(probabilities, eligible, rate)
            hidden = eligible & (rng.random(eligible.shape) < probabilities)
            expected_rate = float(probabilities[eligible].mean()) if eligible.any() else 0.0
        repairs = 0
        if preserve_marginals and not original:
            hidden, repairs = _retain_marginals(eligible, hidden, rng)
        metadata[stage] = {
            "mechanism": name,
            "eligible_count": int(eligible.sum()),
            "hidden_count": int(hidden.sum()),
            "requested_rate": rate,
            "actual_rate": (
                float(hidden.sum() / eligible.sum()) if eligible.any() else 0.0
            ),
            "marginal_repairs": repairs,
            "probability_mode": ("original-repository" if original else
                                 "fixed-count" if name == "mcar" else
                                 "uncalibrated" if rate is None else "calibrated"),
            "probability_mean": expected_rate,
            "value_scale": "working" if original else "declared-input",
            "statistics_scope": "eligible-per-feature" if original else parameters.get("level", "entry"),
            "marginal_preservation": bool(preserve_marginals and not original),
            "rng": "torch-cpu-float32-column-major" if original else "numpy-PCG64",
            "random_seed": random_seed,
            "rule": ("CSVDataset.get_mask: z compared to raw mean-k*std; strict bands; p=0.9/0.5/0.3"
                     if original else name),
        }
        return hidden

    evaluation = draw(
        observed,
        mechanism,
        evaluation_rate,
        evaluation_seed,
        "evaluation",
    )
    available = observed & ~evaluation
    if training_rate == 0:
        # Benchmark orchestration always takes this path. The second-mask
        # utility remains available to an explicit model-owned training caller.
        training = np.zeros_like(observed)
        metadata["training"] = {
            "mechanism": None, "eligible_count": int(available.sum()),
            "hidden_count": 0, "actual_rate": 0.0, "requested_rate": 0.0,
            "probability_mode": "not-applied", "random_seed": None,
            "rule": "no platform training mask; each method owns its training procedure",
        }
    else:
        training = draw(
            available,
            training_mechanism or mechanism,
            training_rate,
            training_seed,
            "training",
        )
    return BenchmarkMasks(observed, evaluation, training), metadata
