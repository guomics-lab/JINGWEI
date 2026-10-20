"""MNAR probabilities and sequential evaluation/training masks."""

import numpy as np
from scipy.special import expit
from .masking import BenchmarkMasks, sample_hidden_mask

MECHANISMS = ("mcar", "mnar-logistic", "mnar-threshold", "mnar-jingwei")


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
    if values.shape != eligible.shape or not np.isfinite(values[eligible]).all():
        raise ValueError("masking values must be finite at eligible entries")
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {mechanism}")
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
    **parameters,
):
    evaluation_seed, training_seed = np.random.SeedSequence(seed).spawn(2)
    metadata = {}

    def draw(eligible, name, rate, rng, stage):
        if name == "mcar":
            hidden = sample_hidden_mask(eligible, 0.2 if rate is None else rate, rng)
        else:
            probabilities = missing_probabilities(values, eligible, name, **parameters)
            if rate is not None:
                probabilities = calibrate_probabilities(probabilities, eligible, rate)
            hidden = eligible & (rng.random(eligible.shape) < probabilities)
        repairs = 0
        if preserve_marginals:
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
        }
        return hidden

    evaluation = draw(
        observed,
        mechanism,
        evaluation_rate,
        np.random.default_rng(evaluation_seed),
        "evaluation",
    )
    available = observed & ~evaluation
    training = draw(
        available,
        training_mechanism or mechanism,
        training_rate,
        np.random.default_rng(training_seed),
        "training",
    )
    return BenchmarkMasks(observed, evaluation, training), metadata
