"""Shared evaluation masking followed by model-specific training corruption.

All masks are boolean. *_hidden means True is removed; *_visible means True is
available. Keep the evaluation truths outside the matrices passed to methods.
"""

from dataclasses import dataclass

import numpy as np


def _mask(values, shape, name):
    mask = np.asarray(values)
    if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return mask.copy()


@dataclass(frozen=True)
class BenchmarkMasks:
    """Immutable, validated masks for one evaluation repeat."""

    observed: np.ndarray
    evaluation_hidden: np.ndarray
    training_hidden: np.ndarray

    def __post_init__(self):
        observed = np.asarray(self.observed)
        if observed.dtype != np.bool_ or observed.ndim != 2:
            raise ValueError("observed must be a 2-D boolean mask")
        observed = observed.copy()
        evaluation = _mask(self.evaluation_hidden, observed.shape, "evaluation_hidden")
        training = _mask(self.training_hidden, observed.shape, "training_hidden")
        if (evaluation & ~observed).any():
            raise ValueError(
                "evaluation masking may only remove originally observed entries"
            )
        available = observed & ~evaluation
        if (training & ~available).any():
            raise ValueError(
                "training masking may only remove entries remaining after evaluation masking"
            )
        for name, mask in (
            ("observed", observed),
            ("evaluation_hidden", evaluation),
            ("training_hidden", training),
        ):
            mask.setflags(write=False)
            object.__setattr__(self, name, mask)

    @property
    def available(self):
        """Common input for all methods; also the supervised target mask."""
        return self.observed & ~self.evaluation_hidden

    @property
    def training_visible(self):
        return self.available & ~self.training_hidden

    def method_input(self, values):
        values = np.asarray(values, dtype=float)
        if (
            values.shape != self.observed.shape
            or not np.isfinite(values[self.observed]).all()
        ):
            raise ValueError(
                "values must match the masks and be finite at observed entries"
            )
        return np.where(self.available, values, np.nan)

    def training_input(self, values):
        values = self.method_input(values)
        return np.where(self.training_visible, values, np.nan)


def sample_hidden_mask(eligible, rate, rng):
    """Sample an MCAR hidden mask from eligible cells, without replacement.

    The rate is a fraction of eligible observations, rounded to the nearest
    count. MNAR generators can supply their own boolean masks to BenchmarkMasks.
    """
    eligible = np.asarray(eligible)
    if eligible.ndim != 2 or eligible.dtype != np.bool_:
        raise ValueError("eligible must be a 2-D boolean mask")
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    candidates = np.flatnonzero(eligible)
    count = int(np.floor(candidates.size * rate + 0.5))
    hidden = np.zeros(eligible.shape, dtype=bool)
    if count:
        hidden.flat[rng.choice(candidates, size=count, replace=False)] = True
    return hidden


def make_benchmark_masks(observed, evaluation_rate=0.2, training_rate=0.2, seed=42):
    """Create one shared MCAR evaluation mask, then corrupt only remaining cells.

    Independent RNG streams prevent training settings from changing the test
    mask. Call once per repeat and reuse the evaluation mask for every method.
    """
    observed = np.asarray(observed)
    if observed.ndim != 2 or observed.dtype != np.bool_:
        raise ValueError("observed must be a 2-D boolean mask")
    evaluation_seed, training_seed = np.random.SeedSequence(seed).spawn(2)
    evaluation = sample_hidden_mask(
        observed, evaluation_rate, np.random.default_rng(evaluation_seed)
    )
    available = observed & ~evaluation
    training = sample_hidden_mask(
        available, training_rate, np.random.default_rng(training_seed)
    )
    return BenchmarkMasks(observed, evaluation, training)
