import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from .masking import BenchmarkMasks


class CSVDataset(Dataset):
    def __init__(
        self, csv_file, *, masks=None, seed=None, missing_values="nonpositive"
    ):
        """Load CSV; supplied masks define one evaluation/training repeat.

        With masks=None, only training corruption is generated (imputation use).
        With BenchmarkMasks, test and training-hidden entries are fixed by the
        benchmark caller. Statistics and supervision exclude evaluation truths.
        Use missing_values='nan' for data with valid zero or negative values.
        """
        df = pd.read_csv(csv_file, header=None, skiprows=1)
        data_values = df.iloc[:, 1:].values.astype("float32")
        if np.isinf(data_values).any():
            raise ValueError("input intensities must not contain infinity")
        if missing_values not in ("nonpositive", "nan"):
            raise ValueError("missing_values must be 'nonpositive' or 'nan'")
        observed = np.isfinite(data_values)
        if missing_values == "nonpositive":
            observed &= data_values > 0
        if masks is None:
            empty = np.zeros(observed.shape, dtype=bool)
            base_masks = BenchmarkMasks(observed, empty, empty)
        else:
            if not isinstance(masks, BenchmarkMasks) or not np.array_equal(
                masks.observed, observed
            ):
                raise ValueError("benchmark observation mask must match the input CSV")
            base_masks = masks
        self.observation_mask = torch.tensor(observed, dtype=torch.float32)
        self.evaluation_mask = torch.tensor(
            base_masks.evaluation_hidden.copy(), dtype=torch.bool
        )
        self.supervision_mask = torch.tensor(base_masks.available, dtype=torch.float32)
        if not self.supervision_mask.any():
            raise ValueError(
                "no observed training values remain after evaluation masking"
            )
        self.data_filled = torch.tensor(np.where(observed, data_values, 0.0))

        data_for_stats = self.data_filled.clone()
        data_for_stats[self.supervision_mask == 0] = float("nan")
        self.mean = torch.nanmean(data_for_stats, dim=0, keepdim=True)
        self.std = torch.sqrt(
            torch.nanmean((data_for_stats - self.mean) ** 2, dim=0, keepdim=True)
        )
        self.mean[torch.isnan(self.mean)] = 0.0
        self.std[torch.isnan(self.std)] = 1.0
        self.std[self.std < 1e-8] = 1e-8
        normalized = (self.data_filled - self.mean) / self.std
        self.data_normalized = torch.where(
            self.supervision_mask.bool(), normalized, 0.0
        )
        if masks is None:
            training_visible = self.get_mask(seed=seed).bool().numpy()
            training_hidden = base_masks.available & ~training_visible
            self.benchmark_masks = BenchmarkMasks(
                observed, base_masks.evaluation_hidden, training_hidden
            )
        else:
            self.benchmark_masks = masks
        self.mask = torch.tensor(
            self.benchmark_masks.training_visible, dtype=torch.float32
        )

    def __len__(self):
        return self.data_normalized.size(0)

    def __getitem__(self, idx):
        sample = self.data_normalized[idx]
        mask = self.mask[idx]
        return sample, mask

    def get_feature_dim(self):
        return self.data_normalized.size(1)

    def get_mask(self, seed=None):
        data_norm = self.data_normalized
        dynamic_mask = self.supervision_mask.clone()
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        # The original pandas/torch loader draws in column-major order. The
        # validated masks above are C-contiguous, so rand_like here would map
        # the same seed to different cells even with the same probabilities.
        random_values = torch.empty_strided(data_norm.shape, (1, data_norm.shape[0]),
            dtype=data_norm.dtype, device=data_norm.device).uniform_(generator=generator)
        # Preserve the benchmark's historical get_mask rule, including its
        # mixed-scale comparisons and strict interval boundaries.
        mean, std = self.mean, self.std
        dynamic_mask[(data_norm < mean - 3 * std) & (random_values < 0.9)] = 0.0
        dynamic_mask[(mean - 3 * std < data_norm) & (data_norm < mean - 2 * std) & (random_values < 0.5)] = 0.0
        dynamic_mask[(mean - 2 * std < data_norm) & (data_norm < mean - std) & (random_values < 0.3)] = 0.0
        return dynamic_mask

    def inverse_transform(self, normalized_tensor):
        if normalized_tensor.dim() == 3:
            normalized_tensor = normalized_tensor.squeeze(1)
        mean = self.mean.squeeze(0) if self.mean.shape[0] == 1 else self.mean
        std = self.std.squeeze(0) if self.std.shape[0] == 1 else self.std

        if normalized_tensor.dim() == 1:
            if normalized_tensor.shape[0] != mean.shape[0]:
                raise ValueError(
                    f"Shape mismatch: tensor {normalized_tensor.shape}, mean {mean.shape}"
                )
            original = normalized_tensor * std + mean
        elif normalized_tensor.dim() == 2:
            if normalized_tensor.shape[1] != mean.shape[0]:
                raise ValueError(
                    f"Shape mismatch: tensor {normalized_tensor.shape}, mean {mean.shape}"
                )
            original = normalized_tensor * std.unsqueeze(0) + mean.unsqueeze(0)
        else:
            raise ValueError(
                "Input tensor dimension error: expected 1, 2 or 3, got "
                f"{normalized_tensor.dim()}"
            )

        return original
