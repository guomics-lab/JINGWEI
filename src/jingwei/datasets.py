import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None, skiprows=1)
        data_values = df.iloc[:, 1:].values.astype("float32")
        is_missing = (data_values <= 0) | np.isnan(data_values)
        initial_data_mask = torch.tensor(~is_missing, dtype=torch.float32)
        df_data_only = df.iloc[:, 1:]
        df_filled = df_data_only.fillna(0)
        self.data_filled = torch.tensor(df_filled.values.astype("float32"))

        data_for_stats = self.data_filled.clone()
        data_for_stats[initial_data_mask == 0] = float("nan")
        self.mean = torch.nanmean(data_for_stats, dim=0, keepdim=True)
        self.std = torch.sqrt(
            torch.nanmean((data_for_stats - self.mean) ** 2, dim=0, keepdim=True)
        )
        self.mean[torch.isnan(self.mean)] = 0.0
        self.std[torch.isnan(self.std)] = 1.0
        self.std[self.std < 1e-8] = 1e-8
        self.data_normalized = (self.data_filled - self.mean) / self.std
        dynamic_mask = self.get_mask()
        self.mask = initial_data_mask * dynamic_mask

    def __len__(self):
        return self.data_normalized.size(0)

    def __getitem__(self, idx):
        sample = self.data_normalized[idx]
        mask = self.mask[idx]
        return sample, mask

    def get_feature_dim(self):
        return self.data_normalized.size(1)

    def get_mask(self):
        data_norm = self.data_normalized
        mean = self.mean
        std = self.std
        dynamic_mask = torch.ones_like(data_norm, dtype=torch.float32)
        random_values = torch.rand_like(data_norm)
        dynamic_mask[(data_norm < mean - 3 * std) & (random_values < 0.9)] = 0.0
        dynamic_mask[
            (mean - 3 * std < data_norm)
            & (data_norm < mean - 2 * std)
            & (random_values < 0.5)
        ] = 0.0
        dynamic_mask[
            (mean - 2 * std < data_norm)
            & (data_norm < mean - 1 * std)
            & (random_values < 0.3)
        ] = 0.0
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
