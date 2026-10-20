"""Mask-aware convolutional encoder with a global reconstruction bottleneck.

This is the current DCAE implementation. It uses only the values and mask supplied by its
caller; no fitting statistics or evaluation targets are stored in the network.
"""

import torch
from torch import nn


AE_ARCHITECTURES = ("native", "global")
MAX_GLOBAL_AE_PARAMETERS = 50_000_000


def build_dcae(architecture="global", *, dropout=.1, **kwargs):
    if architecture == "native":
        from .models import LegacyDCAE
        return LegacyDCAE(**kwargs)
    if architecture == "global":
        return GlobalDCAE(**kwargs, dropout=dropout)
    raise ValueError(f"unknown AE architecture: {architecture}")


class _ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _ResidualConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.branch = nn.Sequential(_ChannelNorm(width), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation))
        self.scale = nn.Parameter(torch.full((1, width, 1), .1))

    def forward(self, x):
        return x + self.scale * self.branch(x)


class _ResidualDense(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.branch = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.scale = nn.Parameter(torch.full((width,), .1))

    def forward(self, x):
        return x + self.scale * self.branch(x)


class GlobalDCAE(nn.Module):
    """DCAE variant with explicit missingness and feature-specific decoding.

    Observations whose normalized value is zero remain observed when the mask
    says so. Unavailable values are discarded before any learned operation.
    Without an explicit mask, finite input values are treated as observed.
    """

    def __init__(self, input_feature_dim, ae_dim, mask_predictor_hidden_dim=128,
                 num_encoder_blocks=3, num_decoder_blocks=3, dilation=3, dropout=.1):
        super().__init__()
        dimensions = (input_feature_dim, ae_dim, mask_predictor_hidden_dim,
                      num_encoder_blocks, num_decoder_blocks, dilation)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("AE dimensions, block counts and dilation must be positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("AE dropout must be in [0, 1)")
        self.input_feature_dim = input_feature_dim
        self.ae_dim = ae_dim
        self.mask_aware = True
        self.architecture_version = "global-ae-v1"
        width = 16
        length = input_feature_dim
        for _ in range(num_encoder_blocks):
            length = (length + 1) // 2
        hidden = mask_predictor_hidden_dim
        estimated = (112 + 1616 * num_encoder_blocks + width * length * ae_dim + ae_dim
                     + (2 * ae_dim if ae_dim > 1 else 0) + 2 * (ae_dim * hidden + hidden)
                     + num_decoder_blocks * (2 * hidden**2 + 5 * hidden)
                     + 2 * (hidden * input_feature_dim + input_feature_dim))
        if estimated > MAX_GLOBAL_AE_PARAMETERS:
            raise ValueError(f"global AE would require {estimated:,} parameters; reduce latent/hidden dimensions or decoder blocks")
        length = input_feature_dim
        encoder = [nn.Conv1d(2, width, 3, padding=1), nn.GELU()]
        for _ in range(num_encoder_blocks):
            encoder.extend((_ResidualConv(width, dilation), nn.Conv1d(width, width, 3, stride=2, padding=1)))
            length = (length + 1) // 2
        self.encoder = nn.Sequential(*encoder)
        self.bottleneck = nn.Sequential(nn.Flatten(1), nn.Linear(width * length, ae_dim),
            nn.LayerNorm(ae_dim) if ae_dim > 1 else nn.Identity(), nn.GELU(), nn.Dropout(dropout))
        decoder = [nn.Linear(ae_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        decoder.extend(_ResidualDense(hidden, dropout) for _ in range(num_decoder_blocks))
        self.decoder = nn.Sequential(*decoder)
        self.reconstruction = nn.Linear(hidden, input_feature_dim)
        self.mask_predictor = nn.Sequential(nn.Linear(ae_dim, hidden), nn.GELU(),
            nn.Linear(hidden, input_feature_dim))
        # Start near the normalized protein mean, without blocking encoder
        # gradients as a zero-initialized output matrix would at the first step.
        nn.init.normal_(self.reconstruction.weight, std=.001)
        nn.init.zeros_(self.reconstruction.bias)

    def encode(self, x, observed_mask=None):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 2 or x.shape[1] != self.input_feature_dim:
            raise ValueError("AE input must be samples by the configured protein count")
        if observed_mask is None:
            observed_mask = torch.isfinite(x)
        elif observed_mask.ndim == 3 and observed_mask.shape[1] == 1:
            observed_mask = observed_mask[:, 0]
        if observed_mask.shape != x.shape:
            raise ValueError("AE values and observation mask must have matching shapes")
        observed = observed_mask.bool()
        values = torch.where(observed, x, 0)
        channels = torch.stack((values, observed.to(x.dtype)), dim=1)
        return self.bottleneck(self.encoder(channels))

    def reconstruct(self, x, observed_mask=None):
        return self.reconstruction(self.decoder(self.encode(x, observed_mask)))

    def forward(self, x, observed_mask=None):
        latent = self.encode(x, observed_mask)
        return self.reconstruction(self.decoder(latent)), self.mask_predictor(latent)
