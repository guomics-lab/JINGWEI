"""Reproducible parameters from the repository's original benchmark script.

This preset reproduces model/training settings, not the old script's column
dropping, masking bug, preprocessing leakage or per-method score filtering.
"""

ORIGINAL_BENCHMARK = {
    "jingwei-dmf": {"max_epochs": 50, "batch_size": 1024, "latent_dim": 64,
                    "hidden_dims": [256, 128], "learning_rate": 5e-4, "mask_weight": .5},
    "jingwei-dcae": {"max_epochs": 50, "batch_size": 1024, "latent_dim": 256,
                     "mask_predictor_hidden_dim": 128, "encoder_blocks": 3,
                     "decoder_blocks": 3, "dilation": 3, "learning_rate": 5e-5, "mask_weight": .5},
}


def preset_parameters(preset, method):
    if preset == "original-benchmark":
        return dict(ORIGINAL_BENCHMARK.get(method, {}))
    if preset != "standard":
        raise ValueError(f"Unknown JINGWEI preset: {preset}")
    return {}
