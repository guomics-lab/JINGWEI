"""Compatibility runner using the original Lightning Imputer wrappers."""

import numpy as np
import time


def run_jingwei_lightning(name, matrix, training_hidden, params, *, device, seed):
    started = time.perf_counter()
    available = np.isfinite(matrix)
    import torch
    import pytorch_lightning as pl
    from ._native.dmf import DMFImputer
    from ._native.dcae import DCAEImputer

    pl.seed_everything(seed, workers=True)
    counts = available.sum(axis=0)
    mean = np.divide(
        np.where(available, matrix, 0).sum(axis=0),
        counts,
        out=np.zeros(matrix.shape[1]),
        where=counts > 0,
    )
    variance = np.divide(
        np.where(available, (matrix - mean) ** 2, 0).sum(axis=0),
        counts,
        out=np.ones(matrix.shape[1]),
        where=counts > 0,
    )
    sd = np.sqrt(variance)
    sd[sd < 1e-8] = 1.0
    normalized = np.where(available, (matrix - mean) / sd, 0).astype("float32")
    common = dict(
        full_data_tensor=torch.tensor(normalized),
        full_mask_tensor=torch.tensor(
            available & ~training_hidden, dtype=torch.float32
        ),
        supervision_mask_tensor=torch.tensor(available, dtype=torch.float32),
        batch_size=params.get("batch_size", 64),
        num_workers=0,
    )
    if name == "jingwei-dmf":
        model = DMFImputer(
            **common,
            embedding_dim=params.get("latent_dim", 64),
            hidden_dims=params.get("hidden_dims", [256, 128]),
            mask_weight=params.get("mask_weight", 0.5),
            lr=params.get("learning_rate", 0.001),
        )
    else:
        model = DCAEImputer(
            **common,
            ae_dim=params.get("latent_dim", 64),
            mask_predictor_hidden_dim=params.get("hidden_dims", [256])[0],
            lambda_mask=params.get("mask_weight", 0.5),
            num_encoder_blocks=params.get("encoder_blocks", 2),
            num_decoder_blocks=params.get("decoder_blocks", 2),
            dilation=params.get("dilation", 2),
            learning_rate=params.get("learning_rate", 0.001),
        )
    class TrainingDeviceRecorder(pl.Callback):
        def on_train_start(self, trainer, module):
            self.device = next(module.parameters()).device

    recorder = TrainingDeviceRecorder()
    trainer = pl.Trainer(
        max_epochs=params.get("max_epochs", 50),
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[recorder],
    )
    if device == "cuda":
        torch.cuda.synchronize()
    initialized = time.perf_counter()
    trainer.fit(model)
    if device == "cuda":
        torch.cuda.synchronize()
    trained = time.perf_counter()
    if recorder.device.type != device:
        raise RuntimeError(f"requested {device}, model trained on {recorder.device}")
    # Lightning tears down the accelerator after fit and moves the module to CPU.
    # Restore the requested device for inference, including all registered buffers.
    model.to(device)
    prediction = model.get_imputed_data().detach().cpu().numpy() * sd + mean
    finished = time.perf_counter()
    metadata = {
        "device": device,
        "training_device": str(recorder.device),
        "training_protocol": "fixed second mask; supervision on all remaining observations",
        "normalization": "per-feature mean/std fitted after evaluation masking",
    }
    actual_device = next(model.parameters()).device
    if actual_device.type != device:
        raise RuntimeError(f"requested {device}, model trained on {actual_device}")
    metadata.update(actual_device=str(actual_device), torch_version=torch.__version__, training_engine="lightning",
                    precision="float32", epochs_trained=trainer.current_epoch, training_steps=trainer.global_step,
                    initialization_s=initialized-started, training_s=trained-initialized, prediction_s=finished-trained)
    return prediction, metadata
