"""Compatibility runner using the original Lightning Imputer wrappers."""

import numpy as np
import time


def run_jingwei_lightning(name, matrix, training_hidden, params, *, device, seed, folder=None):
    started = time.perf_counter()
    from .convergence import validation_split, ValidationMonitor
    original_available = np.isfinite(matrix)
    validation_hidden = validation_split(matrix, training_hidden, params, seed, folder)
    original_matrix = matrix
    matrix = np.where(validation_hidden, np.nan, matrix)
    available = np.isfinite(matrix)
    monitor = ValidationMonitor(params, enabled=validation_hidden.any(), validation_mask=validation_hidden)
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
            mask_predictor_hidden_dim=params.get("mask_predictor_hidden_dim", params.get("hidden_dims", [256])[0]),
            lambda_mask=params.get("mask_weight", 0.5),
            num_encoder_blocks=params.get("encoder_blocks", 2),
            num_decoder_blocks=params.get("decoder_blocks", 2),
            dilation=params.get("dilation", 2),
            learning_rate=params.get("learning_rate", 0.001),
            ae_architecture=params.get("ae_architecture", "global"),
            ae_dropout=params.get("ae_dropout", .1),
        )
        print(f"AE architecture={params.get('ae_architecture', 'global')} | parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    class TrainingDeviceRecorder(pl.Callback):
        def on_train_start(self, trainer, module):
            self.device = next(module.parameters()).device

        def on_train_epoch_end(self, trainer, module):
            if not monitor.enabled:
                return
            module.eval()
            with torch.no_grad():
                prediction = module.get_imputed_data().detach().cpu().numpy() * sd + mean
            loss = float(np.mean((prediction[validation_hidden] - original_matrix[validation_hidden]) ** 2))
            trainer.should_stop = monitor.observe(trainer.current_epoch + 1, loss, module.model)
            if monitor.wait and monitor.wait % max(1, monitor.patience // 3) == 0:
                for optimizer in trainer.optimizers:
                    for group in optimizer.param_groups:
                        group['lr'] = max(group['lr'] * .5, 1e-7)
            print(f"Validation | epoch={trainer.current_epoch+1} | MSE={loss:.6g} | best={monitor.best:.6g} | wait={monitor.wait}/{monitor.patience}", flush=True)
            module.train()

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
    monitor.restore(model.model)
    if monitor.enabled:
        model.full_data_tensor.copy_(torch.as_tensor(np.where(original_available, (original_matrix - mean) / sd, 0), dtype=torch.float32, device=device))
        model.full_mask_tensor.copy_(torch.as_tensor(original_available, dtype=torch.float32, device=device))
        model.supervision_mask_tensor.copy_(torch.as_tensor(original_available, dtype=torch.float32, device=device))
    prediction = model.get_imputed_data().detach().cpu().numpy() * sd + mean
    prediction[original_available] = original_matrix[original_available]
    if folder is not None and monitor.restored:
        from pathlib import Path
        torch.save({'state_dict':model.model.state_dict(), 'best_epoch':monitor.best_epoch}, Path(folder) / 'best_model.pt')
    finished = time.perf_counter()
    metadata = {
        "device": device,
        "training_device": str(recorder.device),
        "training_protocol": ("fixed model-supplied mask; supervision on remaining observations" if training_hidden.any()
                              else "method-owned training on common evaluation-corrupted input; no platform training mask"),
        "normalization": "per-feature mean/std fitted on model-training observations; validation excluded" if monitor.enabled else "per-feature mean/std fitted on method input observations",
        "supervision_entries_per_epoch": int(available.sum()),
        "training_input_entries": int((available & ~training_hidden).sum()),
        "training_hidden_entries": int(training_hidden.sum()),
        **monitor.metadata(),
        "input_observations": int(original_available.sum()),
        "mask_loss_scope": "all input mask entries; no missing intensity targets",
        "mask_supervision_entries_per_epoch": int(available.size),
        "mask_positive_entries": int((available & ~training_hidden).sum()),
        "mask_negative_entries": int((~available | training_hidden).sum()),
    }
    actual_device = next(model.parameters()).device
    if actual_device.type != device:
        raise RuntimeError(f"requested {device}, model trained on {actual_device}")
    metadata.update(actual_device=str(actual_device), torch_version=torch.__version__, training_engine="lightning",
                    precision="float32", epochs_trained=trainer.current_epoch, training_steps=trainer.global_step,
                    initialization_s=initialized-started, training_s=trained-initialized, prediction_s=finished-trained)
    metadata["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    if name == "jingwei-dcae":
        metadata.update(ae_architecture=params.get("ae_architecture", "global"),
                        model_architecture_version=getattr(model.model, "architecture_version", "native-dcae-v1"),
                        ae_dropout=params.get("ae_dropout", .1) if getattr(model.model, "mask_aware", False) else 0.,
                        mask_head_input="latent" if getattr(model.model, "mask_aware", False) else "reconstruction")
    return prediction, metadata
