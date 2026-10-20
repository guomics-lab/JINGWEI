"""Lean PyTorch execution of the native JINGWEI networks.

The published package is unchanged. The software uses bilinear DMF and global
DCAE with fixed model-supplied denoising masks, known reconstruction targets and
AdamW, avoiding per-row GPU collation and Lightning metric bookkeeping.
"""

import random
import hashlib
import time
import types

import numpy as np


def normalize(matrix):
    available = np.isfinite(matrix)
    counts = available.sum(axis=0)
    mean = np.divide(np.where(available, matrix, 0).sum(axis=0), counts,
                     out=np.zeros(matrix.shape[1]), where=counts > 0)
    variance = np.divide(np.where(available, (matrix - mean) ** 2, 0).sum(axis=0), counts,
                         out=np.ones(matrix.shape[1]), where=counts > 0)
    sd = np.sqrt(variance)
    sd[sd < 1e-8] = 1.0
    normalized = np.where(available, (matrix - mean) / sd, 0).astype("float32")
    return normalized, available, mean, sd


def masked_loss(reconstruction, mask_logits, target, input_mask, supervision, weight):
    """Reconstruct known targets and classify the complete known input mask.

    Missing intensities never enter reconstruction loss. Their mask labels are
    known zeros, however, and belong in the original full-matrix BCE objective.
    Restricting BCE to reconstruction supervision loses every negative label in
    a single-mask benchmark.
    """
    import torch.nn.functional as F
    count = supervision.sum().clamp_min(1)
    reconstruction_loss = ((reconstruction - target).square() * supervision).sum() / count
    if weight > 0:
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, input_mask)
    else:
        mask_loss = mask_logits.sum() * 0.0
    return reconstruction_loss + weight * mask_loss


def _factorized_rows(self, row_indices):
    """Linear([row, col]) = W_row row + W_col col + bias.

    Keep every original parameter and all following activations/dropout. Unlike
    concatenating embeddings for every sample/protein pair, the first projection
    is computed once per row and once per protein. FP32 reduction order differs.
    """
    import torch.nn.functional as F
    first = self.reconstruction_mlp[0]
    split = self.embedding_dim
    rows = F.linear(self.row_embedding(row_indices), first.weight[:, :split], first.bias)
    columns = F.linear(self.col_embedding.weight, first.weight[:, split:])
    hidden = (rows[:, None, :] + columns[None, :, :]).reshape(-1, first.out_features)
    for layer in self.reconstruction_mlp[1:]:
        hidden = layer(hidden)
    return hidden.reshape(len(row_indices), self.n_cols)


def factorize_dmf(model):
    import torch.nn as nn
    layers = getattr(model, "reconstruction_mlp", None)
    dimension = getattr(model, "embedding_dim", None)
    if (not isinstance(layers, nn.Sequential) or not len(layers)
            or not isinstance(layers[0], nn.Linear)
            or dimension is None or layers[0].in_features != 2 * dimension
            or not callable(getattr(model, "_reconstruct_rows", None))):
        return False
    model._reconstruct_rows = types.MethodType(_factorized_rows, model)
    return True


def run_jingwei_torch(name, matrix, training_hidden, params, *, device, seed, folder=None):
    started = time.perf_counter()
    import torch
    from torch.utils.data import DataLoader
    from ._native.models import DMF, DCAE

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from .convergence import validation_split, ValidationMonitor, DEFAULT_MAX_EPOCHS
    original_available = np.isfinite(matrix)
    validation_hidden = validation_split(matrix, training_hidden, params, seed, folder)
    fit_matrix = np.where(validation_hidden, np.nan, matrix)
    normalized, available, mean, sd = normalize(fit_matrix)
    monitor = ValidationMonitor(params, enabled=validation_hidden.any(), validation_mask=validation_hidden)
    if name == "jingwei-dmf":
        from .dmf_models import build_dmf
        model = build_dmf(params.get('dmf_architecture','bilinear'),n_rows=len(matrix), n_cols=matrix.shape[1],
                    embedding_dim=params.get("latent_dim", 64),
                    interaction_dim=params.get('dmf_rank'),
                    hidden_dims=params.get("hidden_dims", [256, 128]),
                    mask_predictor_hidden_dim=(params.get("hidden_dims", [256, 128]) or [128])[0])
        factorized = factorize_dmf(model)
    else:
        from .ae_models import build_dcae
        model = build_dcae(params.get("ae_architecture", "global"), dropout=params.get("ae_dropout", .1),
                     input_feature_dim=matrix.shape[1], ae_dim=params.get("latent_dim", 64),
                     mask_predictor_hidden_dim=params.get("mask_predictor_hidden_dim", params.get("hidden_dims", [256])[0]),
                     num_encoder_blocks=params.get("encoder_blocks", 2),
                     num_decoder_blocks=params.get("decoder_blocks", 2),
                     dilation=params.get("dilation", 2))
        factorized = False
        print(f"AE architecture={params.get('ae_architecture', 'global')} | parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    if name == 'jingwei-dmf' and callable(getattr(model,'initialize_from_observed',None)):
        if params.get('dmf_initialization','random')=='svd':
            visible=torch.as_tensor(available & ~training_hidden)
            model.initialize_from_observed(torch.from_numpy(normalized)*visible,visible)
    model.to(device)
    target = torch.from_numpy(normalized).to(device)
    input_mask = torch.as_tensor(available & ~training_hidden, dtype=torch.float32, device=device)
    supervision = torch.as_tensor(available, dtype=torch.float32, device=device)
    inputs = target * input_mask
    validation_mask = torch.as_tensor(validation_hidden, dtype=torch.bool, device=device)
    validation_truth = torch.as_tensor(np.where(validation_hidden, matrix, 0), dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(sd, dtype=torch.float32, device=device)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    # Shuffle row indices on CPU and gather each batch in one device operation.
    loader = DataLoader(torch.arange(len(matrix)), batch_size=params.get("batch_size", 64),
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get("learning_rate", 0.001), weight_decay=1e-5)
    weight = params.get("mask_weight", 0.5)
    epochs = params.get("max_epochs", DEFAULT_MAX_EPOCHS)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    synchronize()
    initialized = time.perf_counter()
    model.train()
    steps = 0
    log_interval = max(1, epochs // 20)
    epochs_trained = 0

    def reconstruct(batch, mask, indices):
        if name == "jingwei-dmf":
            return model._reconstruct_rows(indices)
        if getattr(model, "mask_aware", False):
            return model.reconstruct(batch, mask)
        return model.ae(batch.unsqueeze(1)).squeeze(1)

    for epoch in range(epochs):
        model.train()
        total_loss = torch.zeros((), device=device)
        batches = 0
        for indices in loader:
            indices = indices.to(device)
            batch_input = inputs.index_select(0, indices)
            if name == "jingwei-dmf":
                reconstruction, logits = model(batch_input, row_indices=indices)
            elif getattr(model, "mask_aware", False):
                reconstruction, logits = model(batch_input, input_mask.index_select(0, indices))
            else:
                reconstruction, logits = model(batch_input)
            loss = masked_loss(reconstruction, logits, target.index_select(0, indices),
                               input_mask.index_select(0, indices), supervision.index_select(0, indices), weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            batches += 1
            steps += 1
        epochs_trained = epoch + 1
        if monitor.enabled:
            model.eval()
            with torch.no_grad():
                squared_error = torch.zeros((), device=device)
                for begin in range(0, len(matrix), params.get('batch_size', 64)):
                    end = min(begin + params.get('batch_size', 64), len(matrix))
                    prediction = reconstruct(inputs[begin:end], input_mask[begin:end], torch.arange(begin, end, device=device))
                    difference = prediction * scale_tensor + mean_tensor - validation_truth[begin:end]
                    squared_error += difference[validation_mask[begin:end]].square().sum()
                validation_loss = float(squared_error / validation_mask.sum())
            stop = monitor.observe(epoch + 1, validation_loss, model)
            # Give optimization a chance to improve at smaller steps before
            # declaring a validation plateau. This never consults test scores.
            if monitor.early_stopping and monitor.wait and monitor.wait % max(1, monitor.patience // 3) == 0:
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * .5, 1e-7)
            if epoch == 0 or (epoch + 1) % 10 == 0 or stop:
                print(f"Validation | epoch={epoch+1} | MSE={validation_loss:.6g} | best={monitor.best:.6g} | wait={monitor.wait}/{monitor.patience}", flush=True)
            if stop:
                print(f"Validation plateau | restoring epoch {monitor.best_epoch}", flush=True)
                break
        if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch+1}/{epochs} | loss={float(total_loss / max(1, batches)):.6g} | steps={steps}", flush=True)
    synchronize()
    trained = time.perf_counter()
    if folder is not None and monitor.policy in ('fixed', 'fixed-validation'):
        from pathlib import Path
        torch.save({'state_dict':model.state_dict(), 'epoch':epochs_trained}, Path(folder) / 'last_model.pt')
    monitor.restore(model)
    model.eval()
    # Validation observations are available input at final inference. The
    # normalization statistics and fitted weights never used their targets.
    inference = torch.as_tensor(np.where(original_available, (matrix - mean) / sd, 0), dtype=torch.float32, device=device)
    inference_mask = torch.as_tensor(original_available, dtype=torch.float32, device=device)
    parts = []
    with torch.no_grad():
        for begin in range(0, len(matrix), params.get("batch_size", 64)):
            end = min(begin + params.get("batch_size", 64), len(matrix))
            if name == "jingwei-dmf":
                output = model._reconstruct_rows(torch.arange(begin, end, device=device))
            elif getattr(model, "mask_aware", False):
                output = model.reconstruct(inference[begin:end], inference_mask[begin:end])
            else:
                # Native DCAE inference does not need its auxiliary mask head.
                output = model.ae(inference[begin:end].unsqueeze(1)).squeeze(1)
            parts.append(output.cpu().numpy())
    synchronize()
    predicted = np.concatenate(parts) * sd + mean
    predicted[original_available] = matrix[original_available]
    if folder is not None and monitor.restored:
        from pathlib import Path
        torch.save({'state_dict':model.state_dict(), 'best_epoch':monitor.best_epoch}, Path(folder) / 'best_model.pt')
    finished = time.perf_counter()
    actual = next(model.parameters()).device
    if actual.type != device:
        raise RuntimeError(f"requested {device}, model trained on {actual}")
    return predicted, {
        "device": device, "actual_device": str(actual), "training_device": str(actual),
        "torch_version": torch.__version__, "training_engine": "torch",
        "training_protocol": ("fixed model-supplied mask; supervision on remaining observations" if training_hidden.any()
                              else "method-owned training on common evaluation-corrupted input; no platform training mask"),
        "normalization": "per-feature mean/std fitted on model-training observations; validation excluded" if monitor.enabled else "per-feature mean/std fitted on method input observations",
        "dmf_factorized_first_linear": factorized,
        "precision": "float32", "epochs_trained": epochs_trained, "training_steps": steps,
        **monitor.metadata(),
        "input_observations": int(original_available.sum()),
        "supervision_entries_per_epoch": int(available.sum()),
        "training_input_entries": int((available & ~training_hidden).sum()),
        "training_hidden_entries": int(training_hidden.sum()),
        "training_mask_sha256": hashlib.sha256(training_hidden.tobytes()).hexdigest(),
        "training_mask_policy": "fixed for entire fit",
        "mask_loss_scope": "all input mask entries; no missing intensity targets",
        "mask_supervision_entries_per_epoch": int(available.size),
        "mask_positive_entries": int((available & ~training_hidden).sum()),
        "mask_negative_entries": int((~available | training_hidden).sum()),
        "initialization_s": initialized - started, "training_s": trained - initialized,
        "prediction_s": finished - trained,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        **({'model_architecture_version':getattr(model,'architecture_version','legacy-dmf-v1'),
            'dmf_architecture':params.get('dmf_architecture','bilinear'),
            'dmf_interaction_rank':getattr(model,'interaction_dim',None),
            'dmf_initialization':getattr(model,'initialization',{'method':'legacy random embeddings'}),
            'mask_head_input':'row latent' if hasattr(model,'row_tower') else 'reconstruction'}
           if name=='jingwei-dmf' else {}),
        **({"ae_architecture": params.get("ae_architecture", "global"),
            "model_architecture_version": getattr(model, "architecture_version", "native-dcae-v1"),
            "ae_dropout": params.get("ae_dropout", .1) if getattr(model, "mask_aware", False) else 0.,
            "mask_head_input": "latent" if getattr(model, "mask_aware", False) else "reconstruction"}
           if name == "jingwei-dcae" else {}),
    }
