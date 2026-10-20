"""Lean PyTorch execution of the native JINGWEI networks.

The published package is unchanged. This adapter keeps its architectures,
normalization, two-mask objective and AdamW settings while avoiding per-row GPU
DataLoader collation and Lightning metric bookkeeping in isolated benchmarks.
"""

import random
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
    """The native masked means, expressed without GPU nonzero/index syncs.

    A zero denominator contributes zero loss, matching the native empty-batch
    branches. Dense zero weights never admit E or native missing values.
    """
    import torch.nn.functional as F
    count = supervision.sum().clamp_min(1)
    reconstruction_loss = ((reconstruction - target).square() * supervision).sum() / count
    if weight > 0:
        mask_loss = (F.binary_cross_entropy_with_logits(mask_logits, input_mask, reduction="none") * supervision).sum() / count
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


def run_jingwei_torch(name, matrix, training_hidden, params, *, device, seed):
    started = time.perf_counter()
    import torch
    from torch.utils.data import DataLoader
    from ._native.models import DMF, DCAE

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    normalized, available, mean, sd = normalize(matrix)
    if name == "jingwei-dmf":
        model = DMF(n_rows=len(matrix), n_cols=matrix.shape[1],
                    embedding_dim=params.get("latent_dim", 64),
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
    model.to(device)
    target = torch.from_numpy(normalized).to(device)
    input_mask = torch.as_tensor(available & ~training_hidden, dtype=torch.float32, device=device)
    supervision = torch.as_tensor(available, dtype=torch.float32, device=device)
    inputs = target * input_mask
    # Shuffle row indices on CPU and gather each batch in one device operation.
    loader = DataLoader(torch.arange(len(matrix)), batch_size=params.get("batch_size", 64),
                        shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get("learning_rate", 0.001), weight_decay=1e-5)
    weight = params.get("mask_weight", 0.5)
    epochs = params.get("max_epochs", 50)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    synchronize()
    initialized = time.perf_counter()
    model.train()
    steps = 0
    log_interval = max(1, epochs // 20)
    for epoch in range(epochs):
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
        if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch+1}/{epochs} | loss={float(total_loss / max(1, batches)):.6g} | steps={steps}", flush=True)
    synchronize()
    trained = time.perf_counter()
    model.eval()
    parts = []
    with torch.no_grad():
        for begin in range(0, len(matrix), params.get("batch_size", 64)):
            end = min(begin + params.get("batch_size", 64), len(matrix))
            if name == "jingwei-dmf":
                output = model._reconstruct_rows(torch.arange(begin, end, device=device))
            elif getattr(model, "mask_aware", False):
                output = model.reconstruct(target[begin:end] * supervision[begin:end], supervision[begin:end])
            else:
                # Native DCAE inference does not need its auxiliary mask head.
                output = model.ae((target[begin:end] * supervision[begin:end]).unsqueeze(1)).squeeze(1)
            parts.append(output.cpu().numpy())
    synchronize()
    predicted = np.concatenate(parts) * sd + mean
    predicted[available] = matrix[available]
    finished = time.perf_counter()
    actual = next(model.parameters()).device
    if actual.type != device:
        raise RuntimeError(f"requested {device}, model trained on {actual}")
    return predicted, {
        "device": device, "actual_device": str(actual), "training_device": str(actual),
        "torch_version": torch.__version__, "training_engine": "torch",
        "training_protocol": ("fixed model-supplied mask; supervision on remaining observations" if training_hidden.any()
                              else "method-owned training on common evaluation-corrupted input; no platform training mask"),
        "normalization": "per-feature mean/std fitted after evaluation masking",
        "dmf_factorized_first_linear": factorized,
        "precision": "float32", "epochs_trained": epochs, "training_steps": steps,
        "supervision_entries_per_epoch": int(available.sum()),
        "training_input_entries": int((available & ~training_hidden).sum()),
        "initialization_s": initialized - started, "training_s": trained - initialized,
        "prediction_s": finished - trained,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        **({"ae_architecture": params.get("ae_architecture", "global"),
            "model_architecture_version": getattr(model, "architecture_version", "native-dcae-v1"),
            "ae_dropout": params.get("ae_dropout", .1) if getattr(model, "mask_aware", False) else 0.,
            "mask_head_input": "latent" if getattr(model, "mask_aware", False) else "reconstruction"}
           if name == "jingwei-dcae" else {}),
    }
