"""Execute original notebook adapters on the same host as the reference.

Historical algorithms are isolated from the fair benchmark. Only masked values
and the original pre-mask normalization statistics reach this process.
"""

import importlib
import time
import types

import numpy as np


NOTEBOOK_ADAPTERS = {
    "knn": "knn", "feature-minimum": "minimum",
    "nmf": "nmf", "mice-pmm": "mice", "missforest": "missforest",
    "pimms-dae": "dae", "pimms-vae": "vae",
}


def run_notebook_method(name, matrix, training_hidden, params, legacy, *, device, seed, fallback, folder, rng_input=None):
    from .notebook_rng import notebook_random_stream
    # A notebook imports Lightning before seed_everything(0). Its first import
    # advances Python's RNG (through dependencies), so do not import it after
    # restoring a preceding method's checkpoint in a fresh worker.
    if name.startswith(("jingwei-", "pimms-")):
        import pytorch_lightning
        if name.startswith("pimms-"):
            from pimmslearn.sklearn.ae_transformer import AETransformer
        else:
            from ._legacy import dmf, dcae
    elif name in NOTEBOOK_ADAPTERS:
        importlib.import_module("._legacy." + NOTEBOOK_ADAPTERS[name], __package__)
    with notebook_random_stream(legacy, seed, rng_input, folder, neural=name.startswith(("jingwei-", "pimms-"))):
        return _compute(name, matrix, params, legacy, device=device, fallback=fallback)


def _compute(name, matrix, params, legacy, *, device, fallback):
    started = time.perf_counter()
    available = np.isfinite(matrix)
    valid = available.any(axis=0)
    metadata = {"evaluation_protocol": "original-notebook", "zero_cols": int((~valid).sum()),
                "normalization": "original float32 statistics fitted before evaluation masking",
                "training_protocol": "original single mask; no second training mask",
                "source_revision": "98686b54fdb9918a4d4cc9d99ac4880a6dadcefe"}
    if name.startswith("jingwei-"):
        prediction, detail = _run_jingwei(name, available, params, legacy, device)
    else:
        if not valid.any():
            raise ValueError("original notebook has no nonempty columns after masking")
        filtered = np.ascontiguousarray(matrix[:, valid], dtype=np.float32)
        mask = np.isfinite(filtered).astype(np.float32)
        full = np.where(mask.astype(bool), filtered, 0).astype(np.float32)
        adapter = NOTEBOOK_ADAPTERS.get(name)
        if adapter in ("dae", "vae"):
            import pandas as pd
            from pimmslearn.sklearn.ae_transformer import AETransformer

            # The notebook used CPU for DAE, CUDA for VAE. A requested CPU or
            # non-CUDA platform remains supported without inventing GPU support.
            actual_device = "cpu" if adapter == "dae" else device
            model = AETransformer(model=adapter.upper(), hidden_layers=[512], latent_dim=50, batch_size=10)
            frame = pd.DataFrame(np.where(mask.astype(bool), full, np.nan))
            from .pimms_engine import epoch_logging
            with epoch_logging(True):
                model.fit(frame, cuda=actual_device == "cuda", epochs_max=params.get("notebook_pimms_epochs", 200))
            output = model.transform(frame).to_numpy()
            detail = {"device": actual_device, "epochs_requested": params.get("notebook_pimms_epochs", 200),
                      "algorithm": "original PIMMS AETransformer"}
        elif adapter:
            module = importlib.import_module("._legacy." + adapter, __package__)
            if adapter == "nmf":
                from .nmf_engine import run_nmf
                output = run_nmf(full, mask)
            elif adapter == "missforest":
                forest_class = module.RandomForestRegressor
                def budgeted_forest(*args, **kwargs):
                    kwargs['n_jobs'] = params.get('threads', 4)
                    return forest_class(*args, **kwargs)
                module.RandomForestRegressor = budgeted_forest
                try:
                    output = module.impute(full, mask)
                finally:
                    module.RandomForestRegressor = forest_class
            else:
                output = module.impute(full, mask)
            detail = {"device": "cpu", "algorithm": "original Python " + adapter}
        else:
            output, detail = fallback(filtered, np.zeros_like(filtered, dtype=bool))
            detail["algorithm"] = "additional method on original notebook mask"
        if np.asarray(output).shape != filtered.shape:
            raise ValueError("original adapter returned a matrix with the wrong shape")
        prediction = np.zeros(matrix.shape, dtype=np.float32)
        prediction[:, valid] = output
    metadata.update(detail)
    metadata["backend_s"] = time.perf_counter() - started
    return prediction, metadata


def _run_jingwei(name, available, params, legacy, device):
    import torch
    import pytorch_lightning as pl
    from ._legacy.dmf import DMFImputer
    from ._legacy.dcae import DCAEImputer

    # The original wrappers store plain CPU tensors and Lightning transfers
    # batches. Keep that behavior, including CPU inference after fit teardown.
    normalized = torch.as_tensor(legacy["normalized"], dtype=torch.float32)
    visible = torch.as_tensor(available, dtype=torch.float32)
    common = dict(full_data_tensor=normalized, full_mask_tensor=visible, batch_size=params["batch_size"])
    if name == "jingwei-dmf":
        model = DMFImputer(**common, embedding_dim=params["latent_dim"], hidden_dims=params["hidden_dims"],
                           reconstruction_weight=1., mask_weight=params["mask_weight"], lr=params["learning_rate"])
    else:
        model = DCAEImputer(**common, ae_dim=params["latent_dim"], mask_predictor_hidden_dim=params["mask_predictor_hidden_dim"],
                            lambda_mask=params["mask_weight"], num_encoder_blocks=params["encoder_blocks"],
                            num_decoder_blocks=params["decoder_blocks"], dilation=params["dilation"], learning_rate=params["learning_rate"])

    # Keep persistent-worker RNG behavior. Replacing it with num_workers=0
    # consumes a new base seed every epoch and changes later model initializers.
    # One CPU worker has the same parent RNG/shuffle behavior as the original 20.
    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.full_data_tensor * self.full_mask_tensor,
                                                 self.full_data_tensor, self.full_mask_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=1, persistent_workers=True)

    model.train_dataloader = types.MethodType(train_dataloader, model)
    class EpochLog(pl.Callback):
        def on_train_start(self, trainer, module):
            self.training_device = str(next(module.parameters()).device)

        def on_train_epoch_end(self, trainer, module):
            epoch = trainer.current_epoch + 1
            if epoch == 1 or epoch % 5 == 0 or epoch == params["max_epochs"]:
                print(f"Original notebook | epoch {epoch}/{params['max_epochs']}", flush=True)

    recorder = EpochLog()
    trainer = pl.Trainer(max_epochs=params["max_epochs"], accelerator=device, devices=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, log_every_n_steps=1, callbacks=[recorder])
    trainer.fit(model)
    model.cpu().eval()
    with torch.no_grad():
        output = model.get_imputed_data().cpu()
        # Preserve the original float32 inverse transform and operation order.
        mean, std = torch.from_numpy(legacy["mean"]), torch.from_numpy(legacy["std"])
        output = output * std + mean
    return output.numpy(), {"device": device, "actual_device": recorder.training_device,
                            "training_device": recorder.training_device, "inference_device": str(next(model.parameters()).device),
                            "loader_workers": 1, "persistent_workers": True,
                            "training_engine": "original-lightning", "precision": "float32",
                            "epochs_trained": trainer.current_epoch, "training_steps": trainer.global_step,
                            "algorithm": "original " + name}
