"""Official PIMMS fitting with cached, numerically identical loader inputs.

The fastai learner, batching/shuffling, LR finder and all model callbacks stay
upstream. Patches are local to an isolated worker and restored even on errors.
"""

from contextlib import contextmanager
from importlib.metadata import version
import random
import time

import numpy as np
import pandas as pd


def _cached_dataset(base, *, validation=False):
    import torch

    class CachedDataset(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Match pimmslearn.io.datasets.to_tensor exactly, including masks'
            # floating dtype. The native constructor still fits/transforms data.
            tensors = [self.mask_isna, self.data]
            if validation:
                tensors.append(self.target)
            self._tensor_rows = tuple(torch.from_numpy(df.to_numpy()).type(torch.get_default_dtype()) for df in tensors)

        def __getitem__(self, idx):
            parts = tuple(tensor[idx] for tensor in self._tensor_rows)
            return parts if validation else (*parts, parts[1])

    return CachedDataset


def _cached_collab_loader(base):
    from fastai.tabular.core import ReadTabBatch

    class CachedCollabLoader(base):
        def __init__(self, dataset, *args, after_batch=None, **kwargs):
            # ReadTabBatch performs the exact native conversion once, after
            # categorification. Keep native indices/shuffle/drop_last/device.
            self._tensor_columns = ReadTabBatch(dataset)(dataset)
            super().__init__(dataset, *args, after_batch=[], **kwargs)

        def create_batch(self, indices):
            return tuple(column[indices] for column in self._tensor_columns)

    return CachedCollabLoader


@contextmanager
def epoch_logging(enabled):
    """Keep native epoch logs/recorder, omit per-batch console bars and syncs."""
    if not enabled:
        yield
        return
    from fastai.callback.progress import ProgressCallback
    from fastai.torch_core import defaults

    original = defaults.callbacks
    defaults.callbacks = [cb for cb in original if cb is not ProgressCallback]
    try:
        yield
    finally:
        defaults.callbacks = original


@contextmanager
def cached_loaders(enabled, *, points, method):
    detail = {"applied": False, "reason": "native execution"}
    if not enabled:
        yield detail
        return
    # The integration is audited against these upstream implementations. A new
    # dependency version uses native execution until its loaders are verified.
    if version("pimms-learn") != "0.5.2" or version("fastai") != "2.8.12":
        detail["reason"] = "unsupported PIMMS/fastai version; native loaders"
        yield detail
        return
    if points * 24 > 256 * 1024**2:
        detail["reason"] = "tensor cache exceeds 256 MiB; native loaders"
        yield detail
        return
    if method == "pimms-cf":
        from fastai.collab import TabularCollab
        original = TabularCollab._dl_type
        local_attribute = "_dl_type" in TabularCollab.__dict__
        TabularCollab._dl_type = _cached_collab_loader(original)
        try:
            detail.update(applied=True, reason="cached native categorical/target tensors")
            yield detail
        finally:
            if local_attribute:
                TabularCollab._dl_type = original
            else:
                del TabularCollab._dl_type
    else:
        from pimmslearn.io import datasets
        original = datasets.DatasetWithTarget, datasets.DatasetWithTargetSpecifyTarget
        datasets.DatasetWithTarget = _cached_dataset(original[0])
        datasets.DatasetWithTargetSpecifyTarget = _cached_dataset(original[1], validation=True)
        try:
            detail.update(applied=True, reason="cached native row tensors")
            yield detail
        finally:
            datasets.DatasetWithTarget, datasets.DatasetWithTargetSpecifyTarget = original


def run_pimms(name, matrix, training_hidden, params, *, device, seed, folder):
    import torch
    from fastai.torch_core import defaults

    # Fastai loaders own Python RNGs seeded from random, in addition to torch and
    # numpy. Set all three in this isolated process for repeatable comparisons.
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    defaults.cpus = 0
    frame = pd.DataFrame(matrix, index=[f"S{i}" for i in range(len(matrix))],
                         columns=[f"F{i}" for i in range(matrix.shape[1])])
    frame.index.name, frame.columns.name = "sample", "feature"
    train, validation = frame.mask(training_hidden), frame.where(training_hidden)
    options = dict(epochs_max=params.get("max_epochs", 50), cuda=device == "cuda")
    engine = params.get("baseline_engine", "optimized")
    with cached_loaders(engine == "optimized", points=matrix.size, method=name) as acceleration, epoch_logging(engine == "optimized"):
        if name == "pimms-cf":
            from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer

            training_series = train.stack().rename("intensity")
            validation_series = validation.stack().rename("intensity")
            model = CollaborativeFilteringTransformer(
                target_column="intensity", sample_column="sample", item_column="feature",
                n_factors=params.get("latent_dim", 30), batch_size=params.get("batch_size", 64),
                out_folder=str(folder))
            started = time.perf_counter()
            model.fit(training_series, y=validation_series if len(validation_series) else None, **options)
            fitted = time.perf_counter()
            learner = model.learn
            training_steps = len(learner.recorder.losses)
            prediction = model.transform(frame.stack().rename("intensity")).unstack().reindex(index=frame.index, columns=frame.columns).to_numpy()
        else:
            from pimmslearn.sklearn.ae_transformer import AETransformer

            model = AETransformer(model=name.split("-")[1].upper(),
                hidden_layers=params.get("hidden_dims", [512]), latent_dim=params.get("latent_dim", 50),
                batch_size=params.get("batch_size", 64), out_folder=str(folder))
            started = time.perf_counter()
            model.fit(train, y=validation if training_hidden.any() else None, **options)
            fitted = time.perf_counter()
            learner = model.analysis.learn
            training_steps = len(learner.recorder.losses)
            prediction = np.asarray(model.transform(frame), dtype=float)
        if device == "cuda":
            torch.cuda.synchronize()
        finished = time.perf_counter()
    actual_device = next(learner.model.parameters()).device
    if actual_device.type != device:
        raise RuntimeError(f"requested {device}, model trained on {actual_device}")
    metadata = {
        "device": device, "actual_device": str(actual_device), "torch_version": torch.__version__,
        "training_protocol": "fixed second mask supplied as PIMMS internal validation; native model training/corruption; test truths never supplied",
        "pimms_acceleration": acceleration, "fit_s": fitted-started,
        "progress_logging": "epoch" if engine == "optimized" else "native",
        "prediction_s": finished-fitted, "epochs_trained": model.epochs_trained_,
        "training_steps": training_steps,
        "training_observations": int(train.notna().to_numpy().sum()),
        "validation_observations": int(training_hidden.sum()),
        "batch_size": params.get("batch_size", 64),
    }
    # Native plots remain available; close them after fitting to release memory.
    from matplotlib import pyplot as plt
    plt.close("all")
    return prediction, metadata
