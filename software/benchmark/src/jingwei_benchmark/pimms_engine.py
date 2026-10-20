"""PIMMS models with cached loaders and an explicit training stopping policy.

Fixed-budget execution retains native one-cycle fitting. Convergence execution
retains the native model, loss and LR finder, then uses validation-plateau
fitting. Patches are local to the isolated worker and always restored.
"""

from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
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


@contextmanager
def convergent_fitting(monitor):
    """Keep native PIMMS models/loss/LR finder, then fit to a validation plateau.

    A fixed-epoch one-cycle schedule ties its warmup to the arbitrary epoch cap.
    In convergence mode use its LR-finder suggestion with plateau LR reduction,
    best-state restoration and an explicit safety cap instead.
    """
    if not monitor.enabled:
        yield
        return
    from fastai.learner import Learner
    import fastai.callback.schedule  # installs Learner.fit_one_cycle
    from fastai.callback.core import Callback, CancelFitException
    from fastai.callback.tracker import EarlyStoppingCallback

    class ConvergenceCallback(Callback):
        order = 65

        def after_epoch(self):
            index = list(self.learn.recorder.metric_names[1:]).index('valid_loss')
            loss = float(self.learn.recorder.values[-1][index])
            stop = monitor.observe(self.epoch + 1, loss, self.learn.model)
            if monitor.wait and monitor.wait % max(1, monitor.patience // 3) == 0:
                for hyper in self.learn.opt.hypers:
                    hyper['lr'] = max(hyper['lr'] * .5, 1e-7)
            print(f"Validation | epoch={self.epoch+1} | native loss={loss:.6g} | best={monitor.best:.6g} | wait={monitor.wait}/{monitor.patience}", flush=True)
            if stop:
                raise CancelFitException()

        def after_fit(self):
            monitor.restore(self.learn.model)

    original = Learner.fit_one_cycle
    def fit_to_plateau(learn, n_epoch, lr_max=None, **kwargs):
        learn.remove_cb(EarlyStoppingCallback)
        return learn.fit(n_epoch, lr=lr_max, cbs=[ConvergenceCallback()])
    Learner.fit_one_cycle = fit_to_plateau
    try:
        yield
    finally:
        Learner.fit_one_cycle = original


def run_pimms(name, matrix, training_hidden, params, *, device, seed, folder):
    folder = Path(folder)
    from .convergence import validation_split, ValidationMonitor
    selection = validation_split(matrix, training_hidden, params, seed, folder)
    monitor = ValidationMonitor(params, enabled=selection.any(), validation_mask=selection,
                                metric='native PIMMS validation loss')
    corruption = training_hidden | selection
    if name in ("pimms-dae", "pimms-vae"):
        # The upstream scaler/imputer cannot fit a feature whose entire
        # training column is missing. Do not silently change the shared mask,
        # borrow validation/test values or let sklearn drop model dimensions.
        empty = ~(np.isfinite(matrix) & ~corruption).any(axis=0)
        if empty.any():
            from .methods import MethodInputError
            raise MethodInputError(
                f"{name} requires an observed training value in every feature; "
                f"{int(empty.sum())} features have no model-training observation. "
                "The common input is preserved."
            )
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
    validation_mask = selection if monitor.enabled else training_hidden
    train, validation = frame.mask(corruption), frame.where(validation_mask)
    options = dict(epochs_max=params.get("max_epochs", 50), cuda=device == "cuda")
    engine = params.get("baseline_engine", "optimized")
    with cached_loaders(engine == "optimized", points=matrix.size, method=name) as acceleration, epoch_logging(engine == "optimized"), convergent_fitting(monitor):
        if name == "pimms-cf":
            from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer

            training_series = train.stack().rename("intensity")
            validation_series = validation.stack().rename("intensity")
            model = CollaborativeFilteringTransformer(
                target_column="intensity", sample_column="sample", item_column="feature",
                n_factors=params.get("latent_dim", 30), batch_size=min(params.get("batch_size", 4096), len(training_series)),
                out_folder=str(folder))
            started = time.perf_counter()
            model.fit(training_series, y=validation_series if len(validation_series) else None, **options)
            fitted = time.perf_counter()
            learner = model.learn
            training_steps = len(learner.recorder.losses)
            # Native get_preds squeezes a single missing prediction to a scalar
            # in this fastai version. Keep native categorical encoding and
            # forward computation, preserving a vector for every batch size.
            missing = frame.isna().stack()
            indices = missing.loc[missing].index
            prediction = matrix.copy()
            if len(indices):
                test_loader = model.dls.test_dl(indices.to_frame(), shuffle=False, drop_last=False)
                learner.model.eval()
                parts = []
                with torch.no_grad():
                    for batch in test_loader:
                        parts.append(learner.model(batch[0]).reshape(-1).detach().cpu().numpy())
                prediction[~np.isfinite(matrix)] = np.concatenate(parts)
        else:
            from pimmslearn.sklearn.ae_transformer import AETransformer

            model = AETransformer(model=name.split("-")[1].upper(),
                hidden_layers=params.get("hidden_dims", [512]), latent_dim=params.get("latent_dim", 50),
                batch_size=params.get("batch_size", 64), out_folder=str(folder))
            started = time.perf_counter()
            model.fit(train, y=validation if validation_mask.any() else None, **options)
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
        "training_protocol": ("model-supplied mask used as PIMMS validation; test truths never supplied" if training_hidden.any()
                              else "native PIMMS training/corruption on common evaluation-corrupted input; no platform training mask"),
        "pimms_acceleration": acceleration, "fit_s": fitted-started,
        "progress_logging": "epoch" if engine == "optimized" else "native",
        "prediction_s": finished-fitted, "epochs_trained": model.epochs_trained_,
        "training_steps": training_steps,
        "training_observations": int(train.notna().to_numpy().sum()),
        **monitor.metadata(),
        "validation_observations": int(validation_mask.sum()),
        "learning_rate_schedule": "validation-plateau" if monitor.enabled else "native one-cycle",
        "batch_size": params.get("batch_size", 64),
        "effective_batch_size": int(learner.dls.train.bs),
    }
    if monitor.restored:
        torch.save({'state_dict':learner.model.state_dict(), 'best_epoch':monitor.best_epoch},
                   folder / 'best_model.pt')
    # Native plots remain available; close them after fitting to release memory.
    from matplotlib import pyplot as plt
    plt.close("all")
    return prediction, metadata
