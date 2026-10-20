"""Model-owned validation and an auditable validation-plateau stopping rule.

Only the common, already test-corrupted input enters this module. It never has
access to the benchmark's held-out intensity values or its scoring arrays.
"""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MAX_EPOCHS = 5000
DEFAULT_PATIENCE = 30
DEFAULT_MIN_EPOCHS = 30
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_VALIDATION_FRACTION = .1


def validation_split(matrix, training_hidden, params, seed, folder=None):
    observed = np.isfinite(matrix)
    hidden = np.zeros_like(observed)
    if params.get('training_policy', 'fixed') not in ('fixed','convergence'):
        raise ValueError('unknown neural training policy')
    if params.get('training_policy', 'fixed') == 'convergence':
        from .masking import sample_hidden_mask
        from .protocol import _retain_marginals
        fraction = params.get('validation_fraction', DEFAULT_VALIDATION_FRACTION)
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('validation fraction must be between 0 and 1')
        rng = np.random.default_rng(np.random.SeedSequence([seed, 12648430]))
        eligible = observed & ~training_hidden
        hidden = sample_hidden_mask(eligible, fraction, rng)
        hidden, _ = _retain_marginals(eligible, hidden, rng)
        if not hidden.any():
            from .methods import MethodInputError
            raise MethodInputError('insufficient available observations for model-owned validation; convergence cannot be assessed')
        if folder is not None:
            np.savez_compressed(Path(folder) / 'model_validation.npz', hidden=hidden)
    return hidden


class ValidationMonitor:
    def __init__(self, params, *, enabled, validation_mask, metric='validation MSE'):
        self.enabled = bool(enabled)
        self.policy = params.get('training_policy', 'fixed')
        self.patience = params.get('patience', DEFAULT_PATIENCE)
        self.min_epochs = params.get('min_epochs', DEFAULT_MIN_EPOCHS)
        self.min_delta = params.get('min_delta', DEFAULT_MIN_DELTA)
        if (type(self.patience) is not int or self.patience < 1
                or type(self.min_epochs) is not int or self.min_epochs < 1
                or not np.isfinite(self.min_delta) or not 0 <= self.min_delta < 1):
            raise ValueError('invalid convergence patience, minimum epochs or relative tolerance')
        self.metric = metric
        self.validation_count = int(validation_mask.sum())
        self.validation_hash = hashlib.sha256(validation_mask.tobytes()).hexdigest()
        self.best = float('inf')
        self.significant_best = float('inf')
        self.best_epoch = None
        self.wait = 0
        self.state = None
        self.history = []
        self.stopped = False
        self.restored = False

    def observe(self, epoch, loss, model):
        if not np.isfinite(loss):
            raise ValueError('nonfinite validation loss; convergence cannot be assessed')
        self.history.append({'epoch':int(epoch), 'validation_loss':float(loss)})
        if loss < self.best:
            self.best, self.best_epoch = float(loss), int(epoch)
            self.state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if not np.isfinite(self.significant_best) or loss < self.significant_best - self.min_delta * max(abs(self.significant_best), 1e-12):
            self.significant_best, self.wait = float(loss), 0
        else:
            self.wait += 1
        self.stopped = epoch >= self.min_epochs and self.wait >= self.patience
        return self.stopped

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)
            self.restored = True

    def metadata(self):
        status = ('validation_plateau' if self.stopped else 'max_epochs') if self.enabled else (
            'insufficient_validation' if self.policy == 'convergence' else 'fixed_budget')
        return {'training_policy':self.policy, 'converged':self.stopped,
                'convergence_status':status, 'best_epoch':self.best_epoch,
                'best_validation_loss':self.best if np.isfinite(self.best) else None,
                'validation_metric':self.metric if self.enabled else None,
                'validation_observations':self.validation_count,
                'validation_mask_sha256':self.validation_hash,
                'best_checkpoint_restored':self.restored,
                'convergence_patience':self.patience, 'convergence_min_epochs':self.min_epochs,
                'convergence_relative_min_delta':self.min_delta,
                'validation_history':self.history}
