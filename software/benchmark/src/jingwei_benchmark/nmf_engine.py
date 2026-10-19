"""Memory-layout compatibility for the original sklearn NMF adapter."""

from contextlib import contextmanager
import numpy as np


@contextmanager
def contiguous_nmf(module):
    original = module.NMF

    class ContiguousNMF(original):
        def _check_w_h(self, *args, **kwargs):
            w, h = super()._check_w_h(*args, **kwargs)
            return np.ascontiguousarray(w), np.ascontiguousarray(h)

    module.NMF = ContiguousNMF
    try:
        yield
    finally:
        module.NMF = original


def run_nmf(full, mask):
    from ._legacy import nmf
    with contiguous_nmf(nmf):
        return nmf.impute(full, mask)
