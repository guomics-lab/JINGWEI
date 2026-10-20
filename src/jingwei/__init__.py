"""JINGWEI public API with lazy neural imports."""

from importlib import import_module

__version__ = "0.0.1"
_EXPORTS = {
    "CSVDataset": "datasets",
    "DCAE": "models",
    "GlobalDCAE": "ae_models",
    "DMF": "models",
    "DCAEImputer": "dcae",
    "DMFImputer": "dmf",
    "ImputationTrainer": "train",
    "parse_arguments": "train",
    "train_main": "train",
    "BenchmarkMasks": "masking",
    "make_benchmark_masks": "masking",
    "imputation_metrics": "metrics",
    "wasserstein_distance_2": "metrics",
    "wasserstein_distance_2_1d": "metrics",
}
__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(f".{_EXPORTS[name]}", __name__)
    value = getattr(module, "main" if name == "train_main" else name)
    globals()[name] = value
    return value
