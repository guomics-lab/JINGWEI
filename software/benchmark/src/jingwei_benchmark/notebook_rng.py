"""Carry the notebook's random stream across isolated method workers."""

from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np


@contextmanager
def notebook_random_stream(legacy, seed, previous, folder, *, neural):
    arrays = {"torch_cpu": legacy["rng_state"]}
    state = None
    if previous:
        previous = Path(previous)
        with np.load(previous / 'notebook_rng.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        state = json.loads((previous / 'notebook_rng.json').read_text(encoding='utf-8'))
    if neural:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            for index in range(torch.cuda.device_count()):
                key = 'torch_cuda_' + str(index)
                if key in arrays:
                    torch.cuda.set_rng_state(torch.as_tensor(arrays[key], dtype=torch.uint8), index)
        torch.set_rng_state(torch.as_tensor(arrays['torch_cpu'], dtype=torch.uint8))
    # Imports and lazy CUDA initialization must finish before restoring the
    # stream. Otherwise each new process adds random draws absent in a notebook.
    random.seed(seed)
    np.random.seed(seed)
    if state is not None:
        p = state['python']
        random.setstate((p[0], tuple(p[1]), p[2]))
        n = state['numpy']
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    try:
        yield
    finally:
        if neural:
            arrays['torch_cpu'] = torch.get_rng_state().numpy()
            if torch.cuda.is_available():
                for index, state in enumerate(torch.cuda.get_rng_state_all()):
                    arrays['torch_cuda_' + str(index)] = state.cpu().numpy()
        folder = Path(folder)
        np.savez(folder / 'notebook_rng.npz', **arrays)
        n = np.random.get_state()
        state = {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]]}
        (folder / 'notebook_rng.json').write_text(json.dumps(state), encoding='utf-8')
