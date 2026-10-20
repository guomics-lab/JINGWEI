"""Real bundled R under the Unicode/custom directory used by installation QA."""
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.skipif(os.name != 'nt' or not os.environ.get('JINGWEI_UNICODE_INSTALL'),
                    reason='opt-in installed Windows R relocation check')
def test_bundled_r_loads_and_imputes_from_unicode_install_directory(tmp_path, monkeypatch):
    import jingwei_benchmark.methods as methods
    internal = Path(os.environ['JINGWEI_UNICODE_INSTALL']) / '_internal'
    assert any(ord(c) > 127 for c in str(internal))
    monkeypatch.setattr(methods, 'resource_root', lambda: internal)
    monkeypatch.setattr(methods.sys, 'frozen', True, raising=False)
    environment = methods.r_environment()
    assert environment['LC_ALL'] == 'English_United States.utf8'
    states = methods.availability(names=['bpca','qrilc','mice-norm','missforest'])
    assert all(state['available'] for state in states.values()), states
    values = np.random.default_rng(8).normal(4,.3,(16,8))
    values[::3,0] = np.nan
    folder = tmp_path / '中文 输出';folder.mkdir()
    prediction, _ = methods.run_method('bpca',values,np.zeros_like(values,bool),
        {'rank':2,'threads':1},device='cpu',seed=7,folder=folder,
        rscript=methods.find_rscript())
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction[np.isfinite(values)],values[np.isfinite(values)])
