import os
from pathlib import Path
import pytest
from jingwei_benchmark.atomic import write_json


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing violations')
def test_atomic_update_recovers_from_transient_windows_read_lock(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('{"old":true}')
    replace = Path.replace
    calls = []
    def locked(self, target):
        calls.append(1)
        if len(calls) <= 2:
            assert path.read_text() == '{"old":true}'
            raise PermissionError('simulated sharing violation')
        return replace(self, target)
    monkeypatch.setattr(Path, 'replace', locked)
    write_json(path, {'new': True})
    assert '"new": true' in path.read_text() and len(calls) == 3
    assert not list(tmp_path.glob('*.tmp'))
