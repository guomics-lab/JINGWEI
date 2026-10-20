"""A local release must remain buildable after retiring old result guides."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def test_prepare_linux_assets_with_current_guides(tmp_path, monkeypatch):
    path=Path(__file__).resolve().parents[1]/'packaging/prepare_release.py'
    spec=importlib.util.spec_from_file_location('prepare_test_release',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    archive=tmp_path/'runtime.tar';archive.write_bytes(b'local packaging fixture')
    output=tmp_path/'release'
    monkeypatch.setattr(sys,'argv',['prepare_release','--archive',str(archive),'--output',str(output)])
    module.main()
    parts=list(output.glob('*.part*'))
    assert len(parts)==1 and parts[0].read_bytes()==archive.read_bytes()
    checksums=next(output.glob('*.SHA256SUMS')).read_text()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() in checksums
    assert (output/'SE_TESTS.md').is_file() and (output/'LICENSE.txt').is_file()
    assert list(output.glob('INSTALL-*.md')) and (output/'install-linux.sh').is_file()
    assert not (output/'NOTEBOOK_REPLAY.md').exists()
