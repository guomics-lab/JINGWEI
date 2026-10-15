import importlib.util
import os
from pathlib import Path
import subprocess
import pytest

spec = importlib.util.spec_from_file_location("benchmark_fix_r", Path(__file__).resolve().parents[1] / "packaging/fix_r_runtime.py")
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)


@pytest.mark.skipif(os.name == "nt", reason="POSIX R shell wrapper")
def test_relocated_r_wrapper_handles_spaces_quotes_and_is_idempotent(tmp_path):
    prefix = tmp_path / "runtime with spaces and 'quote"
    wrapper = prefix / "lib/R/bin/R"
    wrapper.parent.mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "bin/sed").symlink_to("/bin/sed")
    wrapper.write_text("#!/bin/sh\nR_HOME_DIR=/old/prefix/lib/R\nSED=/old/prefix/bin/sed\nprintf '%s\\n' \"$R_HOME_DIR\" \"$SED\" | ${SED} -e 's/old/old/'\n")
    wrapper.chmod(0o755)
    fix.repair(prefix)
    result = subprocess.run(["/bin/sh", str(wrapper)], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(prefix / "lib/R"), str(prefix / "bin/sed")]
    timestamp = wrapper.stat().st_mtime_ns
    fix.repair(prefix)
    assert wrapper.stat().st_mtime_ns == timestamp


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink isolation")
def test_repair_rejects_wrapper_outside_runtime(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("R_HOME_DIR=/old\n")
    prefix = tmp_path / "runtime"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/R").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        fix.repair(prefix)
    assert outside.read_text() == "R_HOME_DIR=/old\n"
