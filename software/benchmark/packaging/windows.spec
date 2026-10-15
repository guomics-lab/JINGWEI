"""Native desktop build. Windows full builds include the local R runtime/library."""

import importlib.metadata
import os
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

application = Path(SPECPATH).parent
root = Path(SPECPATH).parents[2]
sys.path.insert(0, str(application / "src"))
from jingwei_benchmark.methods import r_environment, find_rscript

if sys.platform != "win32":
    raise RuntimeError("Windows GUI builds must run on Windows; POSIX uses build_posix.py")

datas = collect_data_files("jingwei_benchmark") + collect_data_files("pimmslearn")
datas.append((str(root / "data/Alzheimer.csv"), "data"))
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if name:
        try:
            datas += copy_metadata(name)
        except Exception:
            pass
# Only original adapter scripts are needed at runtime, not entire Git checkouts.
sources = root / ".method-sources"
for name, relative in [("NAguideR", "inst/NAguideRapp/Trunc_KNN/Imput_funcs.r"),
                       ("DreamAI", "Code/R/RegImpute.R")]:
    path = sources / name / relative
    if path.is_file():
        datas.append((str(path), str(Path(".method-sources") / name / Path(relative).parent)))

if sys.platform == "win32":
    rscript = find_rscript()
    runtime = root / ".r-runtime"
    library = Path(r_environment().get("R_LIBS_USER", root / ".r-library"))
    if not rscript or not runtime.is_dir() or not (library / "jsonlite").is_dir():
        raise RuntimeError("Full Windows build requires the local .r-runtime and installed R library")
    datas += [(str(runtime), ".r-runtime"), (str(library), ".r-library")]

a = Analysis(
    [str(application / "src/jingwei_benchmark/desktop_entry.py")], pathex=[str(application / "src")],
    binaries=[], datas=datas,
    hiddenimports=collect_submodules("jingwei_benchmark") + collect_submodules("pimmslearn") +
                  ["matplotlib.backends.backend_pdf", "matplotlib.backends.backend_svg"],
    hookspath=[], hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg", "pdf", "svg"]}},
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "tensorflow", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="JINGWEI Benchmark",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False,
          icon=str(root / "figures/Jingwei_logo.png"),
          version=str(application / "packaging/windows-version.txt"))
console = EXE(pyz, a.scripts, [], exclude_binaries=True, name="jingwei-benchmark",
              debug=False, strip=False, upx=False, console=True,
              icon=str(root / "figures/Jingwei_logo.png"),
          version=str(application / "packaging/windows-version.txt"))
collection = COLLECT(exe, console, a.binaries, a.datas, strip=False, upx=False, name="JINGWEI Benchmark")
