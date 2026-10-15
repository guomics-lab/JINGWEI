# Benchmark release engineering (maintainers only)

These files build the standalone **benchmark application**. They are not a
dependency-installation workflow for end users. The published JINGWEI Python
package and its release process are outside this application's scope.

Release builders copy `software/benchmark/LICENSE`, matching the application's
declared PolyForm Noncommercial 1.0.0 license. Its text is reproduced unchanged
from the [official 1.0.0 source](https://github.com/polyformproject/polyform-licenses/blob/1.0.0/PolyForm-Noncommercial-1.0.0.md).

Use the dedicated `jingwei_benchmark` conda environment. Windows and WSL have
separate environments; never share compiled Python/R libraries across systems.
To create a new builder, use the corresponding profile under `environments/`:
`benchmark.yml` on Windows, `benchmark-posix-r.yml` on Linux, or
`benchmark-macos.yml` on macOS. `condarc` records the selected channels.
The profiles and `requirements-build.txt` are private build inputs, not files to
distribute as benchmark installation requirements.

The native network adapters are fixed within `jingwei_benchmark/_native`;
release builds do not import the public package or an editable root checkout.
`prepare_build.py` installs the other build
dependencies, pinned NAguideR/DreamAI scripts and R backends into that builder;
it does not build, publish or change JINGWEI's Python package.

```bash
conda activate jingwei_benchmark
python software/benchmark/packaging/prepare_build.py --torch-build cpu
```

Windows/Linux CUDA builders can select `--torch-build cu128`. macOS is CPU only;
Apple Silicon uses torch 2.7.1 and Intel Mac uses the legacy torch 2.2.2 profile.
Windows requires native R; use `--rscript PATH` and optionally `--r-library PATH`
to configure it. Full Windows releases use this checkout's `.r-runtime` and
`.r-library`. POSIX releases require all R packages inside the conda prefix.
The private conda-pack tool requires setuptools < 81; keep its conda profile and
Python build constraints aligned instead of overwriting conda-managed files.

## Windows GUI and CLI

```powershell
python software/benchmark/packaging/build_windows.py
python software/benchmark/packaging/validate_windows.py `
  --executable "software/benchmark/dist/windows-x86_64/JINGWEI Benchmark/JINGWEI Benchmark.exe" `
  --output results/benchmark-windows-validation --all-methods --cuda
```

Output contains `JINGWEI Benchmark.exe`, `jingwei-benchmark.exe`, and their shared
`_internal` runtime. Both use the same application source and worker whitelist.
All Python/R backends are bundled. PyInstaller's Windows onedir build avoids
repeated multi-GB extraction by method workers. The original logo is retained.
The GUI includes Chinese and English in the same executable. Validation uses
the Windows Qt platform, switches languages before and during a real job, and
checks the process AppUserModelID plus both native window icon handles. UI
screenshots in both languages are saved with the validation output.

Then wrap the complete directory using Inno Setup 6.7.3 (official compiler):

```powershell
python software/benchmark/packaging/build_installer.py --iscc "C:\path\to\ISCC.exe"
```

The per-user installer supports English/Chinese, Start menu/optional desktop
shortcuts, the original icon and Windows uninstall. It leaves user jobs outside
the installation directory. EXE and all matching BIN slices must be distributed
together. Slices stay below GitHub's 2 GiB asset limit. The Chinese translation
in `languages/` is copied unchanged from jrsoftware/issrc tag `is-6_7_3`,
`Files/Languages/Unofficial/ChineseSimplified.isl`, retaining its credits.

## Linux/macOS CLI

```bash
python software/benchmark/packaging/build_posix.py
python software/benchmark/packaging/validate_posix.py \
  --executable software/benchmark/dist/linux-x86_64/jingwei-benchmark \
  --output results/benchmark-linux-validation --all-methods --cuda
```

Build on the target operating system and architecture. Linux/macOS releases
contain an offline runtime archive packed from the complete conda build
environment, including Python, PyTorch, R, R packages and their shared libraries.
Editable paths are excluded and replaced by actual application/dependency files.
Qt/GUI modules and unrelated web-server packages are excluded. The launcher uses
only the bundled Python/R, clears host override variables and runs the offline
`conda-unpack` path repair at a stable user-cache location. The archive checksum
selects that cache; moving the application folder cannot invalidate R paths.
After relocation, `fix_r_runtime.py` quotes R wrapper paths and sed commands so
cache directories containing spaces or quotes work without altering host R.
The outer release is a `.tar` containing the already compressed runtime archive.

`prepare_release.py` splits the Linux TAR into release assets, writes per-part
checksums and generates `install-linux.sh`. The installer verifies both the
parts and the complete archive before extraction; its command symlink is
resolved by the launcher. It rejects existing destinations and commands.

No host R fallback is part of the launcher. A missing bundled backend fails
validation. Release validation runs outside the repository with host conda/R and
Python paths removed. It checks all method availability, real CPU benchmark
outputs, optional actual CUDA devices, and absence of Qt. This verifies execution
and output integrity, not convergence or comparative scientific performance.

Linux compatibility still depends on the kernel/glibc and GPU driver supported
by the bundled binaries. `build-info.json` records the builder's libc, platform
and dependency versions. Validate on each supported target distribution. macOS
has separate arm64/x86_64 builds, CLI only; do not claim a Mac artifact is tested
without building and running it on a Mac. Signing/notarization and public upload
are separate release steps.

## Tests

```bash
python -m pytest -c software/benchmark/pyproject.toml software/benchmark/tests -q
```

This targets the application tests without modifying the published package's
test configuration. Tests cover mask/metric parity, input handling, W2, device
policy, cancellation, durable history and exports. Native GUI tests run on
Windows only. Existing user job storage and schema remain unchanged.
