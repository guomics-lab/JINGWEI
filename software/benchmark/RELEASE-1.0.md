# JINGWEI Benchmark V1.0

## Windows x86-64

Download `JINGWEI-Benchmark-1.0.0-windows-x86_64-setup.exe` and **all** matching
`setup-*.bin` files into the same folder. Run the EXE and choose English or
简体中文. The installer includes all Python/R runtimes and methods, creates a
Start menu entry, and supports uninstall through Windows Settings. No conda,
Python or R installation is needed. Installation is per user.

启动安装程序后选择中文或英文。请将安装 EXE 和全部同名 BIN 分卷放在同一文件夹；
无需安装 Python、R 或 conda。安装后从开始菜单打开 JINGWEI Benchmark。
默认加载 Alzheimer 示例数据，GUI 默认选择 DMF/DCAE；CLI 默认评测全部 34 种方法。
卸载软件会保留用户任务和结果。

The console is installed beside the GUI. From PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\JINGWEI Benchmark\jingwei-benchmark.exe" --version
```

Unsigned builds may be rejected by an organization's App Control policy. On the
build machine the previous native GUI smoke passed, while the console was
blocked before startup (WinError 4551); this is not a passed console test. No
security policy changes or bypasses are part of installation.

## Linux x86-64

Download **all** `jingwei-benchmark-1.0.0-linux-x86_64.tar.partNNN` files,
`jingwei-benchmark-1.0.0-linux-x86_64.SHA256SUMS` and `install-linux.sh` into one
folder. In that folder run:

```bash
sh install-linux.sh
~/.local/bin/jingwei-benchmark --version
~/.local/bin/jingwei-benchmark methods
~/.local/bin/jingwei-benchmark benchmark --methods mice --device cpu --output results/mice
```

The installer verifies every part and the combined archive, installs under
`~/.local/share/JINGWEI/Benchmark-1.0.0`, and adds a command under
`~/.local/bin`. Optional positional arguments select the installation and
command directories, including paths with spaces:

```bash
sh install-linux.sh "$HOME/apps/JINGWEI Benchmark" "$HOME/bin"
```

Linux 命令行安装同样无需 Python、R 或 conda。首次启动会离线解压运行环境，
需要为安装文件和运行缓存预留磁盘空间。默认评测 34 种方法、重复 5 次；
每种方法每次重复默认超时 30 分钟。不支持某个输入的方法会明确记录失败原因。

The Linux build is tested in Ubuntu under WSL with glibc 2.43. Compatibility
with older Linux distributions is not established. CUDA requires a compatible
NVIDIA driver; CPU execution is also supported. macOS is CLI only, but **no
macOS binary is included in V1.0** because no Mac builder was available.

## Evaluation

V1.0 includes 34 methods, CSV/h5ad import, shared evaluation masks plus a second
training mask, MAE/RMSE/NRMSE/correlations and multivariate sample W2. W2 remains
undefined when no valid complete reference population exists. Python linear
MICE-PMM is the default (`mice`), with five imputations and five sweeps; the
original R PMM is available with `--mice-backend r`.

Checksums are provided in `SHA256SUMS`. For Windows verification:

```powershell
Get-FileHash .\JINGWEI-Benchmark-1.0.0-windows-x86_64-setup* -Algorithm SHA256
```

The published JINGWEI Python package is a separate product and is not released
by this V1.0 application release.
