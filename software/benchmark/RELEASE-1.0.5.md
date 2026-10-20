# jingwei 1.0.5

Windows application: jingwei.exe. Offline installer: jingwei-1.0.5-windows-x86_64-setup.exe.

## 安装 / Installation

下载并运行 [jingwei-Windows-Setup.exe](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.5/jingwei-Windows-Setup.exe)，选择中文或英文，按提示安装。安装程序自动下载并校验完整软件；无需另装 Python、R 或 conda。安装后从 **开始菜单 → jingwei** 启动。

Download and run [jingwei-Windows-Setup.exe](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.5/jingwei-Windows-Setup.exe), choose Chinese or English, and follow the installer. It downloads and verifies the complete application. Python, R and conda are not required on the user's computer. Open **Start → jingwei** after installation.

离线安装：将同一发布页的 `jingwei-1.0.5-windows-x86_64-setup.exe` 和所有同名前缀的 `.bin` 文件下载到同一个文件夹，然后运行该 EXE。可使用 `SHA256SUMS` 校验文件。

For offline installation, download `jingwei-1.0.5-windows-x86_64-setup.exe` and all matching `.bin` files from the same release into one folder, then run that EXE. `SHA256SUMS` lists the file checksums.

Windows 10/11 x86-64; CPU or NVIDIA CUDA. Linux CLI is included in this release. Download all `jingwei-1.0.5-linux-x86_64.tar.part*` files, the Linux SHA256SUMS file and `install-linux.sh` to one folder, then run `sh install-linux.sh`. The installed command is `jingwei`; Python, R and conda are bundled. This release does not include a macOS binary.

## Changes

Configuration is locked while submitting, running or cancelling a task. Reuse
settings restores the actual saved configuration and method parameters. The
Open task button is removed. Empty task names become `task-YYYYMMDD-HHMMSS`.
The software and current CLI are named **jingwei**. In Impute, Missingness and
Mask rate configure JINGWEI DMF/DCAE training corruption. Hidden observations
remain supervised targets; inference and output preserve the original observations.

Run settings contain only Task, Missingness and Mask rate. Device and model
parameters are in Advanced. Protocol, repeats, training-mask and seed controls
and summaries are removed from the desktop form. Chinese and English remain
available; results start empty.

The platform constructs **one shared test mask** and supplies the same corrupted
dataset to all methods. It adds no second training mask or validation split.
Each model retains its own training, validation and denoising procedure.
Imputation creates a JINGWEI-specific training mask without removing target values
or changing comparison-method inputs. CLI training-mask options have
been removed; old saved result files remain readable. Mask protocol 3 records
the single-mask benchmark contract; training_hidden is all-false for benchmarks.

Original MNAR preserves get_mask's formula, strict probability bands and Torch
column-major random draws. Its rate shows Auto. All-empty columns are removed
from comparison-method inputs and excluded from a common scoring scope, as in
the notebook. Feature IDs and held-out/evaluated/excluded counts are recorded;
excluded features remain NaN in exported matrices. Historical preprocessing and
adapters remain available through the CLI's original-notebook profile.

Environment checks complete before the main window appears, with startup
cancel/retry and valid-cache reuse. The installer presents the plain-text
LICENSE.txt with unchanged terms. Installation ID, taskbar identity and saved
preference locations are retained for upgrades.
