# JINGWEI Benchmark V1.0.3

这是旧版本的安装说明。当前 AD 论文结果采用 v1.0.5 GUI 导出的任务；旧 notebook 数值记录已移除，见[当前结果核验](SE_TESTS.md)。

This is a historical installation guide. Current AD manuscript results use an exported v1.0.5 GUI task. Obsolete notebook result records have been removed; see the [current verification report](SE_TESTS.md).

## 中文

1. [下载 Windows 安装程序](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe)，运行后自动下载、校验并安装完整软件。
2. 从开始菜单打开 **JINGWEI Benchmark**，选择数据、方法、评测方案、输出目录和设备，点击 **开始**。GUI 固定单次运行，已移除重复次数和新任务中的重复比较控件。
3. **原 notebook 回放**完整恢复旧 benchmark 的 CSV 预处理、`get_mask()`、单次掩蔽、模型／传统方法和评分流程。**公平评测**保留评测、训练两次掩蔽及 AnnData 支持。两个方案明确区分。

默认数据仍为 Alzheimer.csv。预览不加载 Torch；切换评测方案复用环境检查。中英文、日志、自选输出目录、手动打开任务、表单和环境缓存均保留。

**结果记录**：旧 notebook 的 AD 分数不再作为论文依据。当前 GUI 任务、主表数值与耗时的核验见[测试报告](SE_TESTS.md)。

Linux：下载全部 `jingwei-benchmark-1.0.3-linux-x86_64.tar.part*`、对应 `SHA256SUMS` 和 `install-linux.sh` 到同一文件夹，再运行 `sh install-linux.sh`。已有旧版命令时，选择新安装位置和命令目录，例如 `sh install-linux.sh "$HOME/.local/share/JINGWEI/Benchmark-1.0.3" "$HOME/.local/bin/jingwei-1.0.3"`。

无需安装 Python、R、conda 或方法依赖。Windows 离线安装可下载带版本号的 EXE 和全部同名前缀 BIN。当前发布不含 macOS 二进制；已发布的 JINGWEI Python 包不变。

## English

1. [Download the Windows installer](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe). It downloads, verifies and installs the full application.
2. Open **Start → JINGWEI Benchmark**, choose data, methods, protocol, output folder and device, then **Run**. GUI tasks run once; repeat controls are removed.
3. **Original notebook replay** restores historical preprocessing, `get_mask()`, the single mask, original model/baseline adapters and scoring. **Fair benchmark** retains sequential evaluation/training masks and AnnData support.

The bundled Alzheimer dataset remains the default. Preview avoids importing
Torch; protocol switching reuses the environment probe. Chinese/English, live
logs, custom output folders, manual task opening and persistent preferences
remain available. Dependencies and runtimes are bundled. Offline Windows setup
uses the versioned EXE and all matching BIN files. Linux installation uses all
versioned TAR parts, checksums and `install-linux.sh`, with a fresh destination
and command directory when an older command exists.

Archived notebook AD scores are no longer the manuscript reference. See the
[verification report](SE_TESTS.md) for the current GUI task, main-table values
and elapsed times. There is no macOS binary in this release and
the published JINGWEI Python package is unchanged.
