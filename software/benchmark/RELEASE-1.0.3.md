# JINGWEI Benchmark V1.0.3

## 中文

1. [下载 Windows 安装程序](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe)，运行后自动下载、校验并安装完整软件。
2. 从开始菜单打开 **JINGWEI Benchmark**，选择数据、方法、评测方案、输出目录和设备，点击 **开始**。GUI 固定单次运行，已移除重复次数和新任务中的重复比较控件。
3. **原 notebook 回放**完整恢复旧 benchmark 的 CSV 预处理、`get_mask()`、单次掩蔽、模型／传统方法和评分流程。**公平评测**保留评测、训练两次掩蔽及 AnnData 支持。两个方案明确区分。

默认数据仍为 Alzheimer.csv。预览不加载 Torch；切换评测方案复用环境检查。中英文、日志、自选输出目录、手动打开任务、表单和环境缓存均保留。

**结果核对**：阿尔兹海默数据的形状、观测数量、掩蔽数量和空列数量与旧记录一致。9 个旧 MSE 中 6 个与原显示的三位小数一致；MF、DAE、DCAE 仍有差异。神经网络还与本机直接执行的旧 notebook 函数交叉核验。历史分数尚未全部重现，不应宣称逐位一致。详情见[回放说明](NOTEBOOK_REPLAY.md)和[测试报告](SE_TESTS.md)。

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

Alzheimer masking counts match the archived notebook. Six of nine archived MSE
values match its printed precision; MF, DAE and DCAE still differ. Neural outputs
are also compared with direct execution of the old functions on this machine.
Not every archived score is reproduced. See the [protocol](NOTEBOOK_REPLAY.md) and
[verification report](SE_TESTS.md). There is no macOS binary in this release and
the published JINGWEI Python package is unchanged.
