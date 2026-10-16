# JINGWEI Benchmark V1.0.2

## 中文

1. [下载 Windows 安装程序](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.2/JINGWEI-Benchmark-Windows-Setup.exe)，运行一次，由安装程序下载并校验完整软件。
2. 从 **开始菜单 → JINGWEI Benchmark** 打开。选择输入数据、方法和**输出目录**，点击**开始**。每次运行会在输出目录生成独立的任务文件夹，不覆盖其他任务。
3. **日志**页显示任务进度和方法输出。以后通过**打开任务…**选择任务文件夹查看结果；启动时不再自动载入旧任务，也不再显示任务历史列表。

输入设置、输出目录、方法参数、语言和环境检查缓存仍会保留。打开旧任务只读取结果；点击**复制配置**后才能用相同设置创建新任务。

新增 **MSE** 指标，等于 RMSE 的平方。**高级设置 → 训练 → JINGWEI 参数预设**可选择原始 benchmark 的模型与训练参数。单方法参数优先于预设，预设优先于通用设置。预设保留统一的双重掩蔽和严格评分规则；不代表论文结果的完全复现，也不保证某个方法始终最好。

Linux：下载同次发布的全部 `jingwei-benchmark-1.0.2-linux-x86_64.tar.part*`、对应 `SHA256SUMS` 和 `install-linux.sh`，放在同一文件夹中运行 `sh install-linux.sh`。已有旧版命令时，请选择新位置，例如 `sh install-linux.sh "$HOME/.local/share/JINGWEI/Benchmark-1.0.2" "$HOME/.local/bin/jingwei-1.0.2"`。然后运行该目录中的 `jingwei-benchmark`。首次启动在本机缓存中离线展开运行环境，之后直接复用。

无需安装 Python、R、conda 或方法依赖。本次没有 macOS 二进制，也不更新已发布的 JINGWEI Python 包。

离线 Windows 安装：下载同版 `JINGWEI-Benchmark-1.0.2-windows-x86_64-setup.exe` 和全部同名前缀 BIN，放在同一文件夹后运行 EXE。

## English

1. [Download the Windows installer](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.2/JINGWEI-Benchmark-Windows-Setup.exe). It downloads and verifies the full application automatically.
2. Open **Start → JINGWEI Benchmark**. Select data, methods and an **Output folder**, then click **Run**. Every run creates a separate task folder.
3. **Log** shows task progress and method output. Use **Open task…** to view saved GUI task folders or CLI output folders. Startup leaves the results empty; the history list has been removed.

Form preferences, language and the environment cache persist independently of results. Opening a task is read-only. **Reuse settings** prepares a new task.

**MSE** is now reported alongside RMSE. **Advanced → Training → JINGWEI parameter preset** offers the original repository benchmark's model/training settings. Precedence is method overrides, preset, then common settings. The preset retains shared evaluation/training masks and strict scoring; it does not claim exact reproduction of a paper table or a guaranteed method ranking.

Linux: download every versioned TAR part, its SHA256SUMS and `install-linux.sh` from this release into one folder, then run `sh install-linux.sh`. If an older command exists, select a fresh prefix and command directory as shown in the Chinese example above. The first launch expands the bundled runtime offline into a local cache; subsequent launches reuse it.

No Python/R/conda or method dependencies need to be installed. No macOS binary is included. The published JINGWEI Python package is unchanged. For offline Windows installation, download the versioned setup EXE and all matching BIN volumes together.
