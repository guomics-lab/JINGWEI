# JINGWEI Benchmark V1.0.4

## 中文

- DCAE 本身更新为显式输入掩码、全局瓶颈与全连接解码器；不增加数据集专用方法。
- 新 GUI 默认使用当前评测方案和更新后的模型；旧版表单首次升级时切换，旧 notebook 回放仍可手动选择作历史对照。
- GUI 不显示重复次数，新任务固定一次评测。启动不自动打开上一次任务。
- 支持实时日志、自定义输出目录、手动打开任务、中英文切换与 JINGWEI 任务栏图标。
- 本地离线安装包是一个 EXE，包含 Python、PyTorch、R 和方法运行环境，无需单独安装依赖。

Alzheimer 更新结果及其他数据集尚未重评的范围见 AE_ARCHITECTURE.md。软件版本更新不修改已发布的 Python 包。

## English

- DCAE itself uses the updated mask-aware encoder, global bottleneck and dense decoder. No dataset-specific method is added.
- New GUI jobs default to current models; older saved forms migrate once. Historical notebook replay remains an explicit reference option.
- New GUI jobs run once. Repetition controls are removed and previous task results are not reopened automatically.
- Live logs, custom output folders, manual task opening, Chinese/English and the JINGWEI taskbar icon are included.
- The local offline installer is one EXE bundling Python, PyTorch, R and method runtimes.

See AE_ARCHITECTURE.md for the Alzheimer evaluation and unchanged results on other datasets. The published Python package is not republished by this application release.
