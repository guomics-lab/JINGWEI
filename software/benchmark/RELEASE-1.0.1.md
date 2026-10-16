# JINGWEI Benchmark V1.0.1 — Windows

## 中文

1. [下载 Windows 安装程序（EXE）](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.1/JINGWEI-Benchmark-Windows-Setup.exe)。只需下载这一个文件。
2. 双击运行，选择中文或英文。安装程序会自动下载、校验完整软件并完成安装，无需手动寻找 BIN 分卷，也无需安装 Python、R 或 conda。首次安装需要联网，软件下载量约 2.66 GB。
3. 安装一次后，从 **开始菜单 → JINGWEI Benchmark** 打开软件。以后无需重复运行安装 EXE。

软件会记住上次的数据文件、方法、参数、语言及窗口布局。首次检查计算环境后，后续打开会复用检查结果；软件升级、后端路径、依赖或显卡驱动变化时会重新检查。也可在 **计算环境 → 重新检查** 中手动更新。

升级安装会保留任务与结果。对 V1.0 产生的历史任务，新版会在首次打开时恢复最近任务的设置；原来的任务记录不变。若原数据文件已移动或删除，软件会提示重新选择。

若取消安装，已经校验完整的下载文件可供下次重试使用。若校验失败，安装程序会停止，不执行不完整的软件。

<details>
<summary>需要离线安装？</summary>

从本次发布页下载 `JINGWEI-Benchmark-1.0.1-windows-x86_64-setup.exe` 和全部同名前缀的 `setup-*.bin`，放在同一文件夹后运行 EXE。它们与推荐的安装程序安装相同的软件。

</details>

Linux CLI 此次无需更新，继续使用 [V1.0 离线包](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.0)。本次不包含 macOS 安装包，也不重新发布 JINGWEI Python 包。

本机已通过桌面升级和重开测试；独立 Windows CLI 被 Smart App Control 拦截（错误 4551），尚未通过本机运行验证。桌面 GUI 正常运行，未修改系统安全设置。

## English

1. [Download the Windows installer (EXE)](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.1/JINGWEI-Benchmark-Windows-Setup.exe). This is the only file you need to download manually.
2. Run it and choose English or Chinese. The installer downloads and verifies the complete software automatically; no manual BIN downloads or Python/R/conda installation are needed. Initial installation requires internet access and downloads approximately 2.66 GB.
3. After installation, open **Start → JINGWEI Benchmark**. You do not need to run the setup EXE again.

The desktop remembers your data file, methods, parameters, language and window layout. It reuses the first environment check on subsequent launches, invalidating it when the application, backends, dependencies or GPU driver change. **Environment → Refresh** forces a new check. Workers still validate their actual runtime before computation.

Upgrades retain jobs and results. On the first upgrade from V1.0, the most recent job supplies the initial settings without modifying its records. Missing or moved input files prompt you to select the file again. Verified completed downloads are reused after cancellation; checksum failures prevent execution.

For an offline installation, download the versioned `windows-x86_64-setup.exe` and every matching BIN from the release, then run that EXE with all files in one folder. The Linux CLI remains at V1.0; this is a Windows usability patch.

Desktop upgrade and reopen tests passed. The standalone Windows console was blocked by Smart App Control on the validation machine (error 4551), so console execution is not validated here. The desktop GUI runs normally; system security settings were left unchanged.
