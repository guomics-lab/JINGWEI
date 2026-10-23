# jingwei 1.0.8

**尚未正式发布。** 新程序本体和 66 项回归用例已通过验证；完整离线 EXE 已生成，但当前电脑的 Smart App Control 以错误 4551 拦截其启动，完整安装验证尚未完成。安装器语言逻辑已通过中英文真实编译器小型安装测试。正式下载保持 1.0.7，等待可信代码签名或按现有安全策略批准安装包。

**Unreleased.** Frozen application checks and 66 regression cases passed. The complete offline EXE has been built, but this machine's Smart App Control blocks execution with error 4551, so full installation validation remains incomplete. English/Chinese installer fixtures passed. The stable download remains 1.0.7 pending trusted code signing or approval through the existing security policy.

下载完整 Windows 离线 EXE 后运行，点击 **浏览** 选择安装位置。安装时选择 English，首次打开软件即为英文；选择中文安装，首次打开即为中文。重新安装或升级也会应用此次选择的语言。之后手动切换的语言会被记住。

插补完成后，点击 **打开输出目录** 可直接查看 `results/imputed/<方法名>.csv`。H5AD 输入还会生成对应的 `.h5ad`。原始样本、蛋白 ID 和已观测值保留，旧的逐方法输出仍可用。插补界面已删除关于准确性指标不适用的提示。

Run the complete Windows offline EXE and use **Browse** to choose the installation folder. The application initially uses the language selected in Setup, including after an upgrade/reinstallation. Later manual language choices are remembered.

After an Impute task, **Open output folder** opens `results/imputed/`, containing one `<method>.csv` per successful method. H5AD inputs also produce `<method>.h5ad`. Sample/protein identifiers and observed values are preserved. Detailed per-method outputs remain available. The unnecessary imputation notice has been removed.

Python, PyTorch CPU/CUDA, R and all 35 method dependencies are bundled. Installation requires no adjacent BIN files or payload downloads. This is a GUI/export maintenance release: model implementations, masking rules and the 1.0.7 numerical reference are unchanged. The published Python package is unchanged. The existing Linux binary remains available in the 1.0.7 release.
