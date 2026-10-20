# jingwei 1.0.7

Windows 下载完整离线安装包 [`jingwei-1.0.7-windows-x86_64-offline.exe`](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.7/jingwei-1.0.7-windows-x86_64-offline.exe)，直接运行，在向导中接受纯文本许可，点击 **浏览** 选择安装位置后安装。可在浏览器下载时选择 EXE 的保存位置。全部 Python、PyTorch CPU/CUDA、R 及方法依赖均在这个 EXE 中，安装不再联网下载，也不需要旁边的 .bin 文件或 C 盘的大容量下载缓存。Linux 安装方式不变。

Download and run `jingwei-1.0.7-windows-x86_64-offline.exe`. Accept the plain-text license, then use **Browse** to choose the installation folder. Your browser can save the EXE on any selected drive. The one EXE includes Python, PyTorch CPU/CUDA, R and every method dependency. Installation is offline, needs no adjacent BINs and does not stage a large payload in the C-drive download cache. Linux installation still uses all four archive parts and `install-linux.sh`.

可选在线安装器 `jingwei-1.0.7-windows-x86_64-online.exe` 的暂存目录也已改到用户选择的安装盘，成功后清理。请使用上方完整离线安装包作为默认入口。此次仅调整 Windows 安装方式，软件计算实现和 1.0.7 的结果不变。

The optional `jingwei-1.0.7-windows-x86_64-online.exe` stages downloads on the selected installation volume and removes them after success. Prefer the complete offline EXE above. This update changes Windows packaging only; the 1.0.7 computation and results are unchanged.

## 训练 / Training

五个神经网络默认各训练 **500 个 epoch**，不提前停止，不额外划内部验证集，使用最后一轮权重。500 轮不代表已证明收敛。传统方法保留各自迭代设置。

评测先构造一次所有方法共用的测试掩蔽。DMF/DCAE 再从剩余观测值中固定抽取一次去噪训练掩蔽，机制和比例与运行设置一致。第二次掩蔽仅遮住训练输入，已知值仍是重构目标；不接触评测真值，也不传给其他方法。训练期间不重抽掩蔽。

All five neural models default to **500 epochs**, with no early stopping or extra internal validation partition. Predictions use the final epoch weights, saved as `last_model.pt`; the budget is not a convergence claim. Traditional methods retain their own iterations.

Benchmark makes one common test mask. DMF/DCAE get one additional fixed denoising mask drawn only from the remaining observations, using the selected mechanism/rate. These known values remain supervised reconstruction targets. Test truths never reach methods, the training mask is never redrawn, and other methods receive the common input without this extra mask.

## DMF

DMF 改为显式双线性交互、行列偏置和残差神经网络；交互秩默认 8，不超过潜在维度。掩码预测头共享行潜变量，避免直接要求重构强度编码 0/1 掩码。保留联合重构与掩码损失，未设置 Alzheimer 专用模型。DCAE 结构保持不变。

DMF now uses signed bilinear interactions, row/column biases and residual neural towers. Its interaction rank defaults to 8, capped by latent dimension. The mask head shares row latent factors rather than reconstructed intensities. The joint objective is retained, with no dataset-specific dispatch. DCAE architecture is unchanged.

新版协议不同于旧版验证早停实验。完整参数、掩码、轮数、权重和指标保存在结果目录。单个数据集、单次掩蔽的数值不代表普遍优势。论文及已发布的 Python 包未修改。Windows 支持中文/English；macOS 本次不提供未验证的二进制。

This protocol differs from earlier validation-stopping experiments. Result folders retain masks, parameters, epochs, checkpoints and metrics. One dataset/mask does not establish universal superiority. The manuscript and separately published Python package are unchanged. Windows supports Chinese/English; unvalidated macOS binaries are not included.
