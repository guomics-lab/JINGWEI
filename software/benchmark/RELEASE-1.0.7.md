# jingwei 1.0.7

Windows 运行 `jingwei-Windows-Setup.exe`，在同一向导中接受纯文本许可、浏览或修改安装位置、下载并安装。Python、PyTorch、R 及各方法依赖均已打包，无需自行安装。离线安装须将 setup.exe 和两个 .bin 文件放在同一文件夹。Linux 下载四个 .tar.part 文件及 `install-linux.sh`，运行脚本安装。

Run `jingwei-Windows-Setup.exe` for one installation wizard with an editable destination and Browse. Python, PyTorch, R and method dependencies are bundled. The offline Windows setup.exe needs its two adjacent .bin files. Linux installation uses all four archive parts and `install-linux.sh`.

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
