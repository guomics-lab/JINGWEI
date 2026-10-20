# jingwei 1.0.6

## MNAR 与训练修正 / MNAR and training correction

GUI 的 **MNAR · JINGWEI** 使用新版丰度依赖机制。对 log2 工作矩阵的每个蛋白，
用已观测值计算 z 分数，再令 `p = sigmoid(-z + b)`。截距 b 校准到用户选择的
目标掩蔽比例；低丰度值具有更高的缺失概率。实际抽样比例写入日志和任务记录。
设置中的比例是原始观测值中被人工掩蔽的比例，不含原有缺失值。

评测只构造一次共同测试掩蔽，各方法接收相同缺失矩阵。训练归一化只使用
剩余观测值。DMF/DCAE 的重构损失仅使用可用真值；掩蔽预测 BCE 恢复为整个
输入掩码，包含已知的 0/1 标签，不读取缺失位置的丰度真值。

The GUI's **MNAR · JINGWEI** uses `p = sigmoid(-z + b)`, with z calculated
within each protein on the working scale. A fitted intercept matches the selected
expected mask rate. Lower intensities have higher missingness probability. Actual
draw rates are logged. The rate applies to initially observed entries.

Benchmark creates one common test mask. All methods receive the same missing
matrix; training normalization only uses available observations. Reconstruction
loss uses known intensity targets only. DMF/DCAE mask BCE covers the complete
known input mask, including both observed and missing labels.

CLI: `--mechanism mnar-jingwei-v2 --mask-rate 0.2`. Old mechanism IDs and saved
tasks retain their historical meaning. The historical mixed-scale rule remains
available as **Legacy rule (old tasks)**; it is not the corrected MNAR option.
An older saved GUI form selecting that historical rule migrates to v2. This
protocol change requires new evaluations; earlier numerical results are not
claims about this version.

## 按验证集收敛训练 / Validation-based stopping

五个神经网络默认按验证集表现训练，不统一截断为 50 epoch。各模型内部从
共同输入剩余的观测值中划出 10% 用作验证；验证值不进入训练损失或归一化
统计量，最终测试真值始终不传给模型。默认至少训练 30 轮，验证指标连续
30 轮无显著相对改善（阈值 1e-4）时停止，并恢复最佳权重。

DMF/DCAE 监控工作尺度的验证 MSE；PIMMS 监控各自原生验证损失，保留其
架构、损失及学习率查找，使用平台期学习率衰减。CF 使用其原生默认的
4,096 条目批次，样本型网络默认每批 64 个样本。各方法参数可独立调整。

5,000 轮和每方法 30 分钟是安全上限，触及上限不代表收敛。结果记录
收敛状态、实际轮数、最佳轮数、验证掩码和权重。非迭代方法无需训练收敛；
尚无诊断的迭代或随机传统方法标记为“未评估”，不会仅因执行结束而标记收敛。

All five neural models default to model-owned validation stopping and restore
their best checkpoint. Validation is drawn only from available input entries,
with no benchmark test targets. Defaults: validation fraction 0.1, minimum 30
epochs, patience 30, relative tolerance 1e-4. JINGWEI monitors validation MSE;
PIMMS monitors its native validation objective with plateau LR reduction.
Its native LR finder, model and loss are retained. The 5,000-epoch/30-minute
caps do not certify convergence. `--training-policy fixed --max-epochs N`
remains available for diagnostic smoke runs. Traditional methods without a
convergence diagnostic are explicitly marked not assessed.

## Windows 安装 / Installation

运行 `jingwei-Windows-Setup.exe`，在同一个向导中接受许可、修改安装位置或点击
“浏览”、下载并完成安装。下载后不会再弹出第二个安装向导。离线安装同样始终
提供目录选择。软件名称为 jingwei，已包含 Python、PyTorch、R 和方法依赖。

Run `jingwei-Windows-Setup.exe`. One wizard handles the plain-text license,
editable destination with Browse, download, installation and completion. The
offline installer also always offers destination selection, including upgrades.
Python, PyTorch, R and method dependencies are bundled.

Windows 的内置 R 使用 UTF-8 语言环境，支持包含中文和空格的自定义安装路径。
Bundled Windows R uses a UTF-8 locale so custom installation paths can contain
Chinese characters and spaces.
