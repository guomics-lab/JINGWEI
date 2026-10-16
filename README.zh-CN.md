<p align="center">
  <a href="./README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <img src="./figures/Jingwei_logo.png" alt="JINGWEI 标志" width="112" />
</p>

<h1 align="center">JINGWEI</h1>

<p align="center">
  <strong>蛋白质组数据缺失值插补框架</strong><br />
  深度矩阵分解（DMF） · 扩张卷积自编码器（DCAE）
</p>

<p align="center">
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#python-api-示例">Python API</a> ·
  <a href="#benchmark-软件">Benchmark 软件</a>
</p>

---

<p align="center">
  <a href="https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.1/JINGWEI-Benchmark-Windows-Setup.exe"><strong>下载 Windows 软件安装包（.exe）</strong></a> ·
  <a href="https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.0">Linux CLI</a>
</p>

只需下载一个 EXE 并运行，其余文件由安装程序自动获取。安装后从 **开始菜单 → JINGWEI Benchmark** 打开软件。

JINGWEI 是一个用于蛋白质组数据缺失值插补的深度学习框架，提供 **DMF（深度矩阵分解）** 和 **DCAE（扩张卷积自编码器）** 两种方法。

## 使用方式

| 你的需求 | 使用产品 | 开始使用 |
| --- | --- | --- |
| 用 **DMF / DCAE** 处理自己的数据 | JINGWEI Python 包 | [快速开始](#快速开始) · [Python API](#python-api-示例) |
| 比较 **34 种插补方法** | 独立的 JINGWEI Benchmark 软件 | [Windows 安装包](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.1/JINGWEI-Benchmark-Windows-Setup.exe) · [使用指南](software/benchmark/README.md) |

## 功能

| 方法 | 训练设置 | 监控与记录 |
| --- | --- | --- |
| DMF 和 DCAE | 可配置网络结构与批次大小 | TensorBoard 日志 |
| CPU / CUDA 计算 | 提前停止与可调超参数 | 训练指标与模型检查点 |

## 安装

使用 DMF / DCAE 的 Python 包：

```bash
conda create -n jingwei --override-channels -c conda-forge python=3.12 pip
conda activate jingwei
python -m pip install jingwei
```

`requirements.txt` 列出了 JINGWEI Python 包的核心依赖。**Benchmark 软件使用独立的运行环境，已打包全部依赖，无需安装 Python、R 或 conda。**

## 使用方法

### 快速开始

```bash
# 使用 DMF 方法
python -m jingwei.train --data-path data/your_dataset.csv

# 使用 DCAE，在第 1 号 GPU 上训练
python -m jingwei.train --data-path data/Alzheimer.csv --method DCAE --device cuda --gpu-id 1

# 自定义参数，并启用提前停止
python -m jingwei.train --data-path data/your_dataset.csv \
    --method DMF \
    --hidden-dims 512 256 128 \
    --embedding-dim 128 \
    --early-stopping \
    --max-epochs 100
```

### Python API 示例

```python
import pytorch_lightning as pl

from jingwei import CSVDataset, DMFImputer, DCAEImputer

dataset = CSVDataset("data/Alzheimer.csv")

# 选择以下两种方法之一
model = DMFImputer(
    full_data_tensor=dataset.data_normalized,
    full_mask_tensor=dataset.mask,
    supervision_mask_tensor=dataset.supervision_mask,
    embedding_dim=64,
    hidden_dims=[256, 128],
    batch_size=1024,
)

# 或使用 DCAE
# model = DCAEImputer(
#     full_data_tensor=dataset.data_normalized,
#     full_mask_tensor=dataset.mask,
#     supervision_mask_tensor=dataset.supervision_mask,
#     ae_dim=64,
#     batch_size=1024,
# )

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model)

imputed_normalized = model.get_imputed_data()
imputed = dataset.inverse_transform(imputed_normalized)
```

### 可用参数

<details>
<summary>展开全部训练参数</summary>

#### 必填参数

- `--data-path PATH`：输入 CSV 文件路径。

#### 方法选择

- `--method {DMF,DCAE}`：插补方法，默认 `DMF`。

#### 通用网络参数

- `--hidden-dims DIMS`：隐藏层维度，以空格分隔，默认 `256 128`。
- `--batch-size SIZE`：训练批次大小，默认 `1024`。
- `--learning-rate RATE`：学习率，默认 `0.001`。
- `--weight-decay DECAY`：权重衰减，默认 `0.00001`。
- `--gradient-clip VALUE`：梯度裁剪阈值，默认 `1.0`。

#### DMF 专用参数

- `--embedding-dim DIM`：嵌入维度，默认 `64`。

#### DCAE 专用参数

- `--latent-dim DIM`：潜在空间维度，默认 `64`。
- `--num-encoder-blocks NUM`：编码器模块数量，默认 `2`。
- `--num-decoder-blocks NUM`：解码器模块数量，默认 `2`。
- `--dilation VALUE`：扩张系数，默认 `2`。

#### 损失权重

- `--mask-weight WEIGHT`：掩蔽预测损失权重，默认 `0.5`。
- `--reconstruction-weight WEIGHT`：重建损失权重，默认 `1.0`。

#### 训练控制

- `--max-epochs EPOCHS`：最大训练轮数，默认 `200`。
- `--early-stopping`：启用提前停止。
- `--patience PATIENCE`：提前停止的耐心轮数，默认 `20`。

#### 计算设备

- `--device {cpu,cuda,auto}`：计算设备，默认 `auto`。
- `--gpu-id ID`：GPU 编号，例如 `0`、`1`。

#### 输出设置

- `--results-dir DIR`：结果保存目录，默认 `./results`。
- `--log-interval INTERVAL`：日志记录间隔，默认每 `50` 步。
- `--progress-bar`：显示训练进度条。

</details>

## Benchmark 软件

[**下载 Windows 安装包（.exe）**](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.1/JINGWEI-Benchmark-Windows-Setup.exe) · [安装说明](software/benchmark/RELEASE-1.0.1.md) · [Linux CLI V1.0](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.0)

Windows 安装程序会自动下载并校验完整软件。安装一次后，从开始菜单启动即可。软件会恢复上次的数据、方法、参数、语言和窗口布局，并复用计算环境检查结果；软件、后端或显卡驱动变化时才重新检查。

使用独立软件比较 DMF/DCAE、PIMMS 和传统插补方法。**Python、R、conda 运行环境及各方法依赖均已打包。**

Windows 界面支持 **中文 / English**，默认加载内置的 `data/Alzheimer.csv`。JINGWEI 默认使用优化后的原生 PyTorch 训练方式，也保留 Lightning 方式供比较和复现。

CLI 默认评测 **全部 34 种方法、重复 5 次**，**每种方法每次重复超时 30 分钟**（`--timeout 1800`）。可用 `--methods` 选择部分方法。GUI 默认选择 DMF 和 DCAE，超时设置相同。

MICE-PMM 默认使用 Python 线性模型实现，通过后验抽样和预测均值匹配进行插补，保留 **5 次插补 × 5 轮更新**及全部预测变量。本地 CLI 检查中，完整默认矩阵约需 **40 秒**。使用 `--mice-backend r` 可切换至原始 R PMM；NAguideR 的 MICE-Norm/CART 仍使用 R 实现。实现细节和耗时见[性能说明（英文）](software/benchmark/PERFORMANCE.md)。

PIMMS 和迭代方法也进行了执行优化，保留用户指定的模型计算规模。可在 **高级设置 → 运行环境 → 对比方法执行方式** 中选择原生执行，或使用 `--baseline-engine native`，与上游执行方式比较。

| 平台 | 界面与安装方式 | 验证状态 |
| --- | --- | --- |
| Windows x86-64 | 原生 GUI + CLI 安装包 | 已验证安装、GUI、CLI 和卸载 |
| Linux x86-64 | CLI | 已在 WSL 中验证 |
| macOS arm64 / x86-64 | CLI，仅 CPU | 提供构建配置，尚未完成运行验证或发布安装包 |

**Windows 安装**：下载并运行一个 Windows 安装 EXE 即可；需要离线安装时，仍可下载带版本号的 EXE 和 BIN 分卷。

**Linux 安装**：将全部四个归档分卷、对应校验文件和 `install-linux.sh` 下载到同一文件夹，再运行 `sh install-linux.sh`。

[用户指南（英文）](software/benchmark/README.md) · [性能检查（英文）](software/benchmark/PERFORMANCE.md) · [构建文档（英文）](software/benchmark/packaging/README.md)

<details>
<summary>本地软件路径与验证详情</summary>

本地构建结果位于 `software/benchmark/dist/`：

```text
dist/
├── v1.0.1-windows-x86_64/
│   └── JINGWEI Benchmark/
│       ├── JINGWEI Benchmark.exe    # GUI
│       ├── jingwei-benchmark.exe    # CLI
│       └── _internal/              # 共享运行环境
├── jingwei-benchmark-linux-x86_64.tar
└── releases/
    ├── v1.0.1/                    # Windows 安装程序与校验文件
    └── v1.0.0/                    # Linux CLI 与原版安装程序
```

**2026-10-15 验证记录**：CPU 集成测试中的 **34 种方法**全部完成；Windows 回归测试 **125 项通过**，WSL/Linux **120 项通过**。独立软件包在移除宿主 conda/Python/R 路径后，完成了全部 **34 种方法的 CPU 检查**和 **5 种神经方法的 CUDA 检查**。

Linux 还在仓库目录之外、带空格的缓存路径中进行了验证，且不包含 Qt。这些检查用于确认软件可运行，不代表方法的科学排名或训练已收敛；完整数据的计时方式见性能说明。

V1.0 Windows 安装程序已在带空格的路径中验证，涵盖开始菜单快捷方式、版本注册、安装文件、中英文 GUI、原生图标、公开 CLI 和卸载。安装后的 Windows CLI 与 Linux CLI 均通过全部 34 种方法的 CPU 检查、5 种方法的 CUDA 检查，以及完整默认数据上的 Python MICE 检查。此前的控制台启动拦截在安装后的 V1.0 中未再次出现。

在默认 Alzheimer 矩阵上，SeqKNN、ImpSeq 和 MLE 会明确报告输入不适用的原因；QRILC 和 Python MICE 可在同一输入上完成运行。

尚未生成或运行验证 macOS 安装包；macOS CLI 必须在 Mac 上构建。

**2026-10-16，V1.0.1 桌面验证**：升级和重开检查通过，配置完整恢复，重开时环境探测进程为零。新版独立 Windows CLI 被本机 Smart App Control 拦截（错误 4551），因此尚未通过本机运行验证；桌面 GUI 正常运行，Linux CLI 继续使用 V1.0。未更改系统安全设置。

</details>

## 数据格式

Python 包训练使用的 CSV 文件建议采用以下结构：

- 第一行：表头，读取时跳过。
- 第一列：样本 ID 或名称，读取时跳过。
- 其余列：蛋白质表达数值。
- 缺失值：可使用 `0`、负值或 `NaN` 表示。

示例：

```csv
Sample_ID,Protein_1,Protein_2,Protein_3,...
Sample_001,1.23,0.45,NaN,...
Sample_002,2.34,0,1.67,...
Sample_003,1.45,1.23,2.89,...
```

Benchmark 软件支持自动判断表头和 ID 列，并可单独指定缺失值编码，详见[用户指南（英文）](software/benchmark/README.md#data-and-evaluation)。

## 输出文件

JINGWEI 会在结果目录中生成以下内容：

```text
results/
├── checkpoints/           # 模型检查点
├── logs/                 # TensorBoard 日志
└── outputs/
    └── {METHOD}_{DATASET}_{TIMESTAMP}/
        ├── config.json          # 训练配置
        ├── imputed_data.csv     # 插补后的蛋白质数据
        ├── training_metrics.csv # 训练损失记录
        └── model_final.ckpt     # 最终训练模型
```

## 更多示例

<details>
<summary>自定义网络、CUDA 加速与 CPU 训练</summary>

### 示例 1：自定义 DMF 网络结构

```bash
./src/JINGWEI.sh --data-path data/Alzheimer.csv \
    --method DMF \
    --hidden-dims 512 256 128 64 \
    --embedding-dim 128 \
    --mask-weight 0.3 \
    --learning-rate 0.0005 \
    --max-epochs 150 \
    --early-stopping \
    --progress-bar
```

### 示例 2：使用 GPU 加速 DCAE

```bash
./src/JINGWEI.sh --data-path data/Alzheimer.csv \
    --method DCAE \
    --device cuda \
    --gpu-id 1 \
    --latent-dim 128 \
    --num-encoder-blocks 3 \
    --num-decoder-blocks 3 \
    --dilation 4 \
    --batch-size 512
```

### 示例 3：使用 CPU 并指定输出目录

```bash
./src/JINGWEI.sh --data-path data/Alzheimer.csv \
    --device cpu \
    --results-dir ./my_results \
    --max-epochs 50 \
    --log-interval 10
```

</details>

## 方法介绍

### DMF（深度矩阵分解）

- 使用行嵌入和列嵌入捕获潜在结构。
- 可用于协同过滤形式的缺失值插补。
- 适用于具有结构化缺失模式的数据。

### DCAE（扩张卷积自编码器）

- 使用扩张卷积捕获长距离依赖。
- 适用于具有序列或结构信息的蛋白质数据。
- 用于建模复杂缺失模式。

## 训练监控

### TensorBoard

```bash
tensorboard --logdir results/logs
```

### 训练指标

可监控以下指标：

- `train_loss`：总体训练损失。
- `reconstruction_loss`：数据重建损失。
- `mask_loss`：缺失模式预测损失。

## 常见问题

<details>
<summary>问题排查与参数帮助</summary>

### 问题排查

1. **CUDA 显存不足**：减小 `--batch-size`，或使用 `--device cpu` 在 CPU 上训练。
2. **数据维度不匹配**：检查 CSV 格式及第一列的处理方式，确认表达矩阵中仅包含数值。
3. **训练速度慢**：可使用 `--device cuda` 加速；显存允许时可适当增大 `--batch-size`。
4. **插补效果不佳**：调整 `--mask-weight`（例如尝试 `0.1–0.8`）、尝试不同的 `--hidden-dims`，或启用 `--early-stopping`。

### 查看帮助

```bash
python -m jingwei.train --help
```

</details>

## 仓库结构

<details>
<summary>查看仓库目录</summary>

```text
JINGWEI/
├── README.md                 # 英文文档
├── README.zh-CN.md           # 中文文档
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── JINGWEI.sh            # 旧版训练脚本（可选）
│   ├── train.py              # Python 包训练入口的封装
│   └── jingwei/              # 独立发布的 Python 包
│       ├── __init__.py
│       ├── train.py           # 包内训练入口
│       ├── datasets.py        # 数据加载
│       ├── models.py          # DMF/DCAE 网络结构
│       ├── dmf.py             # DMF 训练器
│       └── dcae.py            # DCAE 训练器
├── software/
│   └── benchmark/            # 独立的 JINGWEI Benchmark 软件
│       ├── README.md          # 用户文档
│       ├── src/               # 软件源码
│       ├── tests/             # 软件测试
│       └── packaging/         # 维护者构建与打包文档
└── data/
    └── your_datasets.csv
```

</details>

## 许可证

本项目采用 [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) 许可证。

<details>
<summary>更新记录</summary>

### Benchmark V1.0.1 — Windows

- 重开时恢复上次的数据、方法选择、参数及窗口布局。
- 跨启动复用环境检查结果；运行环境变化时重新检查。
- 提供单个 Windows 安装 EXE，自动下载并校验完整软件。

### Benchmark V1.0（1.0.0）

- Windows 安装包包含中英文 GUI、CLI、原始图标和卸载功能。
- 提供 Linux 离线 CLI 安装程序，打包全部 Python/R 运行环境和 34 种方法。
- 共享评测掩蔽、第二次训练掩蔽、多变量样本分布 W2 和 Python 线性 MICE-PMM；每方法每次重复默认超时 30 分钟。
- 本次为独立软件发布，不重新发布已发布的 Python 包。

### Python 包 0.0.1

- 首次发布。
- 支持 DMF 和 DCAE 插补方法。
- 支持 GPU 加速。
- 提供可配置参数、TensorBoard 监控及训练记录。

</details>
