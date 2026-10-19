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
  <a href="https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe"><strong>下载 Windows 软件安装包（.exe）</strong></a> ·
  <a href="https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.3">Linux CLI</a>
</p>

只需下载一个 EXE 并运行，其余文件由安装程序自动获取。安装后从 **开始菜单 → JINGWEI Benchmark** 打开软件。

JINGWEI 是一个用于蛋白质组数据缺失值插补的深度学习框架，提供 **DMF（深度矩阵分解）** 和 **DCAE（扩张卷积自编码器）** 两种方法。

## 使用方式

| 你的需求 | 使用产品 | 开始使用 |
| --- | --- | --- |
| 用 **DMF / DCAE** 处理自己的数据 | JINGWEI Python 包 | [快速开始](#快速开始) · [Python API](#python-api-示例) |
| 比较 **35 种插补方法** | 独立的 JINGWEI Benchmark 软件 | [Windows 安装包](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe) · [使用指南](software/benchmark/README.md) |

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

[**下载 Windows 安装包（.exe）**](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe) · [安装说明](software/benchmark/RELEASE-1.0.3.md) · [Linux CLI](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.3)

安装一次后，从开始菜单打开 **JINGWEI Benchmark**。Python、R、conda 及各方法依赖均已打包，无需另装。Windows 原生界面支持 **中文 / English**，默认加载内置的 `data/Alzheimer.csv`。

选择数据、方法、评测方案、输出目录和设备后，点击 **开始**。GUI **每次只运行一次**，已删除重复次数选项。**日志**实时显示进度；启动时结果为空，通过 **打开任务…**手动查看。表单设置和环境检查缓存独立保留。

- **原 notebook 回放**：恢复旧 CSV 预处理、`CSVDataset.get_mask()`、单次掩蔽、原方法和评分流程。它会无条件丢弃第一列，并使用掩蔽前统计量。
- **公平评测**：支持常规 CSV／AnnData，先统一构造评测掩蔽，再从剩余观测值构造训练掩蔽；标准化不使用评测真值。此前的自定义分位数机制单独标注。

新表单默认使用原 notebook 回放；已有配置保留其评测方案。**恢复原代码不等于旧分数全部重现**。阿尔兹海默数据的掩蔽数量一致，9 个旧 MSE 中 6 个与原显示精度一致；MF、DAE、DCAE 仍有差异。另通过本机直接执行旧函数交叉检查神经方法。结果和差异见[回放说明](software/benchmark/NOTEBOOK_REPLAY.md)和[测试报告](software/benchmark/SE_TESTS.md)。

CLI 默认评测 **全部 35 种方法、单次运行**，每种方法默认超时 **30 分钟**。CLI 仍可显式指定多次公平评测。GUI 默认勾选 DMF 和 DCAE。历史回放使用原 Python 线性 MICE；公平方案另有 Python／R MICE-PMM 及其他 R 方法。详见[用户指南](software/benchmark/README.md)。

| 平台 | 软件 | 计算设备 |
| --- | --- | --- |
| Windows x86-64 | 原生 GUI + CLI 安装包 | CPU / NVIDIA CUDA |
| Linux x86-64 | 离线 CLI，在 WSL 中测试 | CPU / NVIDIA CUDA |
| macOS | 提供 CLI 构建配置；本次无二进制及运行验证 | 仅 CPU |

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
- 提供 Linux 离线 CLI 安装程序，打包全部 Python/R 运行环境和 35 种方法。
- 共享评测掩蔽、第二次训练掩蔽、多变量样本分布 W2 和 Python 线性 MICE-PMM；每方法每次重复默认超时 30 分钟。
- 本次为独立软件发布，不重新发布已发布的 Python 包。

### Python 包 0.0.1

- 首次发布。
- 支持 DMF 和 DCAE 插补方法。
- 支持 GPU 加速。
- 提供可配置参数、TensorBoard 监控及训练记录。

</details>
