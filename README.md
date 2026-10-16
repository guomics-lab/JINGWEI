<p align="center">
  <strong>English</strong> · <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="./figures/Jingwei_logo.png" alt="JINGWEI logo" width="112" />
</p>

<h1 align="center">JINGWEI</h1>

<p align="center">
  <strong>Proteomic Data Imputation Framework</strong><br />
  Deep Matrix Factorization · Dilated Convolutional AutoEncoder
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#python-api-example">Python API</a> ·
  <a href="#benchmark-software">Benchmark</a>
</p>

---

JINGWEI is a deep learning framework for missing proteomic data imputation, supporting both **DMF (Deep Matrix Factorization)** and **DCAE (Dilated Convolutional AutoEncoder)** methods.

## Overview

| What you want to do | Use | Get started |
| --- | --- | --- |
| Apply **DMF / DCAE** to your data | JINGWEI Python package | [Quick start](#quick-start) · [Python API](#python-api-example) |
| Compare **34 imputation methods** | Standalone JINGWEI Benchmark software | [Benchmark guide](software/benchmark/README.md) |

## Features

| Methods | Training | Monitoring |
| --- | --- | --- |
| DMF and DCAE | Configurable architectures and batch sizes | TensorBoard logs |
| CPU / CUDA execution | Early stopping and tunable hyperparameters | Training metrics and checkpoints |

## Installation

```bash
conda create -n jingwei --override-channels -c conda-forge python=3.12 pip
conda activate jingwei
python -m pip install jingwei
```

`requirements.txt` lists the core JINGWEI package requirements. The benchmark
software is independent of this environment; it bundles its own runtimes and
requires no Python/R/conda installation.

## Usage

### Quick Start

```bash
# Basic usage with DMF method
python -m jingwei.train --data-path data/your_dataset.csv

# Use DCAE method with GPU 1
python -m jingwei.train --data-path data/Alzheimer.csv --method DCAE --device cuda --gpu-id 1

# Custom parameters with early stopping
python -m jingwei.train --data-path data/your_dataset.csv \
    --method DMF \
    --hidden-dims 512 256 128 \
    --embedding-dim 128 \
    --early-stopping \
    --max-epochs 100
```

### Python API Example

```python
import pytorch_lightning as pl

from jingwei import CSVDataset, DMFImputer, DCAEImputer

dataset = CSVDataset("data/Alzheimer.csv")

# Choose one of the two methods
model = DMFImputer(
    full_data_tensor=dataset.data_normalized,
    full_mask_tensor=dataset.mask,
    supervision_mask_tensor=dataset.supervision_mask,
    embedding_dim=64,
    hidden_dims=[256, 128],
    batch_size=1024,
)

# Or use DCAE
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

### Available Parameters

<details>
<summary>Expand all training parameters</summary>

#### Required Arguments

- `--data-path PATH`: Path to input CSV file

#### Method Selection

- `--method {DMF,DCAE}`: Imputation method (default: DMF)

#### General Network Parameters

- `--hidden-dims DIMS`: Hidden layer dimensions, space-separated (default: "256 128")
- `--batch-size SIZE`: Batch size for training (default: 1024)
- `--learning-rate RATE`: Learning rate (default: 0.001)
- `--weight-decay DECAY`: Weight decay for optimizer (default: 0.00001)
- `--gradient-clip VALUE`: Gradient clipping value (default: 1.0)

#### DMF Specific Parameters

- `--embedding-dim DIM`: Embedding dimension (default: 64)

#### DCAE Specific Parameters

- `--latent-dim DIM`: Latent dimension (default: 64)
- `--num-encoder-blocks NUM`: Number of encoder blocks (default: 2)
- `--num-decoder-blocks NUM`: Number of decoder blocks (default: 2)
- `--dilation VALUE`: Dilation factor (default: 2)

#### Loss Weights

- `--mask-weight WEIGHT`: Weight for mask prediction loss (default: 0.5)
- `--reconstruction-weight WEIGHT`: Weight for reconstruction loss (default: 1.0)

#### Training Control

- `--max-epochs EPOCHS`: Maximum training epochs (default: 200)
- `--early-stopping`: Enable early stopping
- `--patience PATIENCE`: Patience for early stopping (default: 20)

#### Device Settings

- `--device {cpu,cuda,auto}`: Device to use (default: auto)
- `--gpu-id ID`: Specific GPU ID to use (0, 1, etc.)

#### Output Settings

- `--results-dir DIR`: Directory for saving results (default: ./results)
- `--log-interval INTERVAL`: Logging interval in steps (default: 50)
- `--progress-bar`: Show progress bar during training

</details>

## Benchmark software

[**Download JINGWEI Benchmark V1.0**](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.0) · [Installation guide](software/benchmark/RELEASE-1.0.md)

Compare DMF/DCAE with PIMMS and traditional imputation methods using the standalone
benchmark application. **Python, R, conda and method dependencies are bundled.**

The Windows interface supports **中文 / English** and opens the bundled
`data/Alzheimer.csv` by default. JINGWEI uses optimized native PyTorch execution;
the original Lightning runner remains available for comparison and replay.
The benchmark CLI defaults to **all 34 methods and 5 repeats**, with a
**30-minute timeout per method/repeat** (`--timeout 1800`); use `--methods`
to select a subset. The GUI initially selects DMF and DCAE with the same timeout.
MICE-PMM now defaults to a Python linear implementation with posterior draws
and predictive mean matching, using all **5 imputations × 5 sweeps** and all
predictors. The full default matrix completed in about **40 seconds** in the
local CLI check. Select `--mice-backend r` for original R PMM; NAguideR
MICE-Norm/CART remain R methods. Implementation details and timings are recorded
in the [performance notes](software/benchmark/PERFORMANCE.md).
PIMMS and iterative methods also have execution optimizations that retain
their requested model budgets. Use **Advanced → Runtime → Baseline execution**
or `--baseline-engine native` to compare against upstream execution.

| Platform | Interface | Status |
| --- | --- | --- |
| Windows x86-64 | Native GUI + CLI installer | Installation, GUI, CLI and uninstall validated |
| Linux x86-64 | CLI | Validated in WSL |
| macOS arm64 / x86-64 | CLI, CPU only | Build configuration; not runtime-tested |

[User guide](software/benchmark/README.md) · [Performance checks](software/benchmark/PERFORMANCE.md) · [Build documentation](software/benchmark/packaging/README.md)

<details>
<summary>Local software paths and validation</summary>

Local builds are under `software/benchmark/dist/`:

```text
dist/
├── v1-windows-x86_64/
│   └── JINGWEI Benchmark/
│       ├── JINGWEI Benchmark.exe    # GUI
│       ├── jingwei-benchmark.exe    # CLI
│       └── _internal/              # Shared runtime
├── jingwei-benchmark-linux-x86_64.tar
└── releases/v1.0.0/                # Installers, data volumes and checksums
```

Windows: download the setup EXE and both BIN volumes into one folder, then run
the EXE. Linux: download all four archive parts, its checksums and installer,
then run `sh install-linux.sh`. See the [installation guide](software/benchmark/RELEASE-1.0.md).

Validated on **2026-10-15**: all **34 methods** completed the CPU integration
fixture; regression tests passed on Windows (**125**) and WSL/Linux (**120**).
The current standalone bundles were checked with host conda/Python/R paths
removed, using all **34 CPU methods** on the integration fixture and all
**5 CUDA neural methods**.
Linux was also tested outside the checkout, without Qt, with a cache path
containing spaces. These are execution checks, not a scientific ranking or
convergence test; see the performance notes for the full-data timing protocol.

The V1.0 Windows installer was tested in a path containing spaces, including
its Start menu shortcut, version registration, installed payload, Chinese/English
GUI, native icons, public console and uninstall. The installed console and Linux
CLI both passed all 34 CPU methods, five CUDA methods and full-data Python MICE.
The previous console startup block did not recur in this installed V1.0 build.
SeqKNN, ImpSeq and MLE report their expected unsupported-input reasons on the
default Alzheimer matrix; QRILC and Python MICE complete on that same input.

No macOS artifact has been produced or runtime-tested. Its CLI build must run on
a Mac.

</details>

## Data Format

For Python package training, the input CSV file should have the following format:

- First row: Header (will be skipped)
- First column: Sample IDs/names (will be skipped)
- Remaining columns: Protein expression data
- Missing values: Use 0, negative values, or NaN

Example:
```csv
Sample_ID,Protein_1,Protein_2,Protein_3,...
Sample_001,1.23,0.45,NaN,...
Sample_002,2.34,0,1.67,...
Sample_003,1.45,1.23,2.89,...
```

The benchmark application detects headers and ID columns automatically and lets
you choose the missing-value encoding; see the [benchmark guide](software/benchmark/README.md#data-and-evaluation).

## Output Files

JINGWEI generates the following outputs in the results directory:

```
results/
├── checkpoints/           # Model checkpoints
├── logs/                 # TensorBoard logs
└── outputs/
    └── {METHOD}_{DATASET}_{TIMESTAMP}/
        ├── config.json          # Training configuration
        ├── imputed_data.csv     # Imputed protein data
        ├── training_metrics.csv # Training loss history
        └── model_final.ckpt    # Final trained model
```

## Examples

<details>
<summary>More examples: custom architectures, CUDA and CPU training</summary>

### Example 1: DMF with Custom Architecture

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

### Example 2: DCAE with GPU Acceleration

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

### Example 3: CPU Training with Custom Output Directory

```bash
./src/JINGWEI.sh --data-path data/Alzheimer.csv \
    --device cpu \
    --results-dir ./my_results \
    --max-epochs 50 \
    --log-interval 10
```

</details>

## Method Descriptions

### DMF (Deep Matrix Factorization)

- Uses row and column embeddings to capture latent patterns
- Suitable for collaborative filtering-style missing data
- Good for datasets with structured missing patterns

### DCAE (Dilated Convolutional AutoEncoder)

- Uses dilated convolutions to capture long-range dependencies
- Suitable for sequential or structured protein data
- Better for complex missing data patterns

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir results/logs
```

### Training Metrics

Monitor the following metrics:
- `train_loss`: Overall training loss
- `reconstruction_loss`: Data reconstruction quality
- `mask_loss`: Missing data pattern prediction accuracy

## Troubleshooting

<details>
<summary>Common issues and parameter help</summary>

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch-size`
   - Use `--device cpu` for CPU training

2. **Shape Mismatch Errors**
   - Check CSV format (ensure first column is skipped)
   - Verify data contains only numeric values

3. **Slow Training**
   - Use GPU acceleration with `--device cuda`
   - Increase `--batch-size` if memory allows

4. **Poor Performance**
   - Adjust `--mask-weight` (try 0.1-0.8)
   - Experiment with different `--hidden-dims`
   - Enable `--early-stopping`

### Getting Help

For help with parameters:
```bash
python -m jingwei.train --help
```

</details>

## File Structure

<details>
<summary>View the repository structure</summary>

```
JINGWEI/
├── README.md                 # English documentation
├── README.zh-CN.md           # Chinese documentation
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── JINGWEI.sh            # Legacy training script (optional)
│   ├── train.py              # Thin wrapper for package entry
│   └── jingwei/              # Published Python package (left untouched)
│       ├── __init__.py
│       ├── train.py           # Package training entry
│       ├── datasets.py        # Data loading utilities
│       ├── models.py          # Model architectures (DMF/DCAE)
│       ├── dmf.py             # DMF trainer
│       └── dcae.py            # DCAE trainer
├── software/
│   └── benchmark/            # Standalone JINGWEI Benchmark application
│       ├── README.md          # End-user documentation
│       ├── src/               # Application source
│       ├── tests/             # Application tests
│       └── packaging/         # Maintainer build/packaging documentation
└── data/
    └── your_datasets.csv
```

</details>

## License

This project is licensed under the [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).


<details>
<summary>Changelog</summary>

### Benchmark V1.0 (1.0.0)

- Windows installer with Chinese/English GUI, CLI, original icon and uninstall.
- Offline Linux CLI installer; all Python/R runtimes and 34 methods bundled.
- Shared evaluation masks, second training mask, sample-distribution W2 and
  Python linear MICE-PMM; default method timeout is 30 minutes.
- Application release only; the published Python package is unchanged.

### Python package 0.0.1

- Initial release
- Support for DMF and DCAE methods
- GPU acceleration
- Comprehensive parameter configuration
- TensorBoard integration

</details>
