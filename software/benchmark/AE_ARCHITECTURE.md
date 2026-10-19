# Experimental global AE

开发版新增可选的 **全局掩码 AE**，用于改进 DCAE 的重建路径。GUI 在
**公平评测 → DCAE 参数 → AE 架构**选择；CLI 使用
`--ae-architecture global`。默认 `native` 和原 notebook 回放保留原架构。
这是源码开发功能，已发布的 V1.0.3 安装包不含此选项。

The experimental `global` architecture uses two input channels: normalized
abundance and the supplied observation mask. A genuine normalized zero is
therefore distinct from an unavailable entry. Masked values are discarded
before any learned operation.

Residual dilated convolutions encode each row. A flattened, fully connected
bottleneck mixes information across the protein profile; a residual dense
decoder produces a separate output for every protein. The output projection
starts near the training-derived protein mean. Light dropout regularizes the
bottleneck and decoder. No input-to-output identity shortcut bypasses the
bottleneck. The auxiliary mask head reads the latent code, allowing its loss
to update the encoder without directly constraining the reconstruction head.
`--mask-weight 0` disables this auxiliary objective.

The observation mask is an input, so auxiliary mask prediction can exploit
that information. Its accuracy is not evidence of identifying a biological
MNAR process, and this comparison does not isolate the benefit of each component.
Protein order still matters to the convolutional encoder; this is not a
permutation-equivariant model. Fixed latent/hidden widths give parameter growth
linear in the number of proteins. Configurations requiring more than 50 million
parameters fail before allocating the network.

## Calling the model

In the Windows source GUI, select **Fair benchmark**, highlight DCAE and open
**Parameters**. Select **Global mask-aware AE (experimental)**. The architecture
is recorded in job parameters, logs and metrics; results show a Global AE suffix.
Historical replay rejects the new architecture instead of silently changing its
model. Both the PyTorch and Lightning runners support CPU/CUDA; the macOS
application policy remains CPU only.

```bash
jingwei-benchmark benchmark --input data.csv --methods jingwei-dcae \
  --evaluation-protocol fair --ae-architecture global \
  --max-epochs 300 --device cuda --output results/global-ae
```

This command uses normal CLI defaults for the remaining parameters. The
diagnostic architecture comparison below instead uses latent dimension 256,
mask-head/decoder width 128, three encoder and decoder blocks, dropout 0.1,
AdamW learning rate 0.00005, weight decay 0.00001, mask weight 0.5, and one
full-row batch per epoch (batch size 1024). These are experimental settings,
not a universally recommended training duration. Method JSON or GUI parameters
can set them explicitly.

## Validation protocol

The development comparison reuses the Alzheimer outer evaluation mask, but
both architectures fit normalization only to their training observations.
It retains that example's first-column removal and positive-value log2
preprocessing (210 samples × 1541 proteins). Normal fair CLI import can retain
all numeric columns; the example command above therefore does not by itself
recreate this exact diagnostic mask.
It therefore differs from historical replay, which used pre-mask statistics.
An inner 20% MCAR mask is drawn exclusively from outer-training observations;
it leaves at least one fitting observation per nonempty protein column. This
inner mask is excluded from normalization and gradient supervision. A second
20% corruption of the fitting entries supplies denoising inputs.

Each architecture is trained for at most 2000 epochs and checked every 50
epochs. The best inner-validation epoch determines a fresh fit on all
outer-training observations. Outer targets are loaded only by the separate
final scoring process after all fits finish. Model/inner-mask seeds 0, 1 and 2
are paired across architectures. A protein-mean baseline uses the same available
observations. This is an exploratory comparison on one fixed outer mask; its
MCAR inner validation, dataset and initialization scope limit generalization.

## Alzheimer measurements

All runs score the same 229,401 held-out entries. Measurements on 2026-10-19:

| Seed | Native AE MSE | Global AE MSE | Native epochs | Global epochs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.679168 | 0.543732 | 2000 | 300 |
| 1 | 0.708301 | 0.539789 | 2000 | 250 |
| 2 | 0.687146 | 0.538422 | 2000 | 300 |
| Mean | **0.691538** | **0.540648** | | |
| Sample SD across seeds | 0.015055 | 0.002757 | | |

The global variant reduces mean MSE by **21.82%** in this comparison. The
protein-mean baseline is **0.632483**. Parameter counts are 719,798 for native
DCAE and 1,359,850 for the global variant. The comparison changes the complete
architecture, including its mask head and initialization; it is not a component
ablation. Native selections reached the 2000-epoch budget, so the result does
not establish native performance with a larger budget.

The earlier 1500–3000 epoch diagnostic used historical pre-mask statistics and
one mask. This experiment adds training corruption and fits normalization after
holdouts, which is why its native baseline is recomputed. These measurements
support this optional model on one dataset/mask; they do not establish broader
superiority or reproduce a historical manuscript table.

Local evidence is retained under `results/se-qa/ae-design/`: the fitting script,
inner validation histories, source/input settings, selected weights, separate
outer scorer and `confirmation/outer-scores.csv`. No dependency additions are
required. The published Python package and the notebook snapshot are separate
from this development variant.
