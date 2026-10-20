# DCAE implementation

当前源码已直接替换 DCAE 的实现：显式输入掩码、全局瓶颈和全连接解码器。
Python API、CLI 与 GUI 中仍然只有原来的 DCAE 方法，不设置 Alzheimer 专用方法或预设。
目前仅更新 Alzheimer 的实验结果，其他数据集数值保留，等待后续重新评测。
Windows 1.0.5 构建包含此实现；已发布的 Python 包和旧安装包保持原状。

## Architecture

`jingwei.models.DCAE`, `jingwei.DCAE` and `DCAEImputer` now use the updated
architecture by default. The bundled benchmark uses the same implementation.
Its version metadata is `global-ae-v1`; the method ID remains `jingwei-dcae`.
There is no dataset detection or dataset-specific architecture selection.

Two channels contain standardized observed values and their explicit binary
observation mask. Unavailable values are discarded before any learned
operation; an observed normalized zero remains distinct from a missing zero.
A 16-channel residual dilated-convolution encoder downsamples the profile.
A flattened global bottleneck feeds a residual dense decoder with a separate
output for each protein. There is no input-to-output identity shortcut.
The mask head reads the shared latent vector. Dropout regularizes the bottleneck
and decoder; the reconstruction projection starts near the training-derived
protein mean. The input mask can facilitate the auxiliary task, so mask
prediction does not identify a biological MNAR mechanism. Protein order still
matters to the convolutional encoder.

The current GUI configuration has 1,542 features, latent size 64, hidden width 256,
two encoder/decoder blocks and dilation 2. It contains 1,489,372 parameters.
Configurations requiring more than 50 million parameters are rejected before allocation.
Existing convolutional-model checkpoints have different keys and cannot be
loaded into the updated architecture. `LegacyDCAE` and explicit
`ae_architecture="native"` remain only for historical comparisons.

## Python and software

```python
from jingwei import DCAE, DCAEImputer

network = DCAE(input_feature_dim=1542, ae_dim=64,
               mask_predictor_hidden_dim=256,
               num_encoder_blocks=2, num_decoder_blocks=2, dilation=2)
reconstruction, mask_logits = network(normalized_values, observation_mask)

imputer = DCAEImputer(
    full_data_tensor=normalized_values,
    full_mask_tensor=input_mask,
    supervision_mask_tensor=supervision_mask,
    ae_dim=64, mask_predictor_hidden_dim=256,
    num_encoder_blocks=2, num_decoder_blocks=2, dilation=2,
    learning_rate=0.001, lambda_mask=0.5, batch_size=64,
)
```

Supply normalization fitted on the training observations only. The supervision
mask includes training-corrupted observations and excludes evaluation targets.
Both PyTorch and Lightning runners preserve supplied observations and support
CPU/CUDA. macOS remains CPU only. In the source GUI, select DCAE as usual;
there is no extra architecture selector or Alzheimer-specific preset.

```bash
jingwei-benchmark benchmark --input data.csv --methods jingwei-dcae \
  --evaluation-protocol fair --max-epochs 50 --device cuda --output results/dcae
```

This is a single run with the chosen settings. The current AD GUI reference
configuration and its scope are described below.

## Alzheimer GUI reference

The updated AD manuscript results use the actual Windows 1.0.5 GUI job runner,
with its saved configuration: original-repository MNAR, GUI base seed 0, one run,
50 epochs, batch size 64, latent dimension 64, hidden dimensions 256 and 128,
learning rate 0.001 and mask weight 0.5. This records the selected GUI settings;
a fresh form can use a different missingness selection.

Automatic import keeps all 1,542 numeric protein columns. There are 255,165
initial observations, 229,386 test holdouts and 25,779 method-input observations.
The platform creates one shared test mask and no training mask. Sixteen empty
columns are excluded from the shared score, leaving 229,311 test targets.
Each method owns its training; the JINGWEI reference run has no additional
corruption, inner-validation epoch selection or refit. Normalization uses only
the remaining observations. No standard deviation is reported for one run.

| Method | GUI MSE | GUI worker time (s) |
| --- | ---: | ---: |
| DMF | 0.681597 | 4.499 |
| DCAE | 0.704898 | 7.323 |
| MissForest | 0.589278 | 377.880 |
| KNN | 0.623089 | 2.001 |
| MICE-PMM | 0.674742 | 12.820 |
| MIN | 2.769302 | 0.650 |
| PIMMS DAE | 0.677432 | 9.470 |
| PIMMS VAE | 0.614723 | 25.608 |

These are the software's current results under this configuration. DMF/DCAE do
not outperform MissForest here. All other dataset numerical results remain
unchanged. Old notebook outputs and the superseded AD-specific selection/refit
comparison are removed as reference records. Old AD ablation/order tables are
removed from the revised manuscript, rather than populated with unmeasured values.

Local evidence: `results/gui-ad-reference-20261020/`, including the complete GUI
task, configuration, masks, predictions, versions and independently checked
metrics. The task can be reopened with **Open task…**. GUI `time_s` covers worker
initialization, algorithm execution and prediction serialization. Timings depend
on hardware. Sample W2 remains undefined because no reference sample is complete
on the retained common feature set.
