# Original notebook replay

中文：GUI 选择 **评测方案 → 原 notebook 回放**。GUI 每次只评测一次，没有重复次数选项。该方案恢复旧 CSV 流程：无条件丢弃第一列、对正值做 log2、用掩蔽前的 float32 数据计算均值和标准差，然后调用原 `CSVDataset.get_mask()`。它只构造一次掩蔽，不构造第二次训练掩蔽；评分排除全空蛋白列和非有限预测。常规 CSV/AnnData 和双重掩蔽请选 **公平评测**。

原 notebook 的历史行为是明确保留的，包括 z-score 与未标准化阈值的比较。**恢复原代码流程不等于保证历史神经网络分数逐位一致。**旧记录没有保存完整依赖、运行环境、模型权重和中间随机状态。原 notebook 只报告 MSE 和耗时；新增指标没有旧数值可比，耗时也取决于硬件。对照结果及差异见本目录的测试报告。

## Source and execution

The reference is `benchmark.ipynb` and its Python modules at commit
`98686b54fdb9918a4d4cc9d99ac4880a6dadcefe`, before notebook removal. Python
snapshots live in the private `_legacy` application namespace. No notebooks are
restored and the separately published `jingwei` package is not imported.

1. Read a raw CSV with its header and discard the first column unconditionally,
   even when that column contains protein intensities. This is a historical
   compatibility rule, not automatic ID detection.
2. Apply `log2` to positive values, preserve the original CSV round trip, and
   load float32 values through the original `CSVDataset`. Fit per-feature mean
   and population standard deviation before evaluation masking.
3. Draw the original Torch `rand_like` mask. Compare normalized values with
   unnormalized `mean−3*std`, `mean−2*std`, and `mean−std`. The strict bands use
   probabilities 0.9, 0.5 and 0.3. Keep strict boundary behavior, column-major
   random draws, and the original missing-value handling. Do not calibrate to
   the percentage controls or repair empty rows/columns.
4. Use the same single mask for every selected method. No additional training
   mask is added in this profile. The fair profile retains sequential E/T masks
   and fits normalization without evaluation truth.
5. Run original methods in their recorded order: MissForest, PIMMS VAE, KNN,
   NMF, PIMMS DAE, per-protein MIN, linear Python MICE, DMF, DCAE. Only selected
   methods run; selecting a subset changes subsequent random streams. Other
   comparison methods run afterward on this same corrupted matrix.
6. Carry Python, NumPy and CPU/CUDA Torch RNG state between isolated workers.
   Import training dependencies before restoring that state. Preserve the
   persistent DataLoader behavior with one worker instead of twenty. Native
   networks, shuffled row behavior, full-mask auxiliary BCE, original parameters
   and 50 actual epochs are retained. The old 500/2000 variables were unused.
7. Score on original dynamic holdouts, excluding proteins with no remaining
   observations and nonfinite predictions, as the notebook did. Report
   `n_held_out`, `n_evaluated`, `n_excluded` and `zero_cols` explicitly. Use the
   original sklearn float32 MSE. New metrics and prediction exports remain
   available; undefined sample W2 stays empty when reference samples are
   incomplete on the selected proteins.

Original PIMMS uses latent dimension 50, hidden layer 512, batch size 10 and
200 requested epochs; DAE runs on CPU and VAE uses the selected supported
device. Original MICE is linear chained-equation imputation, not PMM. Original
MissForest is the Python forest script, not the separate R backend. NMF has a
memory-layout compatibility wrapper for current sklearn; its objective,
initialization, seed and budget are unchanged. Forest parallelism respects the
CPU thread budget. All effective parameters and protocol details are saved.

## Full Alzheimer comparison

The source's saved output uses 210 × 1541 entries, 254,968 initial observations,
229,546 dynamic holdouts, 25,422 remaining observations and 20 empty protein
columns. Excluding those columns leaves 229,401 evaluated entries. These counts
and the deterministic KNN/MIN/NMF/MICE scores are regression checks.

From the installed CLI (the bundled Alzheimer CSV is the default input):

Run the historical CLI in a writable working directory: its original PIMMS
adapters also write loss plots and recorders there. The GUI runs inside the
selected task folder.

```bash
jingwei-benchmark benchmark --evaluation-protocol original-notebook --seed 0 \
  --methods missforest pimms-vae knn nmf pimms-dae feature-minimum mice-pmm jingwei-dmf jingwei-dcae \
  --device auto --timeout 5400 --output alzheimer-notebook
```

This explicitly allows 90 minutes per method for historical MissForest, whose
saved runtime exceeded 30 minutes. The normal software timeout remains 1800
seconds. Use Advanced → Runtime in the GUI to change it. A single run is the
default; the CLI can still request repeated fair evaluations explicitly.

The `original-benchmark` training preset alone restores only model parameters.
Likewise, the fair profile's `mnar-jingwei-original` mechanism restores only the
threshold kernel, with statistics restricted to each stage's eligible entries.
Neither is a substitute for the complete `original-notebook` profile. The older
`mnar-jingwei` ID remains available as **custom quantile MNAR** for old settings;
it is not relabeled as the original repository or a paper reproduction.
