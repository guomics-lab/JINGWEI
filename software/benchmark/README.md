# jingwei

Standalone software for comparing JINGWEI DMF/DCAE with PIMMS and traditional
proteomics imputation methods. This application is separate from the published
JINGWEI Python package. **Benchmark users do not install Python, R, conda, or
method dependencies.**

| System | Application | Compute |
| --- | --- | --- |
| Windows x86-64 | Native GUI and console CLI | CPU / NVIDIA CUDA |
| Linux x86-64 | Console CLI | CPU / NVIDIA CUDA |
| macOS Apple Silicon / Intel | Console CLI | CPU only |

Each release targets one operating system and architecture. A CUDA build also
runs on CPU. GPU execution requires the machine's NVIDIA driver.

## Windows

Download the complete [Windows offline installer (.exe)](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.7/jingwei-1.0.7-windows-x86_64-offline.exe).
Run it and click **Browse** to select the installation folder, including another
drive. All CPU/CUDA, Python and R components are inside this one file; installation
needs no network connection or adjacent BIN files. You can choose where to save
the EXE in your browser before downloading. Then open **Start → jingwei**.
See the [installation guide](RELEASE-1.0.7.md).
The GUI uses the original JINGWEI bird for the
application, window and Windows taskbar.

The application is named **jingwei**. Its main Run settings have
only **Task**, **Missingness** and **Mask rate**. Choose data, methods and an
output folder, then **Run**. Device and model parameters are under **Advanced**.
Each GUI task runs once; there are no protocol, repeat, training-mask or seed
controls. DMF/DCAE are selected by default and `data/Alzheimer.csv` is bundled.

Switch **中文 / English** at the top right. **Preview** shows the detected
header/ID and matrix; **Import settings** can override numeric IDs or headers.
**Runtime** reports method availability. Method-specific **Parameters** override
shared training settings.

Each run creates its own folder beneath the selected **Output folder**. The
**Log** tab opens at run start and shows live progress and method output.
Configuration is locked during submission, execution and cancellation; stop,
logs and results remain available. **Reuse settings** restores the current task's
common settings and effective method parameters. The **Open task** button has
been removed. Blank task names use `task-YYYYMMDD-HHMMSS` (local time); generated
folder names add a unique suffix. Startup leaves results empty. Configuration, output folder and window layout are saved
independently of results. Cancellation, plots and ZIP export remain available.

Environment checks complete before the main window opens, with a small startup
screen supporting cancel and retry. Valid cached checks are reused; changing
runtime paths, software or drivers invalidates the cache. **Runtime → Refresh**
checks again. The installer presents a plain-text `LICENSE.txt`.

### Masking and result records

A benchmark constructs **one shared test mask**. Every method receives the same
corrupted dataset, with test truth removed. DMF/DCAE receive one additional fixed
training-corruption mask from the remaining observations, using the selected
mechanism/rate. These hidden training values remain supervised targets; the mask
is not redrawn each epoch. Other methods use the common input and retain their
native training procedures. All five neural methods default to exactly 500
epochs, no early stopping or extra internal validation split, and final weights.

In **Impute**, **Training missingness** and **Training mask rate** construct the
JINGWEI DMF/DCAE training input. Artificially hidden observations remain available
as supervised targets. Inference receives the original observations; exports
preserve them and fill native missing entries. Comparison methods receive the
original dataset with no JINGWEI training mask. Imputation creates no test mask.

**MNAR · JINGWEI** uses within-protein standardized log2 intensities and
`p = sigmoid(-z + b)`, calibrating `b` to the chosen mask rate. Lower abundance
means greater missingness probability. The explicit original-repository
compatibility option retains historical `get_mask()` behavior and is not the
current AD reference. Mask arrays, seeds and realized rates are saved with each
run. Protocol 4 stores an unchanged common method input plus a separate JINGWEI
training-input bundle. Only the latter contains the extra training mask.

Original MNAR may leave a protein with no observations. In that case,
these columns are removed from comparison-method inputs and excluded from the
shared scoring scope. `feature_scope.json` records their IDs and counts;
metrics include `n_held_out`, `n_evaluated` and `n_excluded`. Exports keep those
columns as missing, without inventing estimates. Nonfinite predictions on
retained features remain errors rather than being silently omitted.

MSE, RMSE, MAE, NRMSE, correlations and sample W2 are exported with their scope.
The updated Alzheimer manuscript results use the current GUI task exports,
including actual model settings, input hash, shared masks and per-method timing.
Obsolete notebook result records are removed. Old compatibility code remains
isolated from this reference workflow. See [verification](SE_TESTS.md).


## Linux and macOS

For Linux V1.0.7, download all parts, its checksums and `install-linux.sh` from
the [release page](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.7).
Run `sh install-linux.sh`; the command is placed in `~/.local/bin`.
Custom installation locations are described in the [installation guide](RELEASE-1.0.7.md).
No macOS binary is included in this release. From the installed application folder:

```bash
./jingwei --help
./jingwei methods
./jingwei inspect --input data.csv
./jingwei benchmark --input data.csv \
  --methods jingwei-dmf jingwei-dcae mean knn bpca missforest \
  --mechanism mnar-jingwei-v2 --mask-rate 0.2 \
  --repeats 1 --device cpu --output results/my-benchmark
```

The first launch expands the bundled runtime into the user's local cache and
repairs its internal paths **offline**. Subsequent launches reuse that runtime.
No packages are downloaded or installed into system/conda environments. The
application folder can be moved; its bundled runtime is identified by its hash.
`JINGWEI_BENCHMARK_CACHE` can select a different writable cache directory. A
CUDA distribution is large and needs enough disk space for both the archive and
expanded runtime. Linux/macOS distributions do not include the GUI or Qt.

Linux supports `--device cuda --gpu-id 0`. macOS exposes only `--device cpu`.
`benchmark` defaults to all 35 methods; `--methods` selects a smaller set.
Default settings make 35 method attempts (one evaluation), with a **30-minute
(1800-second) timeout per method/repeat**. Use `--timeout SECONDS` to change it;
the Windows GUI exposes the same setting under Advanced → Runtime.
Wide inputs can reach that limit for iterative methods;
timeouts are recorded and the comparison continues.
Check `methods` before starting;
missing bundled backends are a distribution defect, not an instruction to
install dependencies. CLI options are documented by `benchmark --help`.
If `--input` is omitted, the bundled Alzheimer dataset is used, regardless of
the current working directory. Explicit `--input` always takes precedence.

## Execution speed

The development source directly updates [DCAE](AE_ARCHITECTURE.md) with an
explicit input mask, global bottleneck and dense decoder. Select DCAE as usual;
there is no extra method or dataset-specific preset. Only Alzheimer result
values have been updated using current GUI exports; other datasets await
reevaluation. Previously published release binaries retain their original implementation.

`mice` / `mice-pmm` now defaults to the application's **Python linear MICE-PMM**
implementation. It uses posterior draws from ridge-regularized linear
conditionals and type-1 predictive mean matching, with **5 imputations × 5 full
sweeps**, five candidate donors and all other proteins as predictors. Sample
Gram caching and incremental column updates accelerate the same conditional
model; no forests or predictor screening are used. Each run saves all five
completed matrices in `mice_draws.npz`, their seeds, scales and update counts.
Observed values are preserved. A feature with one distinct observed donor value
has a deterministic PMM result; an entirely missing feature is rejected.

Use `--mice-backend r` or **Advanced → Runtime → MICE implementation** for the
original R PMM implementation. NAguideR MICE-Norm/CART retain their original R
implementations. The Python ridge prior and variance draw differ from R mice's
predictor filtering and residual degrees of freedom, so results record the
implementation explicitly. The average of completed matrices is used for
reconstruction evaluation; it is not inferential pooling.

Comparison methods default to `--baseline-engine optimized`. Choose
**Advanced → Runtime → Baseline execution → Native (replay)** or
`--baseline-engine native` to retain upstream loaders and checks.

- R MICE reuses an eigendecomposition only when its numeric matrix, dimensions
  and names are identical within the same native collinearity check. Chains,
  predictor decisions, donor matching, iterations and RNG draws are unchanged.
  For Python MICE, `--baseline-engine native` recomputes the sample Gram for each
  conditional as an uncached reference; it keeps the same Python statistical model.
- PIMMS CF/DAE/VAE cache native loader tensors and use epoch logs instead of
  per-batch console bars. The official LR finder, scheduler, corruption,
  early stopping and requested batches/epochs are retained.
- Iterative Ridge/RF release each fitted regressor after it has filled its
  column. This single-matrix application never transforms a later dataset with
  those old models. Small forest tasks use one thread when starting several
  threads costs more than fitting; actual thread counts are recorded.

These adapters have upstream-version/body guards and fall back to native
execution when unsupported. PIMMS caches are limited to 256 MiB and use no
DataLoader subprocesses. PIMMS seeds Python, NumPy and PyTorch RNGs; older runs
that did not seed Python's loader RNG may differ. Each R run saves the exact
adapter scripts and their hashes alongside its parameters and predictions.

JINGWEI defaults to a lean **PyTorch** runner using the published native DMF/DCAE
networks, the same masked losses and AdamW settings, and the full requested
training budget. DMF reuses row/column projections in its first linear layer;
training gathers batches directly on the selected device. The original
Lightning runner remains available via `--training-engine lightning` or
**Advanced → Training → JINGWEI training engine** for comparisons and replay.
Both paths use float32. Changes in floating-point reduction order can produce
small numerical differences; the engine and execution details are recorded.

R methods exchange explicit little-endian float64 matrices without CSV parsing.
Their original algorithms and parameters are retained. Random forests stay
within the configured CPU thread budget. Inter-method execution remains sequential,
and each method/repeat retains its own timeout and cancellable process tree.
Wide-data iterative methods such as MICE and random forests may still require
substantial time.

`time_s` includes worker setup, method execution and prediction serialization.
Additional fields separate worker setup, backend execution and writing;
JINGWEI also records initialization, training, inference, epochs and update
counts. Compare timing results under the same engine and resource settings.
The [performance notes](PERFORMANCE.md) describe the measurements and checks.

## Data and evaluation

QRILC uses its official feature-by-sample R API. SeqKNN and ImpSeq need at least
two complete protein profiles after masking; unsupported inputs are reported
explicitly. The native `norm` MLE backend is also rejected when its 32-bit
missing-pattern encoding would overflow, even if it could return finite numbers.
These checks preserve the common evaluation mask and do not substitute another
imputation method. `failure_kind=unsupported_input` distinguishes these cases.

- CSV first row/column: automatic character/numeric detection with explicit
  overrides for numeric IDs and numeric headers. Rows default to samples.
- AnnData: `.h5ad`, X or a selected layer; original obs/var are preserved.
- Missing encoding and raw/log2 scale are explicit. Zero/negative values are
  valid observations when missing encoding is `nan`.
- In the current source, one shared evaluation mask E is generated. DMF/DCAE
  receive one additional fixed denoising mask drawn only from the remaining
  observations. Its known values remain reconstruction targets; other methods
  receive the common input without this extra mask. Evaluation truths never reach workers.
  Repeated runs record the seed, masks, parameters and dependency versions.
- **MNAR · JINGWEI** (`mnar-jingwei-v2`) uses within-protein z-scores on the
  working scale and `p = sigmoid(-z + b)`, with the intercept calibrated to the
  selected expected rate. Lower intensities have higher missingness probability.
  The realized rate and probability range are saved. MCAR, custom
  logistic/threshold/quantile MNAR and a historical compatibility kernel remain.
- All five neural models default to **500 epochs without early stopping**.
  No additional internal validation split is made by default; predictions use
  the final epoch checkpoint (`last_model.pt`). The JINGWEI denoising mask is
  constructed once and remains unchanged for all 500 epochs. Its mechanism and
  rate follow the selected GUI/CLI mask settings.
- DMF uses signed bilinear row/column interactions, residual neural towers,
  learned sample/protein biases, and a mask head sharing the row latent rather
  than reconstructed intensities. The interaction rank defaults to 8, capped by
  the latent dimension; random initialization avoids the observed late-epoch
  overfitting of SVD initialization. There is no dataset-specific model dispatch.
- `--training-policy fixed-validation` optionally records a fixed internal
  validation split while still running the full budget; `convergence` retains
  optional early stopping. Neither is the current default. Traditional methods
  retain their own iterations; an epoch budget does not apply to them.
  The per-method timeout remains 30 minutes. CF's entry batch
  size defaults to its native 4096 (`--cf-batch-size`); row models default to
  64 samples (`--batch-size`). Per-method settings can override these values.
- MSE/MAE/RMSE/NRMSE and correlations use evaluation entries only. Undefined
  correlations remain empty. W2 compares multivariate **sample distributions**
  on a common protein subset, never a flattened intensity distribution. W2
  requires originally complete reference samples and reports its scope.
- Result folders retain successful and failed method records, predictions,
  replay masks, manifest, logs, raw metrics, summaries and paired comparisons.
  New GUI jobs have one evaluation; repeat comparisons remain available when opening older multi-run tasks.

These results compare methods under the chosen data/masking configuration;
short smoke runs verify software execution, not model convergence.

## Development and release status

Application source, tests and maintainer-only packaging tools are contained in
this directory. See [packaging/README.md](packaging/README.md). Build/runtime
configuration is not an installation workflow for benchmark users.

The 1.0.7 release uses the fixed-500/DMF update above. Historical tasks retain
their recorded version and protocol.

Windows and WSL are the available verification platforms. macOS CLI packaging
must run on a Mac; a macOS artifact cannot be produced or validated on Windows.
Do not treat the macOS build configuration as an already tested release.
