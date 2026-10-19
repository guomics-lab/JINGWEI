# JINGWEI Benchmark

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

Download and run the [Windows installer (.exe)](https://github.com/guomics-lab/JINGWEI/releases/download/v1.0.3/JINGWEI-Benchmark-Windows-Setup.exe).
It automatically downloads and verifies the complete application. Install once,
then open **Start → JINGWEI Benchmark**. No manual BIN downloads are required.
See the [installation guide](RELEASE-1.0.3.md).
The GUI uses the original JINGWEI bird for the
application, window and Windows taskbar.

The compact workbench keeps data, methods and run settings on the left, with
**Results / Plot / Log** tabs on the right. JINGWEI DMF and DCAE are selected by
default. The bundled `data/Alzheimer.csv` is loaded on startup; **Example**
returns to this dataset. Choose CSV/h5ad data, confirm its input scale, select methods, masking,
a protocol and device, then click **Run**. GUI tasks run once; there is no repeat control. Run and Stop stay visible.

Use **中文 / English** at the top right to switch languages immediately; the
choice is remembered and switching keeps the current configuration and job.
**Preview** shows the detected header/ID and a small matrix preview. Numeric IDs
or numeric headers can be specified in **Import settings**. **Advanced** contains
shared training settings, the second training mask, MNAR controls and W2 protein
selection. **Runtime** shows method availability and optional backend paths.
Each method's **Parameters** button overrides shared training settings.

Choose an **Output folder** before running. Each run creates a separate named
task folder containing its input snapshot, configuration, logs and results.
**Open task…** opens that folder later, including a moved GUI task folder or a
CLI output folder. Opening saved results does not rerun algorithms. Startup
leaves the result area empty; there is no history list or automatic last-task loading.
Cancellation, live logs, metrics, paired comparison plots and complete result
ZIP export are available. The Log tab opens when a run starts and shows task
progress plus the selected method's output. Plot export includes source CSV
and provenance JSON. Method IDs and original algorithm logs retain their native
names across languages.

The desktop also restores the last data file, method selection, method-specific
parameters and window layout. It caches the first R/PyTorch environment check;
reopening does not repeat that probe unless the software, backend paths,
dependencies or GPU driver change. **Environment → Refresh** checks again.
Data is still checked when loaded, and workers validate the actual runtime.
Form preferences and the output folder are remembered independently of results.

### Original notebook and fair protocols

**Protocol → Original notebook replay** restores the full historical pipeline,
including unconditional first-column removal, log2 positives, pre-mask float32
statistics, the original `get_mask()`, one mask, original adapters and score
filtering. This is the default for new forms; existing settings retain their
protocol. Fixed historical settings are disabled in the form. **Fair benchmark**
retains automatic CSV/AnnData import and sequential evaluation/training masking.

The [replay guide](NOTEBOOK_REPLAY.md) documents exact rules, method order, random
streams, full Alzheimer commands and the distinction from the parameter-only
`original-benchmark` preset. Historical replay's MICE is the original Python
linear method and its MissForest is the Python script. These are distinct from
fair mode's PMM/R implementations.

MSE, RMSE, MAE, NRMSE, correlations and sample W2 are exported with their scope.
The old notebook only reported MSE and runtime. Full Alzheimer mask counts and
deterministic scores match its saved output; not all archived neural scores
are reproduced. Direct execution of the old functions on the same machine is
also checked. See [verification](SE_TESTS.md). Never interpret this compatibility
profile as an exact paper-table reproduction.


## Linux and macOS

For Linux V1.0.3, download all parts, its checksums and `install-linux.sh` from
the [release page](https://github.com/guomics-lab/JINGWEI/releases/tag/v1.0.3).
Run `sh install-linux.sh`; the command is placed in `~/.local/bin`.
Custom installation locations are described in the [installation guide](RELEASE-1.0.3.md).
No macOS binary is included in this release. From the installed application folder:

```bash
./jingwei-benchmark --help
./jingwei-benchmark methods
./jingwei-benchmark inspect --input data.csv
./jingwei-benchmark benchmark --input data.csv \
  --methods jingwei-dmf jingwei-dcae mean knn bpca missforest \
  --mechanism mnar-jingwei --mask-rate 0.2 --training-mask-rate 0.2 \
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
- In the fair profile, shared evaluation mask E is generated first. Training mask T is generated
  only from the remaining observations. Evaluation truths never reach workers.
  Repeated runs record the seed, masks, parameters and dependency versions.
- Fair mechanisms: MCAR, custom logistic/threshold/quantile MNAR, and the historical JINGWEI threshold kernel. The complete notebook profile is separate.
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

Windows and WSL are the available verification platforms. macOS CLI packaging
must run on a Mac; a macOS artifact cannot be produced or validated on Windows.
Do not treat the macOS build configuration as an already tested release.
