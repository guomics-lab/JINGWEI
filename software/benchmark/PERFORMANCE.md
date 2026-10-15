# Performance and equivalence checks

## Python linear MICE-PMM

The generic `mice` / `mice-pmm` entry defaults to a custom NumPy/SciPy linear
MICE implementation; this is not MICEforest. `--mice-backend r` retains original
R PMM. NAguideR MICE-Norm/CART remain R methods. Old job JSON without a MICE
backend continues to use R so an in-flight batch or historical replay cannot
silently change its implementation.

The conditional model standardizes predictors once from visible observations,
uses a flat intercept, beta|sigma² ~ N(0, sigma²/lambda I), and scale prior
1/sigma². Lambda is `ridge * n_observed`, with `ridge=1e-5`. After centering on
observed target rows, B=Xc Xcᵀ and A=B+lambda I. The variance draw has scale
lambda ycᵀ A⁻¹ yc and chi-square degrees of freedom `n_observed-1`.
This is a conjugate regularized linear MICE variant, **not** a bitwise port of
R mice's predictor removal and residual-variance formula.

Type-1 PMM matches observed conditional means to missing-row conditional means
with posterior parameter uncertainty. It does not add independent response
residual noise to the matching predictions. The five nearest observed donors
are used, with random tie ordering and donor selection. Original observed values
are never replaced. A constant/single-observation target has only one distinct
admissible donor value; that deterministic result is recorded. Entirely missing
features have no donor distribution and are rejected.

The solver caches the sample Gram Z Zᵀ and applies rank-two updates when a
column changes, rebuilding it once per sweep. Cholesky operates in sample space
instead of forming a protein-by-protein covariance. Every other predictor,
all 5 imputations and all 5 sweeps are retained. The uncached `native` execution
path recomputes the Gram per conditional using the same RNG draws. It is a
reference for this Python model, not the selector for R PMM.

The full 210×1542 default corrupted matrix completed in **36.8 seconds** in
the direct kernel check (4 CPU threads, all 5×5 updates); the public source CLI
worker completed in **39.6 seconds**, evaluating 51,031 held-out entries.
The evaluation/training masks and worker input were bit-identical to the earlier
R diagnostic. All outputs were finite and observed values remained identical.
This is a successful completion measurement, not a speed ratio against the
earlier R run, which failed and used a different conditional implementation.
Timings are descriptive measurements on a shared workstation.

Tests independently check predictions/variance against a dense feature-space
posterior, sampled covariance without response noise, cached/reference sweeps,
donor support, m/iteration counts, saved draws/pointwise pooling, fixed-seed
reproducibility, collinearity, p>n and input failures. All five raw draws,
normalization scales, observed counts, chain seeds and iteration traces are saved.
Evidence: `results/python-linear-mice-full-default-v2/` and
`results/python-linear-mice-cli-full/`.

## Comparison-method execution

`--baseline-engine optimized` is the default for comparison methods. The
Windows GUI exposes the same choice under Advanced → Runtime. `native` keeps
upstream loading/checking behavior for comparison. Model budgets and the
evaluation protocol are identical between these execution paths.

- R MICE: reuse an eigendecomposition only for an **identical** correlation
  matrix within one native collinearity check. Values, dimensions and names
  must match. Explicit symmetric decomposition also avoids redundant symmetry
  checks. All native predictor removal rules, chains, draws and RNG calls stay
  intact. The entire upstream `remove.lindep` body is checked before applying
  the temporary in-memory change; unsupported versions run natively. The
  namespace binding is restored even on exceptions. Remove/review this adapter
  when upstream changes the implementation or eliminates the repeated work.
- PIMMS CF/DAE/VAE: convert native, already-preprocessed loader data to tensors
  once. Preserve the actual upstream loader class, including CF's weighted
  sampling, batch size, shuffle/RNG state, LR finder and callbacks that affect
  fitting. Epoch logging replaces console progress bars. Native plots remain.
  Caches are scoped to one fit, capped at 256 MiB, and use zero loader workers.
  Unsupported dependency versions fall back to native data loaders.
- Iterative Ridge/RF: discard fitted regressors after each column is filled.
  The benchmark only imputes the fitting matrix; it does not reuse an imputer
  on later data. The native fitting loop still fits every regressor, tree and
  iteration. A guarded private single-matrix adapter refuses later `transform`
  calls. Very small forest fits can use one thread within the requested budget.

PIMMS now seeds Python's RNG as well as NumPy and PyTorch; historical runs that
did not seed Python's loader RNG may differ. R methods save their adapter
scripts and SHA256 hashes per run. This protects long-running R jobs from
application/source updates and makes the exact adapter reviewable.

CPU differential tests cover complete/duplicate/constant/all-missing features,
wide and narrow MICE inputs, multiple PIMMS seeds, final model weights,
predictions, RNG state, update counts, namespace restoration/fallback, and
release of obsolete fitted forests. MICE and PIMMS comparisons were bit-exact
on these fixtures. RF's serial/parallel accumulation can differ in low-order
floating-point digits; the measured maximum difference was 5.4e-15.

Local Windows measurements use alternating native/optimized/optimized/native
fresh-process trials with two completed observations per path. MICE-PMM and
Iterative RF use the 16×80 integration matrix, **full native default** MICE
5 chains × 5 iterations and RF 100 trees / maximum 10 iterations. RF converged
after 6 iterations (480 feature fits) in both paths; retained regressors fell
from 480 to zero. These small-matrix timings are not full-Alzheimer timings.
PIMMS uses the complete 210×1542 Alzheimer matrix, 50 maximum epochs, batch
size 64, latent dimension 64, hidden dimensions 256/128, four CPU threads and
RTX 4080 CUDA. AE models trained all 50 epochs / 200 updates; CF's native
early stopping stopped at 4 epochs / 10,204 updates in both paths.

| Method | Dataset | Native worker (s) | Optimized worker (s) | Native fit (s) | Optimized fit (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| MICE-PMM | 16×80 | 83.37 | 14.37 | — | — |
| Iterative RF | 16×80 | 48.77 | 39.35 | — | — |
| PIMMS CF | Alzheimer | 78.12 | 66.16 | 52.67 | 42.14 |
| PIMMS DAE | Alzheimer | 27.86 | 24.46 | 6.14 | 3.48 |
| PIMMS VAE | Alzheimer | 25.43 | 29.18 | 6.06 | 5.21 |

Values are medians. Fit includes the original LR finder and plotting. Worker
time also includes imports, backend setup, inference and prediction writing.
VAE's total latency increased despite faster fitting; startup variation matters.
These are descriptive measurements on a shared workstation, with concurrent
validation/build activity, not controlled statistical estimates or universal
speedups. Interrupted trials were retained separately and rerun to completion.
BPCA's full-data worker medians were 8.15 / 8.14 seconds: no speedup is claimed.
An exploratory full-data MissForest serial run was slower (149 → 237 seconds);
the final heuristic therefore retains the thread budget on the full dataset.

Evidence is under `results/other-method-speed/`, with parameters, predictions,
phase timings and per-run R adapter snapshots; automated checks are in
`test_pimms_engine.py`, `test_iterative_engine.py` and `test_r_acceleration.py`.
Regression checks on 2026-10-15: Windows 107 passed / 2 skipped; WSL/Linux
103 passed / 6 skipped. Small fixture checks establish execution/equivalence,
not successful completion of the full-data, all-method, five-repeat default run.

## Default-data integration findings

The full default CLI selects all 34 methods and 5 repeats. A real-data check
found a QRILC adapter error that the smaller integration matrix did not expose:
its official R function takes feature rows and sample columns. The adapter now
passes that orientation and transposes its result back. A direct comparison to
the official API is part of `test_r_exchange.py`. Replaying the exact original
210×1542 corrupted matrix completed in 1.82 seconds after the correction.

SeqKNN and ImpSeq require complete feature profiles to initialize their native
sequential algorithms; no such profiles remain in this default evaluation mask.
Both now reject the input explicitly rather than failing with an index error or
nonfinite output. SeqKNN's native reduction of k to the available neighbor pool
is retained; this is not a requirement for k complete profiles.

The native `norm::prelim.norm` encodes missingness patterns as signed 32-bit
integers. This dataset overflows those codes and can merge distinct patterns
while still returning finite numbers. The MLE adapter now rejects unsafe
patterns and emits no scores. Earlier diagnostic MLE scores are invalid and
must not be used. These cases have `failure_kind=unsupported_input` and leave
the common evaluation/training masks unchanged. No alternate imputation or
feature/sample filtering is substituted. The initial Iterative Ridge/RF runs
also reached the then-default 600-second timeout on this wide matrix;
MICE-CART subsequently reached it as well. New tasks now default to **1800
seconds (30 minutes) per method/repeat**, in both CLI and Windows GUI. Existing
jobs retain the timeout recorded at launch; the original 600-second run remains
a separate diagnostic record. Longer-timeout replays retain the same masks,
seeds, algorithm budgets and thread limits.

Initial diagnostic output is retained under
`results/default-all-methods-windows-20261015-full/` and marked diagnostic only.
The corrected strict-default run is separate under
`results/default-all-methods-windows-20261015-corrected/`; the live
`default-cli-audit.json` distinguishes progress from a completed 170-run test.

## JINGWEI execution (earlier optimization)

The application uses the published JINGWEI architectures without changing its
Python package or model definitions. The optimized execution path lives in
`src/jingwei_benchmark/torch_engine.py`; the original Imputer/Lightning path is
retained in `lightning_engine.py`.

## What changes

- Lightning's training orchestration and scalar metric collection are replaced
  by an isolated PyTorch loop for DMF/DCAE. AdamW, learning rate, weight decay,
  batch size, epochs, initialization, shuffle order and supervised observations
  retain the native semantics.
- The DMF projection `Linear(concat(row, column))` is computed as two projections
  and a broadcast sum. All original parameters, activations and dropout remain.
  This avoids repeating the first projection for every sample/protein pair.
- Complete device-resident tensors are gathered by batch indices instead of
  collating individual GPU rows. Inference is batched; DCAE does not calculate
  unused auxiliary mask logits during inference.
- R's original methods receive the same numeric matrix through a validated
  float64 binary bridge. CSV compatibility is retained internally for testing.
- Forests respect `--threads`; scratch NPZ prediction/input files omit ZIP
  compression. Replay masks remain compressed. Exported job ZIPs still compress
  the entire result set.

No training epochs, optimizer steps, trees, imputations, predictors or repeats
are automatically removed. No mixed precision, approximate W2, worker pooling
or concurrent method execution is introduced.

## Scientific checks

`tests/test_torch_engine.py` compares against the native package:

- DMF forward outputs, gradients and unchanged parameter names, including
  dropout, repeated sample indices and several hidden-layer configurations.
- Masked loss, gradients and an AdamW step for both models, with zero/nonzero
  auxiliary weights and empty supervision.
- Identical initialized parameters and shuffled batches across multiple fixed
  seeds, equal update counts, and final CPU imputations within float32 tolerance.
- Repeatable results, preservation of observed entries and explicit accounting
  of second-mask input and supervision counts.

`tests/test_r_exchange.py` compares CSV and binary paths for BPCA, MinProb,
MissForest and MICE-PMM. Existing mask, metric, timeout, cancellation, persistent
history, GUI/CLI parity and export checks are retained.

Float32 reassociation and nondeterministic CUDA kernels can change low-order
digits and optimization trajectories. New and old engine outputs are not
promised to be bit-identical. Record the engine and use the compatibility path
when reproducing an older run.

## Measurement protocol

The local measurement uses the full `data/Alzheimer.csv`: 210 samples, 1542
proteins, approximately 21.2% native missingness; raw abundance is transformed
to log2. DMF and DCAE use 50 epochs, batch size 64, latent dimension 64, hidden
dimensions 256/128, CPU thread budget 4 and an NVIDIA RTX 4080 via PyTorch
2.7.1+cu128. Each model performs 200 optimizer updates. Shared evaluation and
second training masks, seeds and model settings are fixed across engines.

Fresh-process trials alternate Lightning / PyTorch / PyTorch / Lightning to
expose startup/cache variability. `time_s` is worker setup plus execution plus
prediction writing; `training_s` isolates fitting and synchronizes CUDA around
the timed region. Total CLI wall time additionally includes parent startup,
inspection, mask generation, evaluation, CSV output and reporting.

This is an execution-speed check on one machine and dataset, not a model
convergence study or an assertion of universal speedups. Startup times can
dominate short runs. Other R/PIMMS algorithms keep their native training logic;
no equivalent speedup is claimed for them from this JINGWEI-only trial. The
comparison-method measurements above are separate.

Local two-trial medians per engine (seconds, 2026-10-15):

| Model | Lightning training | PyTorch training | Training speedup |
| --- | ---: | ---: | ---: |
| DMF | 2.349 | 1.033 | 2.27× |
| DCAE | 4.848 | 3.227 | 1.50× |

Median CLI wall time for the two models together was 56.62 seconds with
Lightning and 46.46 seconds with PyTorch (about 18% lower). Worker startup was
variable: DMF's median per-worker total was 12.47 → 17.52 seconds, while DCAE's
was 25.34 → 10.54 seconds. Thus fitting is faster, but an individual run's total
latency can still increase. These trials ran on a shared workstation, with two
observations per engine; they are descriptive measurements, not controlled
statistical estimates.

The local evidence is saved under `results/speed-balanced-alzheimer/`: raw
per-run metrics, phase timings, complete configurations, identical replay masks
and input hashes. Windows regression tests: 82 passed, 2 skipped. WSL/Linux:
78 passed, 6 skipped. All 34 methods also completed a small CPU integration
fixture, and both optimized native methods completed actual CUDA runs on
Windows and WSL. Small fixtures and reduced smoke-test iteration settings
verify execution and integration, not comparative method quality.
