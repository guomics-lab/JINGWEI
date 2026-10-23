# Software verification

## 1.0.8 installation language and imputation exports — 2026-10-23

Setup records its selected language beside the installed application. The first
launch uses that language even with an older preference in the user profile;
later manual choices persist until the next installation. Both English and
Chinese real-compiler installer fixtures passed. The unnecessary imputation
notice is removed; actual export failures retain a visible failure summary.

Imputation exposes `imputed/<method>.csv` in the CLI output, or
`results/imputed/<method>.csv` inside a GUI task. Open folder opens this
directory directly. H5AD input also produces `<method>.h5ad`. Atomic hard-link
publication avoids duplicating large matrices where supported, with a tested
copy fallback. Detailed per-method exports remain available.

Across focused runs, 66 distinct regression cases passed with no remaining
failures or skips, including GUI workflows, installation language, CSV/H5AD
exports, original identifiers/observations, and explicit export failure handling.
The actual frozen application, with host environment paths removed, completed
all 35 CPU imputation methods and all five CUDA neural methods. Every successful
run produced a finite CSV matching its detailed export. Sparse H5AD CSV/layer
outputs also passed. The default Alzheimer input completed Mean, DMF and DCAE
imputation, with both neural methods running the default 500 epochs; all three
210 × 1542 CSVs retained identifiers and observed values.

The permitted frozen application also passed four isolated hidden native GUI
journeys: English and Chinese installation-language markers, each with an older
opposite-language preference, followed by reopening after a manual language
choice. Initial language, three CSV exports per journey, native taskbar icons,
configuration locks, live logs and warm environment-cache reuse passed. These
were application tests, without executing the blocked complete installer.

All 58 embedded first-party modules match the current source. Fifteen model,
training, masking and metric modules are identical to the 1.0.7 bundle; this
maintenance update does not change scientific algorithms or the public Python
package. Evidence: `results/gui-fixes-20261023/`.

The complete 1.0.8 offline EXE was built successfully (2,147,471,119 bytes,
SHA-256 `5b084509e6282dc99c2c48635397567905db9125c5add7e633de9ec83d2b5c46`).
Its full-installation check is **blocked**, not passed: Windows returned 4551;
Code Integrity events 3033/3077 identify the enforced VerifiedAndReputableDesktop
policy. No security settings or executable contents were changed to bypass it.
The unsigned candidate is distributed as a prerelease, not promoted to stable.
The current GitHub credential also lacks the workflow scope needed to add a new
Windows Actions validation workflow; no remote installation pass is claimed.
Tiny real-installer language checks and frozen-application checks
above are separate from this incomplete full-installation check.

## 1.0.7 complete Windows offline installer — 2026-10-20

The primary Windows download is now one complete offline EXE. The directory
page remains editable with Browse, including upgrades. No adjacent BINs or
installation-time payload downloads are required. The optional online installer
stages downloads on the selected installation volume and cleans them on success.

The actual release EXE installed successfully into a fresh D-drive folder with
Chinese characters and spaces. All 21,330 installed files matched the validated
source by SHA-256. Setup's temporary working files peaked at 4,396,032 bytes;
the previous C-drive download cache, user installation, registry and shortcuts
were unchanged during isolated validation. Nine installer/packaging tests passed,
including missing downloads, failed child installation and corrupt retry cache.

The installed application passed hidden native Windows GUI cold/warm runs in
Chinese and English, CPU/R smoke tests and CUDA DMF/DCAE smoke tests. The warm
launch reused the environment cache with zero probes and became ready in 2.03 s.
All 35 CPU methods and all five CUDA neural methods completed with the compact
runtime. Sparse H5AD input also passed. No host Conda dependencies were available
to these frozen-process checks.

All 57 first-party module code objects and 38 Torch/CUDA library files match the
previous 1.0.7 bundle. Packaging excludes the unused optional Polars backend and
134 R PDF manuals, retaining legal notices and method runtime/data files. Default
500-epoch Alzheimer DMF/DCAE regression tests returned MSE 0.578734 / 0.631094;
the reported scientific results and published Python package are unchanged.

Local evidence: `results/offline-installer-20261020/`. The release includes
`windows-offline-installer.json`, `windows-offline-validation.json` and separate
checksums; previous release assets remain immutable.

## 1.0.7 fixed 500 epochs and revised DMF — 2026-10-20

The full Alzheimer input and common MNAR v2 test mask are unchanged: 255,165
observations, 50,790 artificial test positions (19.9048%) and 204,375 common input
entries. Following the selected training protocol, DMF/DCAE receive one fixed
additional MNAR mask of 41,065 known training positions (20.09297% of available
input). They see 163,310 training input entries but supervise all 204,375 known
targets. The training mask is disjoint from final test positions and never
redrawn. Other methods get the common input without this additional mask.

All five neural methods completed 500 epochs, without early stopping or an extra
internal validation split. Scores use epoch-500 weights, not a selected earlier
test result. All five final checkpoints and the independently recomputed scores
were verified. Known input values are preserved in exports.

| Method | Epochs | Test MSE |
| --- | ---: | ---: |
| Revised DMF | 500 | 0.578734 |
| DCAE | 500 | 0.631094 |
| PIMMS DAE | 500 | 0.639020 |
| PIMMS VAE | 500 | 0.601611 |
| PIMMS CF | 500 | 1.118914 |
| Legacy DMF control, same protocol | 500 | 0.649728 |
| Mean | — | 1.006040 |
| KNN | — | 0.665941 |

The revised architecture reduces MSE 10.93% against the legacy DMF control at the
same 500 epochs and identical masks. It uses signed bilinear interactions, row/
column biases, residual neural towers and a mask head sharing row embeddings.
Rank 8 and small random initialization were selected using an input-only shadow
validation experiment, comparing epoch-500 loss before accessing new test scores.
That diagnostic partition is not used in the final fit or default application.
The architecture is generic, with no Alzheimer-specific dispatch. DCAE was not
redesigned. The published Python package and manuscript were not changed.

This is one dataset/mask, not a universal performance claim. PIMMS CF is worse
at the fixed budget; running 500 epochs does not eliminate overfitting. The older
validation-stopping experiment below has a different training protocol and must
not be substituted for these rows. Runtime measurements include concurrent local
work and are not a controlled speed comparison.

Evidence: `results/dmf-500-20261020/architecture-decision.json`,
`final-verification.json`, `AD-fixed500-summary.csv`, the complete `ad-fixed500/`
run and its same-budget `ad-legacy-fixed500/` control.

## 1.0.6 release checks — 2026-10-20

The final Windows test union covers all 307 collected cases: 305 passed, two
platform-specific cases skipped, no missing cases or remaining failures. This
includes five real installer-fixture checks and a bundled-R regression using
an isolated installation path containing Chinese characters and spaces. The
Windows R subprocess locale is UTF-8; BPCA and R package discovery pass from
that path. WSL focused runs passed 73 cases plus an overlapping 13-case delta.

The frozen Windows runtime passed 35 CPU and five CUDA method dispatches on
execution fixtures, plus hidden native GUI cold/warm workflows. A separate full
Alzheimer run below assesses validation stopping and accuracy; the short
execution fixtures do not establish convergence. Source and frozen predictions
agree within 1.4e-8 MSE. macOS runtime has not been tested.

Evidence: `results/mnar-result-audit-20261020/windows-test-coverage.json`,
`windows-r-fix.xml`, `AD-frozen-convergence-verification.json`, and
`results/mnar-v2-20261020/frozen-validated/validation.json`.

## MNAR v2 and validation stopping — 2026-10-20

This experiment belongs to software 1.0.6 and supersedes no historical task
record. The full Alzheimer input has 210 samples, 1,542 proteins and 255,165
observations. Target MNAR rate 20% hid 50,790 observations (19.9048%), leaving
204,375 common input entries, no empty columns and no platform training mask.
All seven methods received the same input bytes. In the lowest/highest
within-protein abundance quintiles, realized mask rates were 43.9735%/5.2456%.

The five neural adapters independently selected the same 20,438 validation
entries from available input using the recorded model seed. These entries
exclude all test/native missing positions. Training normalization excludes
validation; test intensity targets never reach a worker. Defaults were minimum
30 epochs, patience 30, relative tolerance 1e-4 and a 5,000-epoch safety cap.
Each run stopped on its validation objective and restored its best checkpoint.

| Method | Epochs run | Best epoch | Test MSE |
| --- | ---: | ---: | ---: |
| jingwei-dmf | 57 | 30 | 1.001116 |
| jingwei-dcae | 68 | 38 | 0.576835 |
| pimms-dae | 95 | 65 | 0.605729 |
| pimms-vae | 62 | 32 | 0.656878 |
| pimms-cf | 35 | 5 | 0.676204 |
| mean | — | — | 1.006040 |
| knn | — | — | 0.665941 |

These are one experiment on one fixed mask, with validation-based selection;
they are not a claim of global optimizer convergence or universal superiority.
PIMMS retains its models/losses/LR finder with validation-plateau fitting instead
of a cap-dependent one-cycle schedule. CF uses its native 4,096-entry batch;
row models use 64 samples. Predictions were independently rescored; available
values, validation-mask hashes, best epochs and all five checkpoints verified.

Evidence: `results/mnar-result-audit-20261020/AD-convergence-verification.json`,
`AD-convergence-summary.csv`, and the complete replayable run at
`results/mnar-v2-20261020/alzheimer-convergence/`. Test-based hyperparameter
selection was not performed. Earlier sections below retain their original
version/protocol scope; their numerical rows do not describe MNAR v2.

## Historical 1.0.5 Alzheimer GUI reference — 2026-10-20

The current frozen Windows 1.0.5 GUI job runner completed all ten selected
methods on the full Alzheimer CSV, using the saved GUI configuration and full
50-epoch neural budget. Eight methods update the manuscript main MSE/runtime
tables; mean and PIMMS CF remain additional software results. No notebook
result value is substituted into these exports.

Input SHA256: `1c8aeca1fbd8c2c2814fc4059c257d4f91861c4f7a7deb972558b1e6cff1f116`.
Automatic import retains 210 samples and 1,542 numeric protein columns. Original
MNAR with GUI base seed 0 hides 229,386 of 255,165 observations, leaving 25,779
available values. All methods receive the same input file. No platform training
mask is applied. Sixteen empty columns are excluded from scoring, leaving
229,311 common test entries and 75 excluded holdouts.

Independent checks confirm identical method-input paths, absence of test truth
from that input, preservation of measured values, finite predictions on every
retained test target, and MSE/MAE/RMSE recalculated from saved predictions.
The GUI-loaded `metrics.json` matches the CSV and independent scores. Undefined
sample W2 stays empty; no zero-filled reference is invented.

| Method | MSE | GUI worker time (s) |
| --- | ---: | ---: |
| jingwei-dmf | 0.681597 | 4.499 |
| jingwei-dcae | 0.704898 | 7.323 |
| pimms-dae | 0.677432 | 9.470 |
| pimms-vae | 0.614723 | 25.608 |
| pimms-cf | 0.654669 | 76.987 |
| feature-minimum | 2.769302 | 0.650 |
| mean | 0.638747 | 0.467 |
| knn | 0.623089 | 2.001 |
| mice-pmm | 0.674742 | 12.820 |
| missforest | 0.589278 | 377.880 |

This is one evaluation on one shared mask, with no repeat SD. The saved GUI
settings select original-repository MNAR and the standard neural parameters;
they are not an assertion that every new installation initially selects MNAR.
DMF/DCAE were trained for 50 epochs, without the superseded AD inner-validation
selection/refit protocol. Other datasets were not rerun. The manuscript AD main
MSE/timing rows are updated, its old AD supplementary ablation/order records are
removed, and other dataset numerical cells are preserved.

Local raw evidence is in `results/gui-ad-reference-20261020/`: GUI configuration,
complete reopenable task, shared input/masks, per-method logs, predictions,
versions, full metrics, independent verification and manuscript change ledger.
The obsolete notebook output directories and their numerical comparison guides
have been removed. Compatibility implementation code is not a result source.

## Application and installer regression — earlier 2026-10-20 run

These source/build checks preceded this manuscript-only update; they were not
rerun as new accuracy experiments. Raw evidence remains in
`results/release-v1.0.5/final-verification.json`.

| Check | Result |
| --- | --- |
| Windows application regression | 288 passed, 2 POSIX-only tests skipped |
| WSL/Linux import, CLI and masking | 34 passed |
| WSL/POSIX runtime | 2 passed |
| All 35 method dispatches | Same input path/hash, no platform training mask |
| Frozen source snapshot | 55 application modules match source code objects |
| Standalone CPU/CUDA fixtures | 5 CPU and 5 CUDA methods completed |
| Installed payload | 21,466 files match the tested distribution |
| Installed GUI workflows | Cold/reopen, Chinese/English, live logs, custom output and manual task opening passed |
| Warm startup | Cached environment, zero new probes |

Earlier two-epoch full-data checks verified execution, not model convergence.
The current AD reference above uses the full selected GUI budgets. macOS has
packaging/policy checks but no binary/runtime verification in this environment.
No published Python package, user preferences or installed application was
modified by the current manuscript work.

## Additional failure-path regression — 2026-10-20

The follow-up software audit passed 250 distinct Windows application cases plus
67 public API checks. WSL/Linux passed 276 CLI/API checks, with six platform-specific
skips. The broad Windows application run passed 243 cases with two POSIX-only skips;
overlapping focused runs (73, 65 and 6 passed) covered subsequent boundary fixes.
Public package code was not changed.

The existing frozen Windows EXE completed all 35 methods on a small CPU fixture,
all five neural methods on CUDA, and seven selected methods on the full Alzheimer
original-MNAR input. Cold/warm native GUI startup, icons, environment cache,
Chinese/English, live logs, custom output and manual task opening passed. The saved
ten-method manuscript reference task reopened with unchanged scores and elapsed times.

Source fixes reject malformed saved task JSON/results/metrics, handle AnnData files
with no X matrix, validate saved method overrides, safely restore oversized numeric
preferences and use the single-run default for task progress. Windows/POSIX/release
builders no longer copy the removed notebook guide. Obsolete tests for the removed
AD experimental script were retired; current application checks remain.

SeqKNN and ImpSeq still require complete feature profiles unavailable after masking
the full Alzheimer input. MLE (norm) cannot safely encode its missingness patterns
across 210 sample columns. Those cases return explicit `unsupported_input` failures;
the common input/mask is preserved and no score is fabricated. QRILC and Python
MICE-PMM completed in the same full-input check. The small-fixture all-method test
verifies execution, not convergence or full-AD accuracy.

Evidence: `results/se-followup-20261020/REPORT.md`, `summary.json`, pytest XML/logs
and `frozen/validation.json`. These fixes are source-only: the installed application
and installer have not been rebuilt or replaced. macOS runtime was not tested.

## GUI review comments — 2026-10-20

Configuration controls now lock from submission through execution/cancellation and
unlock after completion, cancellation or submission failure. Stop, results and logs
remain usable; advanced settings can be inspected with their controls disabled.
Reuse settings restores common settings, selected methods, GUI method parameters
recorded in worker jobs, W2 features and the output folder. A cloned task was rerun
after deleting its original CLI override file, confirming its custom KNN setting.
The Open task button is removed in both languages. Blank task names use
`task-YYYYMMDD-HHMMSS`; output folders include a unique suffix and retain that name
when reopened. Explicit names and input-file provenance remain intact.

The final combined GUI/job regression passed 19 cases. Together with the other
reviewed GUI regressions, 43 distinct Windows cases passed after fixes. WSL passed
all ten job lifecycle/naming cases. Isolated hidden native Windows cold/warm runs
completed mean, KNN and DCAE; parameter locks, automatic unlock, icons, languages,
live logs and cache reuse passed. Evidence and UI captures are under
`results/gui-comments-20261020/`. This work changes source GUI behavior; the
existing installer has not been rebuilt.

## Imputation training and application naming — 2026-10-20

The application and current CLI use the name `jingwei`. In Impute, the GUI's
missingness/rate settings construct a training mask for DMF/DCAE only. Artificially
hidden observations remain supervised targets; inference uses the original input
and preserves known observations in exports. Other methods receive the unchanged
input and no JINGWEI training mask. Benchmark still constructs one shared test mask
with no additional platform training mask; its Alzheimer reference is unchanged.

After fixes, 49 distinct Windows targeted cases passed across the final runs;
WSL passed all 43 related CLI, training, packaging and runtime cases. MCAR,
logistic MNAR and original-repository MNAR dispatch were checked. Real Torch and
Lightning DMF/DCAE training completed with verified mask/input/target counts and
finite imputation outputs. A GUI imputation task completed mean, DMF and DCAE;
training settings remained editable before submission, locked during execution,
and survived copying the completed configuration. Both GUI languages passed.

Evidence: `results/gui-release-1.0.5-20261020/`, including merged final pytest
results, GUI workflow exports, build logs and packaged runtime checks.
