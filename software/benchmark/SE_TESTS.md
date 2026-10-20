# Software verification

## Current Alzheimer GUI reference — 2026-10-20

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
