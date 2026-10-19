# V1.0.3 software verification

Validation date: 2026-10-19. Windows 11 and Ubuntu/WSL2, in separate
`jingwei_benchmark` conda build environments. End users receive bundled runtimes.

## Regression and packaged execution

| Check | Result |
| --- | --- |
| Windows full application regression suite | 184 passed, 2 skipped |
| WSL/Linux full application regression suite | 178 passed, 8 skipped |
| Latest focused notebook, session and GUI checks | 26 passed |
| Frozen Windows source snapshot | All 52 application modules match current source |
| Windows public standalone console, host runtime paths removed | 35/35 CPU methods completed |
| Packaged Windows GUI job runner, sequential clean-host-path retry | 35/35 CPU and 5/5 CUDA; sparse AnnData; mask variants; ZIP export passed |
| Linux offline CLI outside checkout, host paths removed, cache path with spaces | 35/35 CPU and 5/5 CUDA methods completed; no Qt |
| Packaged Windows Qt offscreen cold/reopen workflow | Both languages; single task; live logs; custom output; empty initial results |
| Packaged Windows warm reopen | 1.234 seconds to ready; cached environment, zero probe processes |
| Windows frozen full Alzheimer replay, eight-method subset | All completed; identical mask/scoring counts; all six matching archived MSE values retained |

The focused checks include an additional environment-cache case after the full
suite: switching protocols changes Python/R method availability without another
heavy probe. GUI repeat controls are absent, old repeat preferences cannot change
new tasks, and legacy multi-run result files remain readable. CSV/AnnData import,
mask separation, metric formulas, logging, cancellation, exports, moved tasks and
preference/cache persistence retain their regression coverage.

Initial simultaneous first-launch checks encountered Python-DLL load failures
in several processes, while the public console completed all 35 methods. Both
executables subsequently started normally; sequential GUI/backend checks are
retained separately. The DLL matches the build environment's SHA256. The cause
of the initial load failures is not established. No system application-control
settings were changed. PIMMS uses the same-version official pure-Python
`plum-dispatch` wheel on Windows after a compiled dependency load was blocked.

## Complete Alzheimer notebook replay

The comparison uses the raw bundled Alzheimer CSV, seed 0, original preprocessing,
model/baseline defaults and single mask. The historical source is pinned to commit
`98686b54fdb9918a4d4cc9d99ac4880a6dadcefe`. All nine original methods completed on
the full input; MissForest required 2808.9 seconds (46.8 minutes), so this diagnostic
explicitly allowed 5400 seconds. The application default remains 1800 seconds.

| Mask property | Archived output | Current replay |
| --- | ---: | ---: |
| Samples × proteins | 210 × 1541 | 210 × 1541 |
| Initial observed entries | 254968 | 254968 |
| Dynamic holdouts | 229546 | 229546 |
| Remaining observations | 25422 | 25422 |
| Empty columns after masking | 20 | 20 |
| Scored entries after empty-column exclusion | 229401 | 229401 |
| Second training mask | None | None |

| Method | Archived MSE, printed precision | Current MSE | Printed value reproduced |
| --- | ---: | ---: | --- |
| MissForest (original Python) | 0.607 | 0.601493 | No |
| PIMMS VAE | 0.613 | 0.612827 | Yes |
| KNN | 0.674 | 0.673630 | Yes |
| NMF | 250.693 | 250.693451 | Yes |
| PIMMS DAE | 0.640 | 0.634120 | No |
| Per-protein MIN | 2.738 | 2.737587 | Yes |
| MICE (original linear Python) | 0.816 | 0.815853 | Yes |
| DMF | 0.595 | 0.595042 | Yes |
| DCAE | 1.200 | 1.416695 | No |

**Six of nine archived MSE values match at the saved precision. All historical
scores have not been reproduced.** Parameters and scoring are retained; the
three differences are visible, not replaced with archived numbers.

For an independent control, the historical notebook's actual function definitions
were extracted from Git and executed on this same machine, with the same selected
eight-method subset (the separate full MissForest run is above). Portability
changes close the temporary CSV before writing on Windows and use two persistent
workers instead of twenty. This direct execution gives VAE 0.612827, DAE 0.634120,
DMF 0.594809 and DCAE 1.416685. The application subset gives 0.612827, 0.634120,
0.595042 and 1.416681, respectively. CPU RNG state before both native model
constructors matches the independent reference byte for byte. Unit tests also
check identical original losses, gradients and predictions when evaluation
targets are withheld from the worker.

These controls verify the restored source workflow and identify unresolved
historical-run differences. The archived notebook does not provide a complete
runtime/weight/random-state snapshot, so a claim that every saved neural score
has been recovered would be unsupported. The old notebook only reported MSE
and runtime; additional MAE/RMSE/NRMSE/correlation/W2 values have no archived
counterpart. Timings depend on hardware and process boundaries. Sample W2 is
undefined on all proteins here because no original sample is complete; it stays
empty unless a suitable common reference feature subset is selected.

## Scope

The fair protocol retains evaluation E followed by training T, with no evaluation
truth in fitting statistics or workers. The historical profile intentionally
retains first-column dropping, pre-mask statistics, mixed-scale thresholds,
strict boundary holes, full-mask BCE, shuffled DMF behavior and old score
filtering. Both profiles save masks, settings, versions and scoring counts.

The frozen eight-method replay ran outside the checkout with host Python/R/conda
paths removed, using the complete Alzheimer matrix and original training budgets.
MissForest was checked in the separate full nine-method source run above; its
46.8-minute run was not repeated for the frozen distribution. The packaged replay
retains the DAE/DCAE differences rather than substituting historical values.

Packaged method integration fixtures use reduced neural budgets to check
execution rather than convergence. Full Alzheimer source replay uses the actual
50 native / 200 requested PIMMS epochs. Windows GUI tests are offscreen to avoid
interfering with an active user window; physical desktop/taskbar interactions
and installer registration are not repeated in this session. macOS has platform
policy checks but no runtime/binary validation. The published Python package
and the user's existing application/window are unchanged by these checks.

Raw JUnit files, configurations, complete method results, old-function controls,
logs and screenshots remain in the ignored local `results/se-qa/` directory.

## Development global AE checks

On 2026-10-19 the optional `global` DCAE architecture was added to the development
source. Windows integration regression: **203 passed, 2 skipped**. WSL/Linux:
**196 passed, 9 skipped**. The latest focused global-AE checks: **19 passed**,
including CPU/CUDA inference agreement and the GUI architecture/reset/result
label changes made after the full suites started.

Coverage includes native initial-weight/output equivalence, true-zero versus
missing-zero semantics, masked-value/gradient isolation, short/odd/wide shapes,
capacity validation, CPU runs through PyTorch and Lightning, observed-value
preservation, actual CLI worker dispatch, rejection of new models in historical
replay, and Windows GUI selection/reset. GUI checks remain offscreen.

The [architecture report](AE_ARCHITECTURE.md) records the paired Alzheimer
experiment, inner-validation selection, training-only normalization, three
initialization/validation seeds, protein-mean baseline and limitations. Its
mean MSE is 0.691538 for native versus 0.540648 for global on the common outer
mask; new results are independent of historical-table reproduction. This source
feature is not part of the already released V1.0.3 binaries.

Separate final checks ran both training engines on actual CUDA, with six optimizer
updates each, and independently loaded all three saved global models on CPU.
Their predictions match the stored GPU predictions with maximum absolute
differences below 0.000003. Chinese and English parameter dialogs were rendered
offscreen with an explicitly loaded Windows font and visually inspected.
