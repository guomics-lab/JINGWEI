# V1.0.2 software verification

Validation date: 2026-10-16. Windows 11 and Ubuntu under WSL2, using the dedicated
`jingwei_benchmark` conda build environments. The released applications bundle
their runtimes; end users do not need those environments.

| Check | Result |
| --- | --- |
| Windows application regression suite | 166 passed, 2 skipped |
| WSL/Linux application regression suite | 160 passed, 8 skipped |
| Windows public standalone CLI, host runtime paths removed | 34/34 CPU methods completed |
| Windows desktop's isolated worker path | 34/34 CPU and 5/5 CUDA methods completed |
| Linux standalone CLI, outside the checkout, host runtime paths removed | 34/34 CPU and 5/5 CUDA methods completed |
| Windows packaged GUI components, cold launch and reopen | Both languages; live logs; chosen output; empty initial results; warm environment cache |
| Windows executable source snapshot | All 36 application modules match current source |
| Windows offline installer upgrade | V1.0.1 → V1.0.2; installed executable matches build; application closing disabled |
| Windows uninstallation | Owned test install/registration removed; prior shortcuts restored; external task results preserved |
| Linux split-archive installer | Checksums, fresh runtime, default input, command symlink and duplicate-install protection passed |

User journeys cover CSV header/ID detection (including zero-padded numeric IDs),
blank lines/BOM, invalid inputs and resource limits, sparse AnnData layers,
benchmark and imputation, method selection/configuration, two disjoint shared
masks, four missingness mechanisms, repeats and deterministic replay, all metric
plots, PNG/PDF/SVG exports and provenance, complete task ZIP exports, cancellation,
close confirmation, manual opening of GUI/CLI results, moved/deleted folders,
read-only opening and preference/environment-cache persistence. Stale asynchronous
preview/probe results and transient Windows JSON replacement locks are covered.

Method integration fixtures use 16 samples × 80 proteins and reduced neural
budgets (one epoch, small dimensions). They verify software execution, device
selection, masking and output integrity, not convergence or a scientific ranking.
Full Alzheimer input is separately checked with QRILC/Python MICE and the explicit
unsupported-input outcomes of SeqKNN, ImpSeq and MLE. Failed/unsupported attempts
remain visible and are excluded from paired complete-repeat comparisons.

PyTorch/Lightning outputs, gradients and optimizer steps retain their parity
checks. R/native execution and accelerated loaders retain independent comparisons.
MSE is checked against RMSE² on the common evaluation mask. Original model/training
presets are checked without changing evaluation or training masks.

To avoid interfering with an active user window, the latest packaged GUI journey
ran through Qt's offscreen platform. Native Windows desktop/taskbar interaction
was not repeated in this session. macOS has platform-policy tests but no runtime
or binary validation. No system application-control settings were changed.

The first Linux release test exhausted the test machine's 32 GB `/tmp` memory
filesystem while retaining multiple archives. Owned test copies were removed,
tests moved to persistent storage and the complete suite/standalone checks rerun.
Release parts are generated only from the finished archive and verified by the
offline installer. Raw test logs, JUnit XML, configurations, masks and diagnostics
remain in the local ignored `results/se-qa/` directory.
