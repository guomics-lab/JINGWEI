# Historical notebook reference

These Python snapshots come from JINGWEI commit
`98686b54fdb9918a4d4cc9d99ac4880a6dadcefe` (before notebook removal).
`datasets.py` is `src/datasets.py`; `models.py` is `src/jingwei/models.py`;
the remaining modules are the corresponding `src/methods` scripts. Whitespace
is normalized without changing the algorithms. The two
Lightning imports of `jingwei.models` are redirected to `.models` so this
application cannot import or modify the separately published Python package.
No notebooks are restored.

Only the explicit `original-notebook` protocol uses these snapshots. Historical
first-column dropping, pre-mask statistics, mixed-scale thresholds, strict
inequalities, shuffled DMF row behavior, full-mask BCE and score filtering are
preserved. The wrapper uses one persistent DataLoader subprocess (rather than
twenty, preserving parent RNG and shuffle behavior), quiet logging and the
selected supported device. Original PIMMS DAE uses CPU; VAE uses the requested
device. Results still depend on library versions, hardware and the random stream
consumed by earlier notebook cells/methods. Archived values are comparison
targets, never hard-coded predictions or fabricated scores.
