# jingwei 1.0.5

Windows application: jingwei.exe. Offline installer: jingwei-1.0.5-windows-x86_64-setup.exe.

Run settings contain only Task, Missingness and Mask rate. Device and model
parameters are in Advanced. Protocol, repeats, training-mask and seed controls
and summaries are removed from the desktop form. Chinese and English remain
available; results start empty and saved tasks can be opened manually.

The platform constructs **one shared test mask** and supplies the same corrupted
dataset to all methods. It adds no second training mask or validation split.
Each model retains its own training, validation and denoising procedure.
Imputation uses the supplied dataset directly. CLI training-mask options have
been removed; old saved result files remain readable. Mask protocol 3 records
the single-mask contract; training_hidden is an all-false compatibility array.

Original MNAR preserves get_mask's formula, strict probability bands and Torch
column-major random draws. Its rate shows Auto. All-empty columns are removed
from comparison-method inputs and excluded from a common scoring scope, as in
the notebook. Feature IDs and held-out/evaluated/excluded counts are recorded;
excluded features remain NaN in exported matrices. Historical preprocessing and
adapters remain available through the CLI's original-notebook profile.

Environment checks complete before the main window appears, with startup
cancel/retry and valid-cache reuse. The installer presents the plain-text
LICENSE.txt with unchanged terms. Installation ID, taskbar identity and saved
preference locations are retained for upgrades.
