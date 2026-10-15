# Application-local JINGWEI adapters

These DMF/DCAE architectures and Lightning wrappers are a fixed copy of the
JINGWEI implementation used by this benchmark application. The wrappers apply
the benchmark's explicit second training mask and supervised target mask;
DMF receives actual sample indices for shuffled batches. The default runner
uses PyTorch directly. They are distributed under this application's PolyForm
Noncommercial 1.0.0 license.

The standalone application imports these local modules, not an editable
`src/jingwei` checkout. This keeps the V1.0 release reproducible without changing
or republishing the separate JINGWEI Python package.
