# Application-local JINGWEI adapters

These are application-local implementations, independent of the published
Python package. DMF uses explicit rank-8 bilinear interactions, row/column biases,
residual neural towers and a mask head sharing row embeddings. `LegacyDMF`
retains the former concatenated-embedding MLP for diagnostic controls. DCAE uses
the existing global autoencoder. The default is 500 epochs without early stopping
or an additional internal validation split, using final-epoch weights.

The wrappers apply
the benchmark's explicit second training mask and supervised target mask;
DMF receives actual sample indices for shuffled batches. The default runner
uses PyTorch directly. They are distributed under this application's PolyForm
Noncommercial 1.0.0 license.

The standalone application imports these local modules, not an editable
`src/jingwei` checkout. This keeps the V1.0 release reproducible without changing
or republishing the separate JINGWEI Python package.
