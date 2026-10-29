# Release asset scope

All three required groups refer to the same frozen synthetic dataset identity
shown in `release-manifest.json`. Archive parts are kept outside Git. Exact
archive/part hashes are in that manifest; member paths, sizes and SHA-256 are in
[the per-file catalog](../provenance/asset_files.json). No numerical file was
regenerated or repacked for the history import.

| Group | Files | Original bytes | Role | Planned public upload |
|---|---:|---:|---|---|
| `processed` | 771 | 234,172,874 | Train/Validation/Test windows, features, masks, topology, metadata, targets and split/hash manifests | Yes |
| `weights` | 6 | 3,253,837 | Three complete C-r8 checkpoints and three frozen prediction streams | Yes |
| `training` | 42 | 42,308,382 | Three seeds' full parents, five-fold parent checkpoints and OOF predictions | Yes |
| `supplement` | 17,152 | 2,907,806,501 | Original synchronized observations and hidden simulator audit states | Offline optional supplement |

The three required archive parts total **280,391,680 bytes**. Their uncompressed
files total 279,735,093 bytes. The larger supplement is not needed by any
training, preprocessing, prediction or evaluation command in this repository.
It remains unchanged in the offline candidate. It supports later raw-window or
simulation-state audits; providing it alone is not a claim that simulator target
generation has been independently reconstructed.

## Training state provenance

`training/checkpoints/` contains nine files: LightweightTransformer, Correction
and the complete S21A-Lightweight-Graph parent for each seed 42/43/44.
`training/crossfit/` contains 30 fold-specific checkpoints (five folds × two
parents × three seeds), plus three OOF arrays. The full parent checkpoints
contain their configuration, selected epoch, preprocessing and same-seed model
state; each fold records the fitting and held families and its own scaler.
`configs/crossfit_partition.json` gives the exact whole-family fold allocation.

The final C-r8 checkpoint is self-contained for inference; the other states are
needed to inspect or reconstruct the staged construction without inventing
missing parent fits. Full from-scratch training can produce its own parents and
OOF arrays and has not been repeated for this release. All these distributed
states were trained using the same original simulation data and original code.
NASA, CES, Aalborg and graph-free adaptation checkpoints are excluded.

Features, units, label/censoring semantics and split counts are described in
[DATASET.md](DATASET.md). Frozen preprocessing statistics are included in the
package and embedded in the deployed checkpoints. Hidden states, absolute age,
IDs and target values do not enter the model input.

Jiajun Ma (2026) licenses the processed dataset and model/OOF groups under
CC BY 4.0; LICENSES/DATA.md and LICENSES/MODELS.md state the exact scopes and
attribution. Third-party dependencies are installed separately and retain
their own licenses.
