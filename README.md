# FH-LiteFormer

FH-LiteFormer estimates the current remaining useful life of a DC-link capacitor from a synchronized heterogeneous system history. This repository contains **C-r8**, the final compressed paper model, and the staged training recipe that produced its parent models.

This is a local release candidate. Publication and the proposed license grants remain pending rights clearance. The adjacent `release-assets` directory contains the prepared data and checkpoints; no public download URL is asserted here.

The model combines a 64-channel temporal encoder with DyT normalization, degree-two edge-function feed-forward layers and a signed-power readout; a condition-modulated, three-layer heterogeneous graph encoder; a within-window GRU correction expert; and an advantage-trained convex fusion gate. C-r8 contracts attention scores, prunes computations outside the target dependency cone, and factorizes relation value maps at rank 8. The complete deployed system stores 130,431 parameters.

## Installation

Python 3.12 was used for the release checks. Python 3.10–3.12 are supported by the package metadata; other Python versions have not been tested.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS), then run:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
```

The pinned PyPI installation supports CPU inference. The original measurements used PyTorch 2.3.1, CUDA 11.8 and an RTX 4080 Laptop GPU. A matching CUDA-enabled PyTorch build can be installed using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). Keep the other dependency versions fixed when comparing checkpoint outputs.

## Data and frozen checkpoints

From this directory in the offline delivery, run:

```bash
python -m fh_liteformer.fetch --base-url ../release-assets --groups processed weights training --output .
python -m fh_liteformer.data --data data/frozen_graph
```

The same downloader accepts an HTTP(S) Release assets directory once a release exists. It verifies every part, merges split archives, verifies the complete archive, and rejects unsafe extraction paths. Asset sizes and SHA-256 values are in `release-manifest.json`; per-file frozen hashes are preserved in `data/frozen_graph/data_hashes.json`.

The processed dataset contains all 256 families and their original Train/Validation/Test allocation (176/32/48). Each family contains 32 layout/design/duty variants. Training and selection use layouts 0/3/5/6; layouts 1/2/4/7 remain held out. Labels are current-operating-policy **Reference RUL in synthetic hours**, with separate censoring information. They are distinct from the measured predictor channels.

The optional supplement contains all synchronized raw observations and hidden simulator states for the same families. Hidden states are audit data and are excluded from every model input:

```bash
python -m fh_liteformer.fetch --base-url ../release-assets --groups supplement --output .
python -m fh_liteformer.data --data data/frozen_graph --include-supplement
```

See [the data description](docs/DATASET.md) for fields, units, masks, topology and exclusions.

## Preprocessing, inference and evaluation

```bash
python -m fh_liteformer.preprocess --data data/frozen_graph --output runs/preprocessing.json
python -m fh_liteformer.predict --data data/frozen_graph --artifacts artifacts --seed 42 --device cpu --output runs/predictions.csv
python -m fh_liteformer.evaluate --data data/frozen_graph --artifacts artifacts --device cpu --output runs/frozen_evaluation.json
python -m pytest -q
```

`predict` reads observations and metadata without opening target files, including censored endpoints. `evaluate` uses the 23,420 common Reference-valid Test endpoints across 48 families, replays all three checkpoints, and compares predictions with the hash-bound accepted streams. Metrics average equally across families, then equally across seeds. RMSE is the mean of within-family RMSE values.

| Seed | Test MAE (h) | Test RMSE (h) |
|---|---:|---:|
| 42 | 870.960820 | 965.565247 |
| 43 | 879.321295 | 975.761243 |
| 44 | 861.438065 | 961.345954 |
| Mean | **870.573393** | **967.557481** |

These are frozen-checkpoint results. C-r8 is a post-hoc checkpoint adaptation on a previously exposed Test set. The result is not independent confirmation, and the operational-policy synthetic hours have not been calibrated as field lifetime. C-r8 does not denote the historical CES S21-A, uncompressed S21A-Graph, or the graph-free gEOL adaptation.

## Staged training

The model was not trained in one end-to-end optimization. The parent recipe trains the temporal base and graph-conditioned correction expert, builds five whole-Training-family cross-fits with fold-specific normalization, and trains the advantage gate from out-of-fold predictions. C-r8 then adapts only the rank-8 correction expert for six epochs; the base and inherited gate remain frozen. Validation selects the earliest minimum, including adaptation epoch zero.

Run the full parent chain and then C-r8 adaptation in new output directories:

```bash
python -m fh_liteformer.train --data data/frozen_graph --output runs/parents --seeds 42 43 44 --device cuda
python -m fh_liteformer.adapt --data data/frozen_graph --parents runs/parents --output runs/c_r8 --seeds 42 43 44 --device cuda
```

To reproduce the post-hoc adaptation from the supplied frozen parent systems instead, use:

```bash
python -m fh_liteformer.adapt --data data/frozen_graph --parents artifacts/training --output runs/c_r8_from_frozen_parents --seeds 42 43 44 --device cuda
```

Small diagnostics exercise the same stage dependencies without claiming a retrained paper result:

```bash
python -m fh_liteformer.train --data data/frozen_graph --output runs/parent_smoke --seeds 42 --device cpu --smoke
python -m fh_liteformer.adapt --data data/frozen_graph --parents artifacts/training --output runs/adapt_smoke --seeds 42 --device cpu --smoke
```

Full retraining and hardware efficiency measurements are separate from checkpoint replay. [Reproduction notes](docs/REPRODUCTION.md) state the validation scope and numerical conditions.

## Citation, sources and licensing

The manuscript is unpublished. Its author list, final bibliographic metadata and any later DOI must be supplied by the authors; no DOI is assigned in this repository. Until then, identify the model as FH-LiteFormer C-r8 and the data by the frozen identity recorded in `release-manifest.json`.

The intended licenses are MIT for cleared original code and CC BY 4.0 for cleared original simulation data and C-r8 weights. The license grant and accurate copyright/attribution notices are pending rights confirmation; see [LICENSE](LICENSE). The training-parent archive also requires explicit clearance. Dependency packages retain their own licenses. No PatchTST source, NASA/CES raw data or Aalborg raw data are bundled.

External experiments use the [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [Aalborg gEOL dataset](https://vbn.aau.dk/en/datasets/power-cycling-test-dataset-geol-a-gradient-based-end-of-life-crit/). Obtain those data from their official sources and follow their applicable terms. The historical CES experiment had accidental pre-freeze label-byte access after development closed; subsequent formal evaluation used a separately frozen recovery protocol. Those targets must not be described as pristine untouched holdout data.
