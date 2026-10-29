# FH-LiteFormer C-r8 model assets

Creator / Licensor: **Jiajun Ma (2026)**. GitHub account: **kkjid**.
Work: **FH-LiteFormer synchronized simulation dataset and model assets**.
Category: **Model Assets**. License: **CC BY 4.0**; see
[the exact scope and attribution](LICENSES/MODELS.md).

The three complete checkpoints have seeds 42, 43 and 44. Each contains its
frozen configuration, preprocessing statistics and standalone deployed state;
the complete system stores 130,431 parameters. Checkpoint bytes are unchanged.
License/creator metadata are supplied alongside them rather than inserted into
the frozen tensors or serialized payloads. Exact file hashes and sizes are in
`provenance/asset_files.json` and `src/fh_liteformer/configs/artifacts.json`.

The input is a causal 24 h window of synchronized observed node channels,
feature masks, commands, ratings, relative times and physical mounts. The output
is C1 current-operating-policy Reference RUL in synthetic hours. Hidden states,
absolute age, IDs, future information and targets do not enter predictors.

The rank-8 correction expert was adapted from a same-seed lightweight parent;
its temporal base and inherited five-fold OOF advantage gate stayed frozen.
The Test set had already been exposed; this is post-hoc evidence, not
independent confirmation. Synthetic coefficients, instrumentation assumptions
and uncalibrated hours limit field use. Different math libraries can cause
small FP32 replay differences. No energy-saving claim follows from parameter
counts or `ldexp` arithmetic.

Parent/cross-fit/OOF assets are licensed and supplied separately for the staged
recipe. Historical source snapshots do not imply historical checkpoint replay.
Full retraining from scratch and fresh hardware efficiency measurement are not
release-verified. See [REPRODUCTION.md](docs/REPRODUCTION.md) and
[MODEL_EVOLUTION.md](MODEL_EVOLUTION.md).
