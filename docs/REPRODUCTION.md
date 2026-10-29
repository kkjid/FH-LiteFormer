# Reproduction scope

The source in `src/fh_liteformer/models/` is ported from the accepted graph benchmark, lightweight temporal parent and C-r8 deployment implementation. `provenance/source_lineage.json` records the source-file and extracted-symbol identities. Changes concern packaging, imports, paths, explicit device selection and safe checkpoint loading. PatchTST and unrelated model families are outside this package's dependency closure.

The final system has temporal width 64, four attention heads, two temporal layers, three graph layers, GRU width 32 with one layer, gate width 8, and rank-8 relation value maps. All three seeds are retained. The deployed weights embed their exact model configuration and preprocessing statistics.

## Parent and adaptation training

The parent uses 40 epochs, AdamW with learning rate 0.001 and weight decay 0.0001, effective batch size 512, physical batch size 128, gradient clipping at 1, family-balanced replacement sampling, and SmoothL1 beta 0.1. Validation chooses the earliest strict minimum family-macro MAE. A separately trained graph-conditioned GRU expert consumes the frozen temporal parent's output.

Five Training-family cross-fits use partition seed 2026100601. Both parents are freshly trained inside each fitting partition for the already selected full-parent epoch counts; held families do not select epochs or fit scalers. The advantage gate receives only out-of-fold parent predictions and causal diagnostics. Its label is whether the correction has lower absolute scaled error, with BCE weights `0.02 + abs(advantage)`. The final gate is selected on Validation predictions from the full parents.

C-r8 starts from the corresponding frozen lightweight system, contracts scores and backward dependencies, initializes each concatenated 64×64 value map by CPU SVD at rank 8, and adapts only the correction expert. It uses six epochs, AdamW learning rate 0.0003, the inherited batch/weight-decay/clip/SmoothL1 settings, and the earliest Validation minimum including epoch zero. Base and gate parameters remain fixed. The temporal QAT head is sealed for deployment; its stored master weights remain visible in the parameter inventory.

Training commands refuse an existing output directory. They do not write into the distributed weights. A full run may take substantially longer than checkpoint replay. `--smoke` uses a documented small Train/Validation subset and one epoch, writes `DIAGNOSTIC_SMOKE_ONLY`, and does not access Test. It validates the stage chain and finite optimization; it does not reproduce a reported accuracy value.

## Metrics and numerical comparison

For each seed, compute each family's mean absolute error and root mean squared error over its Reference-valid endpoints, then average the 48 family values equally. Average the three seed results equally. This RMSE is not the root of a pooled endpoint MSE. The seeds are separate models; predictions are not ensembled before scoring.

The frozen prediction streams are checked against the accepted metrics to 1e−7 h. Fresh checkpoint outputs are compared pointwise with a 0.02 h engineering FP32 replay tolerance. This tolerance does not select a model or modify any paper result. The original runtime's TF32-disabled, deterministic GPU replay matched all stored predictions exactly. CPU and different math libraries can differ in floating-point reduction order.

Use four Torch threads, deterministic algorithms, disabled cuDNN benchmarking and disabled TF32; the command-line tools set these explicitly. The original environment used Python 3.12.7, NumPy 1.26.4, pandas 2.2.2, PyArrow 16.1.0 and PyTorch 2.3.1/CUDA 11.8 on an RTX 4080 Laptop GPU. The clean PyPI installation uses a CPU Torch wheel on Windows. Checkpoint replay in that environment is a distinct validation from a matched-GPU efficiency run.

The accepted median batch-1 latency was 11.532837 ms (mean of three per-seed medians), using inputs already on the GPU, ten warmups, fifty synchronized CUDA-event repetitions and no concurrent training. Incremental GPU allocation was 8,519,168 bytes. These historical measurements have not been newly established by the offline packaging tests.

## Validation record

`provenance/local_validation.json` is generated from the actual release checks. It distinguishes frozen-checkpoint replay, small training-stage diagnostics, full retraining and hardware efficiency. Full three-seed parent retraining has not been repeated for this release candidate. Publication, GitHub asset downloads and rights clearance remain incomplete while this is an offline candidate.
