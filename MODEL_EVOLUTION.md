# Model evolution

This record follows three verified research commits. The snapshots were imported
on 29 October 2026 with the actual import dates. They are selected original
computational definitions with packaging imports adapted, not a migration of the
private repository's Git history. Original commit IDs, dates, source hashes and
acceptance-record hashes are in [history/manifest.json](history/manifest.json).

## Uncompressed S21A-Graph

Research commit `04b1e3689f80cc9169eda2037a28a1b9521485d5`, 4 October 2026.
[Snapshot](history/s21a_graph.py).

A width-64, two-layer standard Transformer predicts the temporal base. A
condition-modulated three-layer heterogeneous graph encoder supplies C1 features
to a within-window GRU correction expert. Five whole-Training-family cross-fits
provide out-of-fold predictions for an advantage-supervised convex gate. The
complete system stores 350,523 parameters. Its accepted Test family-macro MAE
was 867.584964 h.

## Lightweight temporal parent

Research commit `bbd68224f32d930e0ef43913899542c87e957a99`, 4 October 2026.
[Snapshot](history/lightweight.py), [system diff](history/diffs/temporal_parent.patch).

This separately trained supplemental model replaces four temporal LayerNorms
with DyT, the temporal feed-forward blocks with degree-two learned edge
functions (64 → 32 → 64), and the scalar readout with a signed-power-of-two QAT
head. The graph, GRU correction and cross-fit gate recipe are inherited. These
are real temporal changes; the original uncompressed checkpoint was not merely
renamed or compressed in place.

The complete model has 333,951 parameters and accepted MAE 865.684182 h. The
paired family interval for its MAE difference from S21A-Graph crosses zero;
worst-family error and seed variability worsen. Its contemporaneously measured
inference was slower. The parameter reduction alone does not establish faster
or reliably more accurate complete-system inference.

## Lossless deployment and C-r8

Research commit `9c02826b7f26809a39122c0e80aa97e8add86848`, 6 October 2026.
[Graph export snapshot](history/compact.py),
[relation implementation diff](history/diffs/relation_mapping.patch),
[inference export snapshot](history/deployment.py).

The accepted study branches from the same lightweight parent:

```text
S21A-Graph research recipe
  └─ separately trained lightweight temporal parent
       ├─ A + B: Lossless-Compact deployment (equivalent function)
       └─ A + B + rank-8 value maps + expert adaptation: C-r8
```

A contracts each relation/head's additive attention matrices and attention
vector into two score vectors. B evaluates only C1's backward dependency cone,
including residual and destination-score dependencies. All three physical
message-passing hops remain; successive destination unions are 6/4/1 and
relation unions are 6/5/2. A+B stores 170,367 deployment coefficients, with the
original lightweight training parameter count unchanged. Its function is
preserved within floating-point precision.

C-r8 additionally initializes each concatenated 64 × 64 relation value map by
CPU SVD at rank 8, then adapts only the correction expert for six epochs. The
same-seed temporal base and inherited OOF-trained gate stay frozen; Validation
selects the earliest minimum including epoch zero. Relation/type batching and a
static sealed readout are function-preserving inference exports. C-r8 is the
final balanced paper model: 130,431 parameters, MAE 870.573393 h and RMSE
967.557481 h. Its mean MAE is 2.988429 h above the uncompressed reference and
4.889212 h above the lightweight parent. The accepted median latency is
11.532837 ms; peak incremental GPU allocation remains 8,519,168 bytes.

Lossless-Compact and C-r8 are distinct branches. Other low-rank, basis, width
and distillation candidates from that bounded study are not represented as
linear ancestors of C-r8. No new algorithm experiment is part of this release.

## What has been reproduced here

The historical modules are **historical code snapshots**, with structural and
function-equivalence checks. Original historical checkpoints are not all
distributed or independently replayed by this release. Their reported numbers
above come from the accepted research records, not new release experiments.

The supported commands and full frozen-checkpoint replay concern the final
C-r8 system in `src/fh_liteformer/`. Lightweight parents and cross-fit assets
needed for its staged recipe are supplied separately. Full three-seed training
from scratch and efficiency remeasurement have not been repeated. The
compression studies used an already exposed Test set; they are post-hoc
evidence, not independent confirmation. Synthetic hours are not calibrated
field lifetime. [Reproduction details](docs/REPRODUCTION.md).
