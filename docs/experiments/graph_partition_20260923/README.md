# Graph partitioning: completed feasibility experiments

23 September 2026. **Both prespecified advancement gates failed.** No full-network
training, learned graph-affinity training, second-model replication, or development
evaluation was launched after those failures. The existing training and inference
defaults are unchanged.

The experiments establish a more specific result than “graphs do not work”:
surface/MRI interface descriptors can recover deliberately displaced cuts, but
they do not add localization accuracy to the tested natural-output readout and
they often disturb already correct cuts. Uniform graph regularization gives a
small voxel-level improvement without improving cut location.

## What ran

- All 208 fold-0 training cases; unaugmented early-stopped seed-0 SwinUNETR.
- New CPU inference using the established no-resize 64-cubed preprocessing.
  Every case reproduced the prior cached cut, target, base features (within the
  specified numerical tolerance), and raw Dice. Native image/label hashes agree.
- Fourfold case-held-out readouts on natural outputs and on 624 synthetic variants
  (three per case). Variants of a case always remain in one fold. No parameter
  search: all logistic readouts use the predeclared C=0.1.
- 832 fixed-support graph-cut evaluations: four lambda settings per case. For
  each held-out fold, lambda was selected using only the other three folds.
- A written [protocol](PROTOCOL.md) preceded new extraction and intervention
  results. Its hash and source hashes are retained in [manifest.json](manifest.json).

This is **probe case holdout with an in-sample backbone**, not an out-of-fold
backbone experiment. The backbone was trained on these 208 cases. Participant
linkage is unverified. No development images/prediction caches were opened in
this experiment. Intervals are descriptive case-bootstrap intervals conditional
on fitted fold models; they do not capture model-refitting uncertainty or establish
participant-level generalization.

## 1. Do interface descriptors add useful localization information?

All candidate readouts use the same candidate planes and render their selected
cut onto the entire original predicted foreground. Labels supply training targets
and scoring only. The new descriptors use predicted geometry, never reference
foreground.

The base has 24 probability/position/geometry features. Existing MRI adds 96 pooled
features. New surface descriptors contribute 110 features derived from distance to
the exterior and smoothed signed-distance normals/curvature; edge MRI contributes
120 face-endpoint mean/difference features. Both are pooled globally and in four
X/Z sectors. These are **candidate interface descriptors**, not a GNN, independently
supervised per-edge classifier, or an anatomical uncal-apex annotation.

| Method | Cut MAE, mm | Raw A/P Dice | A/P swap voxels |
|---|---:|---:|---:|
| Original network | 0.5000 | 0.902868 | 8,860 |
| Network's fitted plane | 0.5000 | 0.903643 | 8,336 |
| Base readout | 0.4279 | 0.905394 | 6,972 |
| Base + existing MRI | **0.3990** | **0.906469** | **6,330** |
| Base + new surface descriptors | 0.4423 | 0.905148 | 7,136 |
| Base + new surface/edge MRI | 0.4279 | 0.905623 | 6,860 |
| Base + shuffled new descriptors | 0.4327 | 0.905290 | 7,057 |

The new full descriptor arm has **zero mean cut-MAE change versus base**, with
95% interval [-0.0481, +0.0481] mm. It improves 12 cases and worsens 12. Its Dice
increment is +0.0229 percentage points, interval [-0.0931, +0.1411].

Against existing MRI, new descriptors worsen cut MAE by 0.0288 mm and reduce Dice
by 0.0845 points; both intervals span zero. The tiny advantage over shuffled new
features also has an interval spanning zero. Fold-specific changes versus base
are mixed: one tie, two improvements, and one worsening.

Although the new readout beats the raw network on Dice, the base readout already
does most of that. Crediting the whole difference to the new graph representation
would be incorrect. The new features fail the predeclared incremental-information
criterion and do not beat the existing MRI representation.

## 2. Can those descriptors recover a deliberately misplaced boundary?

The synthetic test replaces the network's A/P margin with a linear transition
centred at the reference best-fit cut minus two, zero, or plus two slices. It keeps
predicted foreground and MRI unchanged. This imposes a known artificial error
distribution; it is not a realistic model of all network mistakes or confidences.
The “correct” starting cut is the reference best-fit plane, which need not exactly
reproduce mixed-slice labels. All scoring retains the original labels.

| Starting displacement | Base readout MAE | Existing MRI MAE | New surface only MAE | New surface/edge MRI MAE | Shuffled new features MAE |
|---|---:|---:|---:|---:|---:|
| -2 mm | 1.995 | 0.923 | 0.841 | **0.755** | 1.822 |
| 0 mm | **0.024** | 0.673 | 0.490 | 0.601 | 0.293 |
| +2 mm | 1.889 | 1.000 | 0.923 | **0.817** | 1.736 |

There is real discrimination within this synthetic task. New full descriptors
reduce MAE versus base by 1.240 mm for the -2 shift and 1.072 mm for the +2 shift.
The corresponding intervals are [-1.351, -1.130] and [-1.183, -0.962] mm.
They also improve over existing MRI by 0.168 and 0.183 mm, with intervals excluding
zero for both shifts. Shuffling largely removes this recovery.

However, new descriptors move **89 of 208 initially correct cuts**, compared with
two for the base readout. At zero displacement they increase MAE over base by
0.577 mm, interval [0.471, 0.683], and reduce Dice by 1.273 percentage points,
interval [-1.520, -1.033].

This supports an anatomical/contextual localization signal under controlled
corruption, but **does not establish when to override an already good prediction**.
The preservation requirement fails. It would be wrong to average the three strata
and claim safe correction without reporting damage to correct cuts.

## 3. Does ordinary graph inference improve existing predictions?

The six-face graph uses conditional A/P negative log probabilities as unaries and
a nonnegative uniform Potts cost for each separated voxel face. It preserves the
entire original foreground, including islands. There are no label-derived seeds,
volume constraints, or learned affinities.

| Lambda | Cut MAE, mm | Raw A/P Dice | Dice change, percentage points | Swaps |
|---|---:|---:|---:|---:|
| 0, original | 0.5000 | 0.902868 | 0 | 8,860 |
| 0.05 | 0.5000 | 0.902886 | +0.00185 | 8,847 |
| 0.2 | 0.5000 | 0.902971 | +0.01030 | 8,791 |
| 1.0, selected in every fold | 0.5240 | 0.903180 | +0.03125 | 8,636 |

Every fold selected lambda=1 using the other folds' raw Dice. Its Dice-change
interval is [+0.0145, +0.0489] percentage points. It helps 103 cases, harms 44,
and leaves 61 unchanged on Dice. It reduces swaps by 224 (2.53%).

But cut MAE worsens by 0.0240 mm, interval [0.0048, 0.0481]: five cases have worse
cuts and none has a better cut. No setting collapses a case to a single A/P class.
The small Dice gain is below the predeclared 0.1-point advancement threshold, and
the cut-localization requirement also fails. This is limited local repair, not
evidence that the graph has found a better anatomical separator.

Capacities are rounded at scale 10,000 for SciPy max-flow. The solver is exact for
that quantized energy; each case records the original floating objective and a
conservative rounding bound. Foreground preservation and lambda-zero identity
were checked for every case.

## Decision and consequence for training

The [machine-readable decision](decision.json) rejects advancement of both tested
branches. No augmented-model replication, learned-affinity training, or full
segmentation-network structured-loss trial was launched. This follows the frozen
staging rule, not a claim that a graph-based training objective cannot help.

The remaining question is narrower: **can independent context identify when a
model's otherwise coherent cut is wrong, and make a reliable directional repair?**
The synthetic recovery result motivates that question, but supplies neither a
validated reliability gate nor evidence of incremental value on natural errors.

A next experiment should obtain genuinely out-of-fold backbone predictions and
test correction-versus-preservation under their natural error distribution. It
would need a separately frozen protocol and comparisons against the existing MRI
readout. Confidence alone is not a validated gate. Expanding graph architecture
or starting a large training sweep from these results is not warranted.

## Verification and reproduction

Fifteen targeted tests passed, including independent exhaustive optimization on
small graphs, unary orientation, disconnected foreground, island removal versus
strong evidence, feature normalization, target/feature separation, and synthetic
displacement behavior. An additional 60 random small-graph comparisons against
exhaustive enumeration had maximum floating-energy discrepancy 8.9e-16.
All 208 native inputs have matching image/label geometry and 1-mm RAS spacing;
their hashes match the extracted cache.

From the repository root:

```bash
rtk proxy env MPLCONFIGDIR=/tmp/hippo-graph-mpl .venv/bin/python -m pytest evaluation/test_graph_partition_probe.py -q
rtk proxy env MPLCONFIGDIR=/tmp/hippo-graph-mpl .venv/bin/python evaluation/graph_partition_probe.py --stage extract
rtk proxy env MPLCONFIGDIR=/tmp/hippo-graph-mpl .venv/bin/python evaluation/graph_partition_probe.py --stage screen
```

Required inputs are the local dataset, the documented early-stopped checkpoint,
and the existing unaugmented context-feature cache. Source/protocol provenance
changes cause the runner to refuse the existing cache. Reproduction needs an
unchanged input state or a new explicitly versioned experiment.

Implementation: [runner and graph solver](../../../evaluation/graph_partition_probe.py),
[tests](../../../evaluation/test_graph_partition_probe.py).

Evidence: [descriptor summary](descriptor_summary.json),
[descriptor cases](descriptor_cases.json), [graph summary](graph_summary.json),
[graph cases](graph_cases.json), [input inventory](input_inventory.json),
[provenance manifest](manifest.json).

Dense MRI-derived arrays remain in ignored `experiments/graph_partition_20260923/`.
The local [results figure](../../../experiments/graph_partition_20260923/results.png)
is also an ignored generated artifact; all numerical results above are retained
in the linked JSON files.
