# Boundary preservation in the saved A/E experiment

This is a post-hoc internal audit, with no training, checkpoint selection,
renderer changes, or deployment decisions. The four folds and one trained seed
are the saved 208-case function-head robustness experiment.

## Comparisons

- A to E raw: differences associated with the trained systems before rendering.
- A to E final: net difference of the final methods.
- E raw to E final: exact mask changes caused by rendering a fixed E prediction.

The first comparison is not a causal isolation of one loss or gradient route.
All comparisons require exactly matched cases and reference labels.

## Fixed reference regions

Extract the union foreground inner surface with 6-neighbor binary erosion.
Measure Euclidean distances between voxel centers in physical millimeters,
using the verified 1 mm isotropic data. Use the same reference regions for
both predictions:

1. Distance 0–1 mm inclusive, on either side of the reference surface.
2. Distance >1 and <=2 mm, on either side.
3. Reference foreground farther than 2 mm: deeper interior.
4. Reference background farther than 2 mm: distant exterior.

The four regions are disjoint and exhaustive. The all-voxel row is their sum.
An empty reference has no surface and all voxels belong to distant exterior.
These regions are evaluation-only. High coverage of errors by the GT band does
not establish that a predicted-surface band would cover the same errors.

## Error accounting

Record a 7x7 transition matrix: correct plus the six directed class errors.
Count exact three-class fixes/breaks separately from union foreground fixes/
breaks. A missed anterior voxel changed to posterior is a union fix but remains
a class error. A swap changed to background is a new union error although the
class label was already wrong. For each region, verify net error change equals
breaks minus fixes, separately for exact labels and foreground union.

Report baseline/candidate FP, FN, swaps, persistent changed wrong labels,
per-case counts, pooled counts, and case-mean Dice/ASSD. ASSD follows the original
equal average of two directed surface-voxel mean distances. The empty-mask
fallback is the physical padded-volume diagonal, as in the historical code.
Bootstrap patients, not voxels, with 3,000 paired resamples and seed 0. Intervals
are conditional on these fitted models and exclude training-seed uncertainty,
overlapping-fitting-set dependence and prior method-selection uncertainty.

## Reproduction and provenance

Freeze source, dataset, split, checkpoint, and audit-script hashes. Verify each
case's A, E raw and E final ASSD and union Dice against the completed component
audit (job 666651), with absolute tolerance 1e-4. Retain historical values too.
That component audit documented a small unresolved difference from historical
rendered E on some voxels; do not silently overwrite or tighten that reference
after seeing the preservation results. Fail on a reproduction mismatch.

The audit streams predictions from the saved models and writes numerical
results only. Existing experiments and checkpoints are not modified.

## Local implementation checks

`evaluation/test_audit_boundary_preservation.py` checks class versus union
transitions, deep-interior losses, boundary FP correction, region partitioning,
physical spacing, empty references, invalid inputs and duplicate patients.
A separate 52-case integration run compared the historical and early-stopping
baseline masks; this is a software check, not the A/E experiment.

## Success of the audit

Complete all three comparisons on 208 unique outer cases, pass reproduction
checks, and identify where corrections and newly introduced errors occur.
No architecture is promoted on counts alone. Report preservation together with
the final ASSD/Dice tradeoff and case/fold heterogeneity.
