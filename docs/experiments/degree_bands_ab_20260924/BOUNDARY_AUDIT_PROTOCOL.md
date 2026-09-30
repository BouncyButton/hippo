# Selected-checkpoint boundary correction audit

Inference only, requested after training completed. Primary comparison: A/B
versus the existing early-stopped augmented Dice-only fold-0 seed-0 run.
Also include the matched existing Dice+bands model to isolate whether the
degree variants outperform ordinary bands. No new training or selection.

Use all 52 original validation cases, the exact frozen model/data source,
each run's selected best checkpoint (Dice 24, bands 21, A 22, B 30), and
identical A100 MIG 3g.40gb CUDA/AMP inference. Verify checkpoint/config and
export hashes, split/data hashes, completion records and matching training
settings. Compare against the earlier completed audit's FP/FN/A-P swap counts.

Primary error support is the original GT whole-hippocampus two-step inner
and outer bands, using six-face morphological connectivity exactly as in
training. Report immediate and second layers separately, and all errors
beyond the bands. These are morphological layers, not an assertion of
equivalence to Euclidean two-millimetre distance.

For every case and support count correct, missed foreground (FN), added
foreground (FP), and A/P swaps separately. Count exactly which baseline
errors were corrected and which previously correct voxels became wrong.
Retain the 4-by-4 error-state transition matrix so that recovered foreground
with the wrong A/P class cannot be mistaken for full semantic correction.
Report both union and full-class corrections and their net change.

Report whole-graph degree 0..6 and grouped 0-1/2-3/4-5/6; anterior and
posterior subsets of the same degree, plus separately labelled within-class
graph degrees. Include the immediate GT A/P interface. These overlapping
strata are descriptive and must not be summed across definitions.

Aggregate pooled voxel counts and equal-case mean metrics. Include macro
A/P Dice, union Dice, class Dice, six-face surface-voxel ASSD/HD95 and surface
Dice at 1 and 2 mm (1-mm isotropic data, pooled bidirectional surface samples;
not area-weighted meshes). Empty surfaces get null distance metrics.
Balanced boundary error is one half of the inner FN fraction plus one half
of the outer FP fraction, computed per case.

Paired case bootstrap: 10,000 resamples, fixed seed 20260924, descriptive 95%
intervals for candidate-minus-reference metrics. This is a single seed and
an already-used development fold; intervals do not account for training-seed
variation or checkpoint/experiment selection. Case-level outputs remain on
the cluster. Retrieve aggregates and provenance only; do not export MRI or
prediction arrays. Job limit: 30 minutes, 3g.40gb, four CPUs, 16 GB host RAM.

## Inference reproducibility follow-up

Audit 667718 stopped at an exact-count assertion for A, after inference and
scoring. The follow-up preserves the failed attempt, repeats inference twice
for each model, reports changed voxel/case counts and aggregate differences
from the original audit, and retains the actual first-pass paired results.
Source/checkpoint/input hash checks remain strict. Numerical differences are
reported explicitly rather than assuming bitwise equality across inference
runs. No training is repeated.
