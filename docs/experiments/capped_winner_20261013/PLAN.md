# Winning loss under the previous four-model compute cap

Authorized follow-up: apply the selected constraint to the existing four-model
budget benchmark, preserving the requested three folds and three seeds. One
sequential Slurm job, dependent on the current standard-epoch replication.

## Scope

- Models: SwinUNETR, nnU-Net, UNETR ViT-B/16, and BasicUNet16 (3D U-Net).
- 36 fresh runs: four models × folds 0/1/2 × seeds 0/1/2.
- Existing ten-training-volume subsets and 52-volume validation sets per fold.
- Optimization-only allowance: 31,976,450,667,090 FLOPs per run. Calibration is
  accounted for separately, as in the previous capped benchmark.
- Keep the frozen architecture, optimizer, precision, mild augmentation,
  learning-rate trajectory, validation-compute checkpoints and absence of early
  stopping from the prior capped campaign. No auxiliary branch is used.
- nnU-Net retains native Dice+CE and all original deep-supervision heads;
  additional anatomical losses act only on full-resolution logits.

## Frozen transferred constraint

Selected recipe: `combo_depth_g100`, which achieved 81.5015718459% best validation
macro Dice in the standard-epoch SwinUNETR fold-0/seed-0 screen. That result is
not a prediction for the capped protocol.

1. Outer bands: 3 mm inside / 2 mm outside, Brier error, equal side mass.
   Inside: 35% ordinary mean + 65% mean over the hardest 70%.
   Outside: 15% ordinary mean + 85% mean over the hardest 10%.
2. Independent inside/outside endpoint targets across the annotated outer
   boundary, with 1- and 3-mm separation and equal scale weights. Require the
   annotated path to remain on the corresponding side.
3. Original depth-balanced multiclass CE, using annotation-derived physical
   depth bins, the original ROI and sparse-bin taper.

Recalibrate coefficients on each model/fold/seed's own ten training volumes.
Bands target 12.5% of the native region-gradient RMS; endpoints target 7.5% of
region+bands; depth targets 10% of region+bands+endpoints. All use the prior 50%
p95 safeguard. Keep the prior capped prefix/ramp: 9/9 updates for nnU-Net,
50/50 for the other models. Geometry and relative budgets stay fixed.

Outer targets are rebuilt from the augmented annotation on its 1-mm model grid.
The valid native field of view follows the same spatial transform, preventing
padding edges from becoming anatomical boundaries. Depth uses the existing
augmented physical affine, exactly as in the prior capped depth experiments.

## Accounting and checks

The new loss has variable sparse-support sizes. Charge a conservative audited
upper bound rather than ignoring its overhead: at most N band voxels per side
and 6N directed endpoint samples per side/scale. Add the extra logit-gradient
accumulation to the unchanged native-model update cost. The declared arithmetic
convention still excludes target preprocessing, augmentation, validation,
comparisons, indexing, top-k ranking and I/O. This is not a GPU instruction count.

Local checks verify exact unaugmented target geometry and individual loss
gradients, numerical equivalence of the combined gradient, unchanged native
warmup, training-only calibration, transformed FOVs, and measured loss costs
below the bound. Before training, an allocation-local gate checks the 30
training-volume/fold entries and an actual BF16 forward/backward/optimizer step
for every architecture; it also checks unchanged nnU-Net coarse-head gradients.

## Comparison and retention

Primary outcome: final-budget macro Dice versus the already completed baseline
and original-bands run with the same model, fold and seed. Retain best-checkpoint
Dice, per-class/per-case metrics, curves, configuration, calibration and budget
ledgers. Verify both checkpoints and preserve their hashes before deleting each
completed run's checkpoint files. Active checkpoints remain resumable.

This remains a development/transfer experiment: the objective was selected on
SwinUNETR fold 0 seed 0, and cross-validation partitions are not an untouched
external test set.
