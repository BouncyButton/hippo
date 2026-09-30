# Locked reset-AdamW diagnostic

This small fitting-only diagnostic precedes any longer continuation screen.
It tests whether actual optimizer updates reproduce the shared-head SGD concern.
No updated weights are saved, promoted, or evaluated on inner/outer validation.

## Design frozen before execution

- Saved E checkpoints in folds 1–4, with their original explicit-loss weights.
- In each fold, sampler seed 1000 draws three distinct fitting batches of two
  cases. Batches 1 and 2 independently supply one update; batch 3 is the shared
  probe. Probe cases never supply a gradient. Fitting cases can overlap across
  folds; the repeated probe is not an independent replicate per update.
- Reset to saved E and fresh AdamW before every arm/update: lr=1e-4,
  weight_decay=1e-5, betas=(0.9,0.999), eps=1e-8. This tests the proposed reset
  continuation, not historical AdamW moments or a multi-epoch trajectory.
- Freeze beta at its saved value for every arm. Keep product occupancy,
  tau=0.5, original head capacity, all original explicit losses, and normalized
  segmentation coefficients. No focal intervention.
- Four arms: J full joint; H stop rendered-loss head-parameter gradients while
  preserving direct/indirect backbone routes; V keep only direct rendered-loss
  voxel gradients; R raw DiceCE with the same total segmentation coefficient.
- One AMP training-mode optimizer step, with identical stored input tensors and
  stochastic seed across arms for that update. The source fitting loader's
  transformations are retained. cuDNN deterministic=True, benchmark=False.
  No gradient clipping, matching the historical loop. Abort on nonfinite
  gradients, a skipped optimizer step, or beta changes.
- Before/after measurement uses float32 eval mode on exactly the same tensors,
  on both the update and the separate fitting-probe batch. Capture actual
  foreground argmax margins (max foreground logit minus background), not a
  binary-union probability threshold as a proxy for the three-class decision.

## Additional common-shift control

After J's update, shift its scalar presence bias by the difference between H
and J's all-ray mean presence logits on the update images. This uses no labels.
Recompute the full head/fit/renderer for update and probe images with that bias.
This J_bias control matches a mean-logit shift, not an FP burden. It is a local
mechanistic comparison; FP-matched inner calibration still belongs in the
subsequent continuation protocol.

## Outcomes and interpretation

Keep thin/erased/hard-empty group membership fixed at pre-step E predictions.
Record both any-ray raw detection and raw overlap with true thin voxels, so
mislocalized raw speckles do not count as preserved true anatomy.

Record presence direction and overlap in fixed groups, thin-voxel recall,
FP-ray and FP-voxel counts, FN voxels, deep-interior FN, raw/final Dice and ASSD,
renderer deletion of raw true foreground, and newly lost/recovered true voxels.
Keep fold, case and update identities. Report all arms without selecting a
winner from the probe outcomes. No bootstrap inference from repeated cases.

H is promising for continuation if its probe response repeatedly improves
correct thin overlap/voxel recall over J without increasing either FP burden,
and cannot be explained by J_bias. If H only moves true/empty scores together,
or improves update batches while harming probes, these outcomes weaken this
remedy and motivate a discrimination/architecture experiment instead.

One-step failure cannot establish that longer training will fail; one-step
success cannot establish a stable segmentation improvement. J/H backbone
parameter differences after the first step are recorded to check the expected
matching, allowing visibility into numerical accumulation differences.

The planned full follow-up remains J/H/V/R with three continuation seeds per
fold and the existing inner selection rules, FP guards, E fallback, and frozen
outer evaluation. This diagnostic does not launch that larger experiment.

## Verification and provenance

Verify data/split hashes, original source hashes against the completed component
audit, and checkpoint hashes against their completion records. Save code hashes,
actual optimizer/scaler step records, case membership and numerical outcomes.
Upload only code and this protocol; use source/data already on the cluster.
The implementation tests check full-loss routing, retained explicit head
supervision, real AdamW execution/frozen beta, fixed ray groups, and bias shifts.
