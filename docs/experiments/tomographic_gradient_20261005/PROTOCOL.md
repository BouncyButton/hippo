# Cumulative ray gradient probe

Fixed before the numerical cohort run, 5 October 2026. Scope: test the requested
training-only tomography proposal, not train a segmentation model.

## Why revisit this branch

The September ray-moment loss already implemented orthogonal thickness (M0)
and optional first/second depth moments. M0 and M0+M1 both passed frozen-logit
repair checks, but completed M0 training pilots did not produce a reliable gain
and introduced anterior/background errors. The September 15 local quadratic
ray redesign also lost to plain wrong-only foreground BCE at matched logit
update RMS. A positive repair result alone therefore cannot reopen training.

This probe tests a distinct measurement: the full cumulative tissue residual at
every depth, averaged over both ray directions. It includes unmodified legacy
M0 and M0+M1, ordinary CE, and plain wrong-only foreground BCE as controls.

## Cohort and source

- MSD fold 0: use training-partition volumes only. Include the retained
  checkpoint's 10 fitting cases and sample 22 more from the remaining training
  partition with NumPy seed 0, before predictions are inspected.
- The retained SwinUNETR is the 610-update-budget, 10-case, seed-0 baseline at
  `budget_baselines_94603_684354/swinunetr/fold0/seed0/run/checkpoint_best.pt`.
  Its checkpoint selection previously used validation. This probe does not
  access validation labels for gradients, scoring, selection, or calibration.
- Historical mature checkpoints were retired. Do not numerically compare this
  probe's repair counts with the earlier audits as if checkpoints were matched.
- Use the baseline's 64-cube center padding/cropping, nonzero MRI z-score,
  original 1-mm axes, and a single frozen FP32 inference per selected case.
- Record actual checkpoint metadata and SHA-256 hashes of inputs and source.
- Save logits/labels remotely for reproduction; never commit medical arrays.

## Measurements and controls

Equal-weight groups are hippocampus union, anterior, and posterior. Predictions
and annotations share the same fixed coordinates; no label-based alignment.

1. Dice including background (MONAI-equivalent, smoothing 1e-5).
2. Multiclass cross-entropy.
3. Wrong-only foreground BCE, detached foreground-argmax error gate and a fixed
   whole-volume denominator, equivalent to the September prototype at strength 0.
4. Legacy M0, unchanged SmoothL1 beta .02.
5. Legacy M0+M1, unchanged.
6. Three-axis thickness L1.
7. Three-axis cumulative-profile L1.
8. Seven-view thickness L1.
9. Seven-view cumulative-profile L1.
10. Seven-view thickness plus cumulative profile, coefficients 1 and 1.

The seven views comprise the three exact axis reductions and four unoriented
body diagonals. Diagonal sampling uses 2-mm detector spacing, 1-mm ray-depth
spacing, zero padding, a canvas covering full interpolation support, and the
same trilinear operator for prediction and one-hot annotation. All measurements
divide by the fixed 64-mm crop length. Cumulative profiles are inclusive sums
from both ends, averaged equally; no data-adaptive depth gates or weights.

## Controlled errors

For the first eight sorted selected cases, test the correct mask, a two-voxel
axis-0 translation, three-slice low-axis-1 truncation, a radius-3 protrusion,
radius-2 internal hole, and a two-voxel A/P interface shift. Synthetic logits
retain the exact candidate argmax, with finite distance-dependent confidence.
For the first two cases, also quadruple translated-logit margins to expose
softmax saturation. These are falsifiers on real shapes, not natural errors.

## Gradient and update comparisons

- Compute gradients with respect to logits, not only mask probabilities.
- Record raw RMS/max, cosine with Dice and CE, per-class energy, and energy and
  helpful/harmful true-class margin directions for correct background,
  correct foreground, FP, FN, swaps, and confident wrong voxels.
- Report every predeclared auxiliary-only update: max-logit step .25/1/2 and
  whole-logit RMS .02/.05/.10. Also record actual RMS and max in each arm.
- Combined diagnostic: auxiliary gradient rescaled to 10% of that case's Dice
  gradient RMS; sum with Dice; normalize the whole update to RMS .10. Compare
  with Dice alone and Dice plus the simple controls at identical total RMS.
- Record semantic and foreground errors fixed/introduced, FP/FN/swaps,
  anterior/posterior/union Dice, CE, and soft Dice after each update.
- Check the loss after a small max-logit step .001. This is diagnostic, not an
  outcome-selected step or a prerequisite for claiming anatomical benefit.
- No optimizer, network parameters, training recipe, or inference path changes.

## Interpretation rule

Evaluate depth profiles against legacy M0/M0+M1 and the same-view thickness loss
at the same update budget. Added-to-Dice benefit must also be compared with CE
and wrong-only BCE. Report tradeoffs and individual harms. A positive result is
only a mechanism result: previous positive frozen-logit gradients did not
predict a training benefit. This task does not authorize a new training pilot.
