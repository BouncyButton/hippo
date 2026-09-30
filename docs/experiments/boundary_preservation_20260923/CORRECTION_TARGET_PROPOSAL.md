# Proposed next test: teach correction limits separately from existence

Status: subsequently implemented and evaluated in job 666769; all 24 selections
retained E. See the [completed results](CORRECTION_CONTROL_RESULTS_20260923.md)
and [frozen protocol](CORRECTION_CONTROL_LOCK_20260923.md). The original proposal
below is preserved as design history, not a claim of successful validation.

## Why this is a different question

The current scalar is trained as a probability of ray existence but also acts
as the amplitude of a whole-ray voxel correction. Increasing it helps some true
thin rays and can activate background. Bigger presence classifiers and stronger
preservation gradients have not demonstrated selective rescue under the guards.

The prior logistic rescue gate classified whether an erased raw-positive ray
was true foreground and then restored its raw mask. The proposed test instead
predicts a continuous renderer control and teaches acceptable voxel outcomes.
It must still beat the earlier controls; a new target is not evidence of success.

## Minimal architecture change

Freeze saved E's backbone, existing function head, fitted edges and beta.
Retain its original existence score for the fit. Add an initially zero residual
to the logit controlling occupancy in the renderer only. The added value must
not feed the quadratic fit, so increasing it has a monotone foreground effect
for this frozen geometry. Initial masks must exactly match E.

Use the existing 24-channel presence-probe architecture and the same image/
voxel/head inputs. Do not change width, pooling and supervision simultaneously.
The original existence score and the new correction control have different
semantics; the latter must not be reported as a calibrated existence probability.

## Proposed supervision

On fitting labels only, derive the range of renderer-control values that:

1. Preserves currently correct foreground voxels on a ray and provides at least
   one correct foreground voxel when the true ray is currently missed.
2. Does not activate a currently correct background voxel on that ray.

For fixed raw logits and geometry, each voxel has a monotone activation
threshold in the scalar control. Their relevant maxima/minima define lower
and upper bounds. A true thin ray can require a high control value even when
the useful change to its actual mask is only one voxel. An empty ray supplies
an upper bound; it must be explicitly included in fitting.

For a nonempty feasible interval, train the control toward that interval with
a small-change preference toward E's original value. For incompatible bounds,
retain E as the training action target and report infeasibility; do not force
a contradictory rescue target. Existing errors are not presumed correct:
ground truth defines protected foreground/background during fitting only.

This teaches a proposed minimum acceptable edit, rather than always pushing a
true ray's scalar toward one. The existing non-deployable minimum-presence
oracle found that 2,685/4,078 failing thin rays could regain one correct voxel
without an extra FP voxel on that ray. That establishes a useful feasible
subset, not that the new head can identify it without labels.

Infeasible bounds, score saturation, zero beta, borders and multi-run rays need
explicit numerical handling and tests. Loss normalization, the small-change
coefficient, bounds tolerance and sampling weights must be frozen before fitting.
This proposal is not a ready-to-launch preregistration until those are specified.

## Matched comparison and decision

- Original E/no edit.
- The same residual head trained with existence BCE, with identical inputs,
  fitting cases, initialization, optimizer and duration.
- The same residual head trained with the correction-interval objective.

Use three head-fitting seeds per existing fold; these do not constitute new
backbone seeds. Include true thin, thicker, hard-empty and other-empty rays.
Select solely on inner data with the existing E fallback, both FP guards,
thin-voxel nondegradation and the predefined 0.002 mm ASSD development margin.
Freeze calibration ranges before training and evaluate outer data only after
all selections are locked. All evidence remains internal development.

Primary success is increased correctly overlapping thin-ray recall while
preserving both FP burdens and thin-voxel recall. Also require fewer renderer
deletions of true voxels, inspect near-surface and deep changes, and report
case/fold harms and raw/final per-class Dice. Mean ASSD alone is insufficient.

If the new target simply raises true and empty scores together, or fails these
guards across seeds, it has not solved selective correction. If it improves
only fitting cases, the targets may be feasible but not learnable from the
available representations. Either result is more informative than another
small focal-weight adjustment or an unstructured width search.
