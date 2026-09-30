# Correction-control experiment: completed, no model promoted

Follow-up completed: the [saved-head fitting/inner discrimination audit](CORRECTION_DISCRIMINATION_RESULTS_20260923.md)
evaluates whether these heads learn more than a constant renderer-control shift.

Job 666769 completed successfully in 14m03s. All 24 head fits completed:
two objectives × three head-fitting seeds × four existing folds. Every locked
selection retained original E. The scalar-bias control also retained E.

The correction-interval target rescued true foreground with a smaller FP cost
than existence BCE, but did not satisfy the prespecified requirement to preserve
E's false-positive burden. This is a negative primary result with a useful
diagnosis, not an improved segmentation model.

## What was tested

The [frozen protocol](CORRECTION_CONTROL_LOCK_20260923.md) kept E's backbone,
original head, geometry and renderer strength fixed. A width-24 residual head
changed only the renderer's presence-control logit, within ±4 logits. Original
E was reproduced exactly at initialization. Both objectives used the same
features, initial weights, case ordering, optimizer and training duration.

Existence BCE classified true versus empty rays. The interval objective learned
the smallest correction that retained currently correct foreground/background
and provided at least one correct voxel on a true ray, where feasible. Targets
used fitting labels only. Empty rays received explicit supervision. These
label-dependent teacher actions are non-deployable; only the learned head was
evaluated on inner/outer images.

All 24 selections were locked before outer inference. The selected methods are
therefore exactly E, with zero paired changes on all 208 outer cases. The table
below describes the prespecified **unselected** epoch-12, strength-1 outputs.
It cannot be used to reverse those selections.

## Outer diagnostic results

Means across cases and three head-fitting seeds; counts are per case.

| Metric | Original E | Existence BCE, unselected | Correction interval, unselected |
|---|---:|---:|---:|
| ASSD, mm | 0.483148 | 0.490045 | 0.483008 |
| Union Dice | 0.886133 | 0.885486 | 0.886233 |
| Anterior Dice | 0.864249 | 0.864522 | 0.864465 |
| Posterior Dice | 0.855504 | 0.853769 | 0.855430 |
| Correctly overlapping thin-ray recall | 55.5501% | 68.0716% | 57.5900% |
| Thin-voxel recall | 53.8431% | 65.6114% | 55.7477% |
| False-positive empty rays | 28.9423 | 43.7292 | 30.9038 |
| False-positive voxels | 393.1635 | 449.0913 | 401.5785 |
| Raw true voxels deleted by renderer | 120.0048 | 80.1939 | 112.7388 |
| True voxels recovered relative to E | 0 | 41.1186 | 7.4006 |
| Additional true voxels deleted relative to E | 0 | 0 | 0 |

Here thin means a true z-span of one or two voxels, excluding border-truncated
rays. Recall is averaged per case. Correct overlap requires a correctly placed
foreground voxel, not merely any prediction somewhere on the ray. These values
must not be substituted for earlier pooled counts or differently defined
thin-presence recalls.

The interval head gains 2.0399 percentage points of thin overlap and 1.9046
points of thin-voxel recall, while adding 1.9615 FP rays and 8.4151 FP voxels per
case. It creates no additional union-foreground deletions relative to E in these
evaluated outputs. This does not mean E's original deletions have disappeared,
or that all anatomical/subclass errors improve. The ASSD difference is only
−0.0001392 mm and does not establish improved boundary localization.

## Why every candidate failed selection

Each objective supplied 108 nonzero inner candidates: four folds × three seeds
× three checkpoints × three nonzero strengths. These are correlated candidates,
not 108 independent experiments.

| Inner diagnostic | Existence BCE | Interval target |
|---|---:|---:|
| Improved thin overlap | 108/108 | 108/108 |
| Increased FP rays | 108/108 | 108/108 |
| Increased FP voxels | 108/108 | 108/108 |
| Failed ASSD margin | 63/108 | 0/108 |
| Reduced thin-voxel recall | 0/108 | 0/108 |
| Increased renderer deletions | 0/108 | 0/108 |
| Eligible under all guards | 0/108 | 0/108 |

The FP guards alone rejected every interval candidate. The head did learn to
relax suppression enough to rescue anatomy, but did not do so selectively enough.
The unchanged protocol requires no increase in either mean FP burden; this
report does not relax that requirement after observing the result.

Outer interval-minus-E differences, averaging the three head-fitting seeds:

| Fold | ASSD, mm | Thin overlap, percentage points | FP rays | FP voxels |
|---|---:|---:|---:|---:|
| 1 | +0.000135 | +2.3184 | +2.3269 | +10.5000 |
| 2 | +0.000028 | +2.2753 | +2.2628 | +9.1603 |
| 3 | −0.000371 | +1.6763 | +1.7949 | +7.4551 |
| 4 | −0.000349 | +1.8897 | +1.4615 | +6.5449 |

All 12 interval fold/seed outputs increased both FP burdens and thin overlap.
ASSD improved in eight and worsened in four fold/seed outputs. Detailed
case-level harms for the unselected arms have not been summarized here; these
fold means are not evidence that every case benefits. Selected outputs have
zero case-level changes by construction.

## Architectural interpretation

**Observed:** mean interval correction was +0.6336 logits on true thin rays and
+0.6301 on hard-empty rays. BCE corrections were +3.9243 and +3.9272 respectively,
close to the +4 limit. The interval objective substantially reduced correction
magnitude but did not demonstrate selective rescue under the guards.

**Suggested, not proved:** much of the learned correction may act like a shared
positive shift. Similar group means do not prove constant predictions or lack
of within-group ranking. We still need correction distributions and a constant
control matched for mean shift/rescue before claiming that equivalence.

**Observed from the objective:** almost all empty-ray teacher actions are zero.
The objective protects currently correct background but does not reward removal
of existing FP voxels. Within the feasible interval, the zero-action preference
has weight 0.01; crossing the upper bound invokes the stronger interval penalty.
Empty rays are supervised, but this is a different negative signal from learning
active FP removal. Whether that asymmetry causes the failed discrimination is
an untested hypothesis.

Only 39.87–42.48% of initially missed fitting thin rays had a feasible rescue
under this bounded scalar action and the protection constraints. Counts are
1,033/2,500; 1,035/2,596; 1,073/2,526; and 1,146/2,752 by fold. These overlapping
fitting exposures are not independent cases. Infeasibility here can reflect the
±4 domain, fixed geometry, or conflicting voxel thresholds; it is not proof
that anatomy cannot be recovered by another architecture.

## Next discriminating test

First audit the saved heads on fitting and inner data: compare their correction
distributions, feasible-rescue ranking, interval violations and resulting masks
against a constant correction matched using fitting/inner data only. Include a
small fitting-set overfit check if the learned head cannot separate the groups
even on its own fitting examples. This separates optimization/representation
failure from generalization failure before changing capacity.

If that audit confirms a shared-shift failure, compare the unchanged interval
objective with a two-sided edit target that also rewards removal of existing FP
on empty rays, while keeping the architecture and positive rescue targets fixed.
Predefine the target, weighting and optimizer before fitting; reuse three head
seeds per fold and the same guards. The hypothesis is falsified if apparent
negative learning merely reduces all corrections and thin rescue vanishes, or
if selective rescue still fails either FP guard. This follow-up is proposed,
not implemented or launched by this report.

## Provenance and limits

Eight implementation tests passed, covering renderer parity, target feasibility,
protection constraints, loss gradients, initialization, guards and a training/
save-load smoke test. Source/data/split/checkpoint hashes were checked; E outer
ASSD/Dice reproduced the existing reference within the frozen tolerance. These
checks establish implementation consistency, not effectiveness.

Results remain internal development on the reused labeled pool and one original
backbone seed. Three head-fitting seeds vary the head training, not the original
segmentation training. No external, clinical or reliable edge-learning claim
follows. Head size, mean pooling, the bounded action, fixed geometry and 12-epoch
optimization remain limitations of this particular negative result.

Original models were left unchanged. New weights and individual records remain
on the cluster at `/home/3160552/correction_control_20260923_01a0cd/results_666769`.
Only non-identifying [aggregate results](CORRECTION_CONTROL_AGGREGATES_666769.json)
were retrieved. The report helper was extended after training to include
per-class Dice and recovered-voxel means; no training or selection code changed.
