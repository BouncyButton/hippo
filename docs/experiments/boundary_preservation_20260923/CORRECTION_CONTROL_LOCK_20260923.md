# Frozen correction-control experiment

Locked before fitting either new arm. Internal development only; the existing
four folds and one trained E backbone seed have informed method development.

## Matched arms and fixed representation

Compare original E, a selected scalar-bias control, a residual head trained with
balanced existence BCE, and an identical residual head trained with correction
intervals. Freeze E's backbone, original head, fitted lower/upper edges, and
beta. Fit 24 heads: two objectives x three head-fitting seeds x four folds.

Use the existing PresenceProbe with width 24, mean pooling and its exact
initial-branch subtraction. Inputs are E's cached seven stem-input channels
and original presence-logit anchor. Targets never enter the input features.
The residual acts only in rendering, not in fitting geometry. The correction
control is not interpreted as a calibrated probability of existence.

Bound the residual to +/-4 logits with 4*tanh(residual/4). Original E is
reproduced exactly at initialization, checked on every inner case. Geometry,
temperature 0.5, product occupancy and beta are identical in all arms.

## Fitting target and loss

Use 166 fitting cases per fold; 42 inner cases select checkpoints; 52 outer
cases per fold remain unevaluated until every one of the 24 selections is locked.
Use the existing source split and preprocessing, inner seed 0.

For fitting rays, retain all currently correct foreground union voxels and all
currently correct background voxels, and require at least one true foreground
voxel on every true ray. With raw logits/geometry fixed, bisection finds the
allowed residual interval within [-4,4]. Use 24 iterations and a 0.001-logit
inward safety margin on finite activation boundaries. Match the actual
float32 renderer and background-wins-ties argmax operation order.

The target action is the projection of zero onto a feasible interval. For
incompatible or numerically unsafe constraints, the target is zero/no change.
Verify the actual teacher mask preserves protected FG/BG. Include border and
multi-run rays without assuming that their intervals are feasible. Record
feasibility and missed-ray counts by stratum. Empty foreground and beta=0 are
handled by the same explicit decision predicates.

For feasible rays, loss is squared distance outside the interval plus
0.01*(residual-target_action)^2. The second term favors the smallest valid edit;
it does not penalize distance to zero when zero would fail the rescue target.
For infeasible rays, loss is residual^2. Average within each nonempty fixed
group (thin, other true, hard empty, other empty), then equally across groups.
The BCE arm uses the same four-group balancing with true existence targets.

Calibrate one positive constant per fold so the interval objective's initial
gradient RMS with respect to output residuals matches BCE's RMS on the first
four sorted fitting cases. Fix it for all seeds/epochs. Abort if either RMS is
zero/nonfinite. This matches one local output scale, not all parameter-gradient
directions or the optimizer trajectory.

## Training and selection

Three head-fitting seeds (0,1,2), identical initialization and case ordering
per seed across objectives. Twelve epochs, one full cached case per step,
float32 AdamW lr=1e-4, weight_decay=1e-5, default betas/eps, no scheduler,
no clipping, no focal term. Check finite losses and gradients.

At epochs 4,8,12 evaluate fixed residual strengths {0,0.25,0.5,1} on inner data.
Strength 0 is original E. A separate scalar-control grid {-4,-2,-1,-0.5,0,
0.5,1,2,4} is selected on inner data before head training. These ranges are
not changed after inspecting results.

Eligibility relative to E requires no increase in mean empty-ray FP, FP voxels,
or renderer-deleted true voxels; no decrease in mean thin-voxel recall; and mean
ASSD at most 0.002 mm worse. This is a development guard, not a clinical margin.
Rank eligible checkpoints by descending correctly overlapping thin-ray recall,
then ascending ASSD, FP rays, epoch and residual strength. E is the explicit
fallback. Bias ties favor smaller absolute shifts, then the signed offset.

Save all selections/hashes across all folds before any outer inference. On
outer data report selected arms and final-epoch strength-1 arms; the latter
are prespecified unselected diagnostics, never used to revise selections.
Report per-class Dice, union Dice, ASSD, thin overlap/voxel recall, both FP
burdens, renderer deletions, newly deleted/recovered true voxels and deep FN.
Also report fold/seed heterogeneity and every fallback selection.

Pair patients after averaging the three head-fitting seeds, not 624 independent
cases. Case-bootstrap intervals are conditional on the original E models and
do not include prior model selection or independent backbone-seed variability.

## Success and failure

Evidence favoring the new target requires better thin retention with both FP
burdens controlled, compared with E and the matched BCE head. Lower ASSD alone,
better fitting losses, or feasible label-dependent teacher actions are not
sufficient. A failure across seeds weakens this target/representation/domain
combination; it does not prove that all boundary-function architectures fail.

Verify source/data/split/checkpoint/code hashes. Save only new small-head
weights and numerical results on the cluster; leave original E checkpoints
unchanged. Upload code/protocol only. Retrieve only non-identifying aggregates;
individual case records remain on the cluster under the current transfer limit.
