# Frozen-head loss, activation and gradient audit

Written before execution. Audit the 24 final heads of job 666786 on exactly
their four fitting cases per fold (seven distinct cases across folds). No
training, checkpoint reselection, inner or outer evaluation.

Decompose the original case/group-balanced interval objective exactly into six
disjoint groups: initially missed feasible thin rays, initially missed infeasible
thin rays, initially overlapping thin rays, other true rays, hard-empty rays,
and other empty rays. Preserve original four-group normalization; splitting a
group for reporting must not increase its loss weight. Split each contribution
into lower hinge, upper hinge, tie and infeasible-zero terms. Assert the losses
and parameter gradients sum back to the unchanged original objective. Compare
paired state-input versus original-input contributions and zero-correction loss.

For currently unrescued feasible thin rays, report distance in residual logits
to the safe rescue lower bound, allowed interval width and the derivative of the
bounded tanh correction. Report fractions of gaps within .01,.1,.25,.5,1,2 logits.
These label-defined measurements are retrospective diagnostics, not features.

Report actual masks for unchanged learned corrections and fixed +0.25/+0.5-logit
increments, clipped to the original domain. Also evaluate a explicitly
non-deployable oracle: increase only currently unrescued feasible true thin rays
to their target lower bound, leaving every other learned correction unchanged.
Assert that this oracle adds no FP voxel or true deletion relative to the learned
mask. It quantifies available selective rescue, not a deployable result or the
share of total ASSD attributable to thin rays. No ASSD is recomputed.

At each saved head, compute gradients for each weighted group contribution,
their norms and pairwise cosines. Compute derivatives of the mean correction
on currently unrescued feasible thin rays, hard-empty rays and already-overlapping
thin rays, pooling each receiver's rays over the four cases. Report the signed
first-order response to minus the total gradient normalized to unit parameter
L2 norm, and each loss group's additive contribution to that same direction.
Positive response means increased mean correction. Direct loss gradients with
respect to correction are reported separately. This is a local SGD diagnostic,
not an AdamW update, historical causal reconstruction or training forecast.
Saved optimizer states are not available and no parameter update is performed.

Numerical amendment after job 666811 stopped on a componentwise float32
gradient check: record reconstruction L2 error relative to the sum of subgroup
norms and the total norm. Require error <= 1e-5 times subgroup-norm sum + 1e-7.
Use the directly computed original-objective gradient for the total direction;
suppress signed responses if its norm is not at least ten times reconstruction
error. Record maximum absolute error as well. This changes diagnostic numerical
validation only, not model weights, loss definitions or subgroup weighting.

Discriminating findings:

- Concentrated loss reduction on feasible missed thin rays supports improvement
  on the intended target, but must be checked against actual threshold crossings.
- Small gaps with unsaturated correction suggest threshold placement is an
  actionable issue; large gaps refute a merely numerical near-threshold account.
- Opposing group gradients and adverse shared-parameter response despite the
  desired direct output signal support local gradient interference.
- Saturation, zero gradients or missing signal suggest different mechanisms.
  None alone establishes the cause of earlier training failures.

Report all heads, paired fold/seed differences and heterogeneous signs. Verify
source/input/checkpoint hashes and reproduce saved fitting losses. Export only
non-identifying aggregates under the user's authorization for current and future
aggregate reports in this investigation. Keep images and individual case records
on the cluster; preserve every saved model.
