# Frozen fitting-only renderer-state overfit comparison

Written before training. This is a diagnostic of learning on fitting examples,
not a candidate model-selection or generalization experiment.

Use the first four sorted fitting case identifiers from each existing fold.
Do not choose cases by performance, rescue feasibility or observed loss. Record
the chosen identifiers only on the cluster and export a subset hash/count.
No inner or outer images are inferred or evaluated. Freeze original E entirely.

Fit two matched arms from E's original residual initialization, with width 24
and unchanged z mean pooling. Both allocate 13 stem input channels, the same
weights and exactly zero initial correction. Copy the original seven-channel
weights and initialize six additional input-channel weights to zero. The
original-input control receives zeros in those six channels. The state arm
receives frozen, prediction-only channels:

1. Original presence-logit anchor / 8, broadcast along z.
2. Signed lower-edge offset `(z-lower+0.5)/4`.
3. Signed upper-edge offset `(upper-z+0.5)/4`.
4. Raw three-class argmax margin `(max(fg logits)-bg logit)/8`.
5. E-rendered three-class argmax margin / 8.
6. Renderer beta / 2, broadcast spatially.

Clamp all six channels to [-4,4]. Ground-truth labels, feasible intervals and
ray groups never construct input features. Both arms have identical allocated
parameter counts; extra-channel weights are inactive in the zero-input control.
Thus this tests access to renderer information through additional active input
connections, not a perfectly equal effective-capacity claim. Width is unchanged.

Keep the existing bounded correction, product renderer, fixed geometry and
fitting-only interval target/loss including group balancing and 0.01 tie term.
For this overfit diagnostic use unscaled interval loss, AdamW lr=3e-4,
weight_decay=0 and default betas/epsilon. These settings are identical between
arms but differ from the earlier full fitting experiment; do not attribute a
cross-experiment change solely to input features.

Three head-fitting seeds per fold, one full cached case per update, matching
case permutations between arms. Run exactly 1,024 updates (256 passes over four
cases), reporting steps 0,128,512,1024. Step 1024 is primary; no checkpoint
selection, early stopping, learning-rate sweep or adaptive extension. There
are 24 fits. Original E weights and prior head checkpoints remain unchanged.

Primary diagnostic: learned fitting objective versus the representable best
constant and paired state-versus-control objectives at step 1024. Report all
fold/seed outcomes and intermediate curves. Meaningful selective fitting also
requires increased thin overlap/voxel recall without increased mean FP rays,
FP voxels or additional true-foreground deletions relative to E. Report
correction distributions, rescue-relevant AUROC, feasible-ray interval
violations, actual recovered/deleted true voxels and both FP burdens. AUC alone
is not a success criterion. Constants are optimized on the same four cases;
they are diagnostic comparators, not deployable selections. ASSD is not the
target of this fitting-only test and is not recomputed.

Predictions and falsification:

- If explicit-state inputs are limiting, the state arm should fit the interval
  target and selective corrections materially better than its matched control.
  Equal failure weakens this specific feature-access explanation.
- If original inputs already suffice for fitting under longer optimization,
  the control should beat its constant and learn selective edits too. This
  weakens the claim that adding renderer-state inputs is necessary.
- If objectives fall but FP/retention tradeoffs remain, fitting the current
  objective is insufficient; do not promote either arm on loss alone.
- Failure after 1,024 updates does not prove that either function class cannot
  fit the examples. Optimization, target conflicts, feature normalization and
  pooling still need discrimination before a capacity claim.

Verify feature construction, initial equality, matched parameter counts,
nonzero gradient access through state channels, and unchanged E in tests.
Write only new small-head weights and numerical results on the cluster. Export
non-identifying aggregates only, following the user's authorized audit-report
workflow. Any future held-out evaluation needs a separately frozen protocol.
