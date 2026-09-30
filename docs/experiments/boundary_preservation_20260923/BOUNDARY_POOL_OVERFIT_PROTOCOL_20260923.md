# Frozen comparison: whole-ray versus boundary-local pooling

Written before fitting. Hypothesis: whole-ray mean pooling discards useful local
boundary evidence and contributes to the strongly coupled corrections observed
in the prior audit. Boundary-local pooling may enable more selective corrections.
This comparison tests that specific intervention, not all pooling choices or
the general adequacy of the backbone.

Use exactly the four fitting cases per existing fold from job 666786: seven
distinct cases, 16 case/fold exposures. Three head-fitting seeds, two arms,
24 fits. No inner/outer evaluation, new data selection or checkpoint selection.
Original E and all prior checkpoints remain unchanged.

Both arms receive the same 13 renderer-state input channels and have width 24
and 8,185 trainable parameters. Both use the same two-layer spatial stem and
48-channel presence readout. The only intervention is the reduction over z:

- Mean: concatenate two copies of the whole-ray hidden-feature mean.
- Boundary: concatenate hidden features pooled around the frozen lower boundary
  and around the frozen upper boundary. Each pool uses weights proportional to
  `exp(-0.5*((z-boundary)/1.0)^2)`, normalized over the available z coordinates.
  Sigma is fixed at one voxel. Softmax computes normalization stably, including
  border cases. No label defines pool centers or weights.

The normalized weights are passed as two auxiliary tensor channels for caching
and batching; the convolutional stem sees only the original 13 feature channels.
The mean arm ignores the auxiliary weights. Both trainable and frozen initial
branches use their arm's pooling. Initial-branch subtraction must produce exact
zero correction and E's unchanged masks at initialization for both arms.

Copy identical original weights into the two arms. Preserve the previous
readout initialization: first 24-channel block contains the original weights,
the second block starts at zero and remains trainable. Thus the boundary arm
initially weights its lower pool in the readout while upper-block weights can
learn; do not add a symmetric initialization intervention. Identical initial
outputs do not imply identical parameter Jacobians. Pooling also changes feature
scales and optimization conditioning; outcomes do not isolate information
retention from those effects.

Keep the frozen geometry, beta, product renderer, +/-4 bounded correction,
interval targets, group normalization and 0.01 tie term unchanged. AdamW
lr=3e-4, weight_decay=0, default betas/epsilon; identical case permutations
between arms per seed. Exactly 1,024 case updates, with diagnostic checkpoints
at 0,128,512,1024. The final step is primary, without tuning sigma, learning rate,
strength or duration after results. The mean control is freshly fitted to avoid
relying on historical optimizer execution. Record any reproduction differences.

Report fitting loss versus the best constant and matched mean arm, correction
AUROC, thin overlap/voxel recall, union Dice, true-voxel recoveries/deletions and
both FP burdens. A fitting success requires increased thin overlap relative to
E, no decline in thin-voxel recall, and no increase in mean FP rays, FP voxels or
true deletions. Lower loss/AUC alone is insufficient. No ASSD improvement is
claimed in this fitting-only diagnostic. Report every fold/seed and all harms.

At each final head, run the unchanged loss-flow audit from job 666826. Verify
loss reconstruction; report subgroup contributions, activation gaps, bounded
correction derivatives, group-gradient cosines and local SGD responses. These
are diagnostics, not AdamW reconstructions. Poor numerical reconstruction must
not be interpreted as a reliable gradient direction.

Evidence favoring the pooling hypothesis requires improved actual selective
recovery alongside less coupled responses, not just less opposed gradients.
Equal or worse selective fitting weakens this particular pooling intervention.
Better fitting without validation does not establish generalization. An apparent
failure after 1,024 updates does not prove an intrinsic representational limit.

Tests must verify label-free, normalized and localized pools; initial equality;
identical parameter counts/weights; mean-control parity with the existing state
head; two-arm fitting/save and final-audit execution; and unchanged original
weights. Verify source, input and checkpoint hashes. Save new weights and any
individual records on the cluster. Export only non-identifying aggregates under
the user's explicit authorization for this investigation.
