# Supervised A/P cut posterior

`APCutPosteriorLoss` is an experimental **supervised structural auxiliary**. It uses
the annotated A/P cut during training. It is not an unsupervised anatomical prior,
an inference decoder, or a demonstrated Dice improvement.

## Formulation

The explicit `axis` is an index in the input tensor's three spatial dimensions;
`anterior_low` states whether anterior label 1 occupies its lower coordinate side.
The module cannot infer anatomical orientation from a tensor. Callers must check
NIfTI affines and any axis permutations or spatial transforms before configuring it.

For each GT-foreground voxel, form conditional log probabilities
`ell = log_softmax(logits[:, 1:3], dim=1)` in FP32 (FP64 if supplied). Average each
class's log probability over the foreground voxels in each occupied slice. A cut
`k` assigns slices `j <= k` to the configured low-side class and `j > k` to the
high-side class. Its score is the mean of these slice scores over all occupied
slices. The posterior is `softmax(scores / temperature)` over **all** integer cuts
between the first and last occupied slices. The loss is the negative log posterior
of the GT cut, averaged over valid cases.

Both per-slice normalization and the outer mean are deliberate. Large cross
sections do not dominate, and `temperature` applies to mean-per-slice scores, not
voxel totals. These choices define a structured score distribution, not a calibrated
Bayesian cut-position posterior. Candidate temperature is explicit and must be
frozen before evaluation. GT cut correctness, posterior confidence, and segmentation
Dice are separate outcomes.

The candidate set uses detached GT foreground extent; it is never cropped around
the answer or selected from the model's top-k positions. No predicted foreground
mask or background logit enters the score. Direct background-channel and
outside-GT-support gradients are zero. This does not guarantee unchanged outer
foreground predictions after an optimizer update: A/P logit differences affect the
full softmax, and network parameters are shared. FP/FN/union geometry still need
evaluation.

Stable logit-space computation gives a corrective gradient at a confidently wrong
cut. It does not require a previously unsaturated probability field. It also does
not prove that network training will generalize or locate the correct slice.

## Geometry rejection and current evidence

The default `invalid_policy="error"` rejects missing A/P classes, mixed-class
foreground slices, incorrect configured orientation, multiple transitions, and an
empty-slice gap at the GT transition (which would leave the exact cut ambiguous).
`invalid_policy="skip"` is explicitly available for audits or a separately specified
partial-cohort experiment. It reports skipped cases and their reasons; an all-skipped
batch returns a finite differentiable zero. Skipping cases changes the training
population and does not establish a universal plane convention.

**The strict candidate fails the current raw-data geometry prerequisite.** On the
2026-09-05 fold-0 raw NIfTI audit, only 161/208 training and 41/52 validation cases
admit an exact unique axis-1 plane. The remaining 47 training and 11 validation
cases have both labels in one or two coronal slices. Their best planar assignment
disagrees with 1,104 training and 284 validation foreground voxels, respectively.
All 260 stored NIfTI sform linear parts are identity; the compatible orientation is
`axis=1, anterior_low=False` (anterior on the high-coordinate side).

All 260 arrays matched the earlier project's NIfTI parser. Direct byte-offset
checks confirmed both labels on each mixed slice. All 52 validation GT exports in
each of the baseline and translation error-map directories equal these arrays
after symmetric 64-cube padding/cropping. The serialized training pickle has not
been loaded in this local environment, so exact train-pickle correspondence remains
unverified. See the machine-readable evidence and reproducible audit at
`experiments/loss_constraint_followup_20260905/gt_geometry_audit.json` and
`audit_gt_geometry.py`.

Do not launch the strict all-case candidate on this dataset. Do not silently turn
these annotations into planes. A future tolerance-aware formulation would need
its own target definition and pre-registered comparison.

## API and validation

```python
from thesis.new_constraints.ap_cut import (
    APCutConfig, APCutPosteriorLoss, compute_ap_cut_posterior,
)

objective = APCutPosteriorLoss(axis=1, anterior_low=False, temperature=1.0)
result = objective(logits, labels)  # ConstraintResult, strict geometry validation
posteriors = compute_ap_cut_posterior(
    logits, labels, APCutConfig(1, False, temperature=1.0, invalid_policy="skip")
)  # Inspect .valid, .reason, .cut_indices, .scores, .log_probs, .gt_cut.
```

Diagnostics include GT-cut probability, absolute cut error in voxel slices,
exact-cut fraction, valid/skipped patient fractions, and per-case invalid reasons.
An argmax tie uses the first candidate; use the complete scores when auditing ties.

```bash
rtk proxy /Users/filippofocaccia/anaconda3/bin/python3 -m pytest \
  thesis/new_constraints/ap_cut/test_posterior.py -q
```

Tests cover all spatial axes and both orientations; mixed, reversed, missing,
multi-transition and ambiguous-gap GT; posterior normalization; temperature;
slice-area/padding invariance; zero background/support gradients; label detachment;
case averaging; finite all-skipped backward; FP16 stability; and corrective,
argmax-changing updates on saturated wrong synthetic logits. The large synthetic
logit update demonstrates mathematical reachability only, not a useful learning
rate or measured training gain.
