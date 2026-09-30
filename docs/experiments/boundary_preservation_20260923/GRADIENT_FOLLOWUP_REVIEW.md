# Review of the signed-gradient findings and routing follow-up

Read-only analysis of the completed 16-batch diagnostic and proposed protocol.
No continuation arm was launched or modified by this review.

## What changes in the diagnosis

The sampled saved E outputs already receive favorable total-loss derivatives
on 430/448 erased thin-ray presence logits and all 734 true voxels in those
rays. Missing preservation supervision is therefore not a good explanation
at these sampled operating points. This does not describe every training step.

The shared-head response is a different measurement: other examples act
through the same parameters. It can oppose a particular ray's output gradient.
This is demonstrated for the tested Euclidean SGD direction with backbone
outputs held fixed. It does not establish the sign of an actual AdamW update,
or what caused the historical fold-4 outcome.

## Additional calculation from the saved batch records

For each of the 16 `fold*_batch*.json` records in
`experiments/presence_decisive_20260923/results_signed_gradients_v2_completed`,
select `full_head_first_order_response`, field `mean_presence_logit`.
Subtract the `seg_rendered` response from `total_existing_E` to obtain the
head-only first-order direction implied by H. Directional responses are linear
in loss gradients. This is a derived diagnostic, not a trained H result.

| Head SGD direction | Mean erased-thin response | Mean hard-empty response | Difference of responses | Batches increasing thin / empty |
|---|---:|---:|---:|---:|
| Existing J | -0.147436 | -0.153758 | +0.006323 | 8/16 / 8/16 |
| H: remove rendered-loss head gradient | +0.944443 | +0.944148 | +0.000296 | 16/16 / 16/16 |
| Diagnostic calibrated focal alone | +1.949513 | +1.951551 | -0.002038 | 16/16 / 16/16 |

Responses are logit change per unit plain-SGD learning rate, averaged equally
over batches. The difference is the change in the two stratum means, not
classification accuracy, calibration, or a ranking metric. All groups are fixed
at their pre-update definitions. The presence response has no beta-parameter
contribution, so freezing beta does not alter this particular subtraction.
Full-backbone and AdamW responses cannot be recovered by this subtraction.

Removing rendered feedback appears to release a broadly shared increase in
presence in this diagnostic. It does not yet indicate improved discrimination.
Nearly identical mean responses suggest a common shift, but cannot identify
whether the scalar bias, shared features, or another parameter group causes it.
They also do not show whether any ray crosses the final segmentation threshold.

The saved group definitions use any predicted foreground to call a ray present.
Of 981 retained thin-ray observations, 967 overlap the reference and 14 do not.
The erased group does not separately require that the raw foreground overlapped
the true voxels. Thus ray deletion and deletion of a previously correct voxel
are distinct measurements. The paired voxel-preservation audit measures the
latter directly; the follow-up protocol appropriately asks for overlapping
thin-ray recall.

## Assessment of the follow-up

J/H/V/R is a useful separation of mechanisms. In particular, the H implementation
uses detached parameter values in a functional head call; it preserves the
head's derivative with respect to raw logits and decoder features. Detaching
geometry would instead implement V. Explicit head supervision must still use
the ordinary trainable head. These are graph properties, not performance claims.

Before running the whole continuation screen:

1. Measure actual matched AdamW steps, with the protocol's optimizer reset,
   frozen beta, identical batches and stochastic state. Keep pre-step ray
   membership fixed when comparing signs. The historical training loop does
   not clip gradients; adding clipping would be another intervention.
2. Record thin-minus-empty mean logit response, score distributions, actual
   overlapping thin-ray retention, and both empty-ray and voxel FP burdens.
   A favorable thin mean alone is insufficient. Check retained and thicker
   anatomy as well.
3. Keep the scalar-presence-bias control: H must outperform a shared shift at
   the predefined FP limits to support improved discrimination. A single bias
   need not match both FP metrics exactly; predefine the feasible set and the
   inner-only tie-breaking rule.
4. If H still shifts both groups together, decompose the response by presence
   output bias/weights, stem, and feature projection before interpreting this
   as insufficient width. The saved aggregate responses cannot provide that
   decomposition.

The V helper currently relies on the default product renderer and tau=0.5.
Those match saved E; assert them in the eventual launcher, or pass the config
explicitly if the helper is generalized. Do not silently change renderer
settings in a gradient-routing comparison.

Keeping the explicit original-E fallback is appropriate. Report both candidate
outcomes and fallback frequency: selection of E can establish that the tested
candidate did not meet the guard, but cannot establish that every larger head
or every routing intervention is ineffective.

## Architecture implication

The strongest present design concern is that one shared presence score acts
both as an existence decision and as a whole-ray correction amplitude. The
next useful success criterion is better thin-versus-empty discrimination at
fixed FP burden. Extra capacity is valuable only if it produces that separation.
Separating existence supervision from correction strength remains a hypothesis;
H tests one gradient route and does not itself perform that architectural split.

Sources: `SIGNED_GRADIENT_RESULTS_20260923.md` and
`RENDERED_GRADIENT_FOLLOWUP_PROTOCOL_20260923.md` under
`docs/experiments/step_contour_20260922`, plus the diagnostic records and
`renderer_gradient_routes.py` in the experiment directory.
