# Signed face-contrast experiment

Requested after degree weighting failed to reduce boundary errors. Test the
proposed probability-space edge consistency on the existing network output;
no auxiliary head and no degree-dependent voxel weighting.

For whole hippocampus foreground y and p=p_A+p_P, penalize
`((p_i-p_j)-(y_i-y_j))**2` on face-adjacent pairs with **both** endpoints in
the union of the existing two-step inner and outer bands. Positive-axis faces
are counted once, with an unweighted mean over faces per case and an equal
mean over valid cases. GT geometry is recomputed after augmentation. Dice and
ordinary bands remain the base objective; degree weighting stays disabled.

Stage 1 is a training-only diagnostic, not another validation sweep. Use the
32 training cases and deterministic mild_v1 views already registered in the
degree-band calibration, with input, split and checkpoint hashes checked.
Calibrate edge lambda from the existing epoch-five Dice checkpoint to median
edge/base logit-gradient RMS ratio 0.1, capped at p95 ratio 0.5. Base means
Dice plus ordinary bands, retaining the historical bands coefficient exactly.
No new baseline or calibration-source training is performed.

At the existing converged early-stopped ordinary-bands checkpoint, measure
raw gradient norms, edge/base gradient cosine, first-order directions on
missed/extra foreground, and paired logit interventions. Update sizes 0.25
and 1.0 use the same base-gradient maximum for base and base+lambda*edge;
size 4.0 is a stress diagnostic only. All interventions are temporary tensor
calculations; no model parameters are changed.

Predeclared gate for a subsequent controlled training run:

- All gradients finite and nonzero; mean gradient cosine nonnegative.
- At both small steps, combined loss causes no increase over the base update
  in either pooled inner FN or outer FP, and no increase in total class errors.
- At least one small step strictly reduces pooled inner-FN plus outer-FP.

This is a conservative screening rule, not a proof that a failed candidate
could never help training, or that a passing candidate will generalize.
It specifically avoids accepting the missed-foreground/false-positive trade
observed with degree weighting. Failure stops promotion under this protocol;
do not switch loss formulations or tune lambda based on the outcome.

If it passes, train one seed-0 fold-0 Dice+ordinary-bands+edge model with the
existing augmentation and optimizer settings, using the same early stopping
(patience 8, minimum 25 epochs, delta 0.0005, maximum 50). Reuse the existing
Dice and Dice+bands controls, and assess paired boundary corrections on the
same validation cases. Record that the development fold has already been used.

## Promotion after the diagnostic

Diagnostic job 667752 passed. Fixed edge coefficient: 0.03137547415201203;
ordinary-bands coefficient: 0.01114736174580409. The new training wrapper
reuses the frozen model/data/optimizer/augmentation/checkpoint utilities.
It disables diagnostic telemetry while preserving the validation metric.
The same job audits all three selected checkpoints with identical inference,
including paired corrections/introductions, inner FN, outer FP, AP swaps,
degree strata, surface metrics and case-bootstrap intervals. Controls are reused.
