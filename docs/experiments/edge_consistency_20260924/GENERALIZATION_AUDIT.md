# Train/validation error audit

Job **667813 COMPLETED on gnode02** in 3m52s, exit 0, no stderr.
Results: GENERALIZATION_REPORT.md and GENERALIZATION_RESULTS.json. Training
fitting improved, but no clear overall validation improvement was observed.
Pre-submission quota had 4.21 GiB free; no cleanup needed.
Submitted on `gpu:3g.40gb:1`, four CPUs, 24 GiB RAM, 30-minute
limit. Remote root:
`/mnt/beegfsstudents/home/3160552/edge_generalization_audit_20260924_01`.
Five audit tests passed; shell syntax and whitespace checks passed. Frozen
archive SHA256: `d78f093839ac8963a6092fb930df159ac543a9dbe863b9699fbdaae03438ac27`.
The initial submission with an afterok dependency returned exit 1. Verified
no duplicate was queued and training was complete, then submitted without
that dependency. The audit itself requires the completed training manifest.

Inference-only follow-up to completed training job 667767 (selected epoch 26,
stopped epoch 32). No new training, coefficient tuning or checkpoint selection.
Reuse the selected early-stopped Dice and ordinary-bands models.

Evaluate all 208 training and 52 validation cases with identical unaugmented
64^3 padding, model.eval(), batch size one and CUDA AMP. Validate dataset/split,
runtime, model-source and selected-checkpoint/export bindings before inference.
Store per-case counts on the cluster; no new dense image/prediction exports.

For each model and split report macro/whole/anterior/posterior Dice, inner FN,
outer FP, AP swaps, errors inside/outside the two-voxel bands, immediate/second
layers, degree strata and surface metrics. Pair each edge prediction with both
controls to count errors corrected and introduced. Existing surface-distance
metrics use the MSD grid's assumed 1 mm spacing.

Primary generalization comparison: change in mean per-case balanced boundary
error (equal contribution from inner-FN rate and outer-FP rate), alongside
both constituent rates and macro Dice. Report mean errors per case and pooled
counts within each split, but never compare raw pooled train and validation
counts because their sizes differ fourfold.

Compute the candidate-minus-control effect separately within each split, then
validation effect minus training effect. Bootstrap cases 10,000 times, paired
across models and independently sampled across the distinct splits. Positive
effect gaps for error metrics indicate less benefit/more harm on validation;
the direction reverses for Dice. Flag point-estimate training-only improvement
descriptively; this is not proof of overfitting or an automatic significance test.

Validation was used for early stopping and previous development. Intervals
exclude checkpoint-selection and seed uncertainty. These selected-checkpoint
results diagnose fitting/generalization differences, not the complete learning
trajectory and not an untouched test-set result.
