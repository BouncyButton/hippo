# Full fold-0 PCA coefficient refiner

Train the 32-mode anatomical correction regressor on all 208 training cases,
then evaluate the frozen model on all 52 validation cases. This extends the
eight-case fitting diagnostic without replacing learned validation evaluation
with annotation-guided oracle scores.

## Protocol fixed before training

- Same canonical Dice-only, no-augmentation, early-stopped baseline checkpoint,
  historical data, and fold-0 split; all three hashes checked.
- Fresh seed-0 CNN initialization; no continuation from the eight-case model.
- All 208 training cases provide CNN examples, anatomical PCA directions,
  mean-error offset, and annotation-guided coefficient targets. No validation
  subject contributes to any of those quantities.
- Same 32-mode signed-distance representation, prediction-centroid integer
  alignment into an 80-cubed canvas, +/-8 clipping, and two-step prediction bands.
- Same five inputs and architecture: MRI, foreground probability, source SDF/4
  clipped to [-1, 1], inner band, outer band; stride-2 convolutions with channels
  16/32/64/64, GroupNorm/GELU, spatial pooling to 4 cubed, 4096->128->32 head.
- Same annotation-guided target fitting: per-case band least squares with
  per-mode RMS normalization and ridge 0.001. CNN coefficient normalization
  uses the union of predicted bands across all 208 training cases.
- Same objective: mean per-case normalized oracle-field MSE, with denominator
  floor 0.0001, plus 0.05 whole-foreground soft Dice loss. No augmentation.
- Adam, initial lr 0.0003 cosine-decaying to 0.000003, gradient clip 5, FP32,
  no weight decay; 200 epochs, batch 8, 26 batches/epoch, 5,200 updates. Each
  epoch uses a fresh seed-0 permutation covering every training case once.
- Evaluate training metrics every five epochs, including epoch zero. Select
  the checkpoint with lowest training normalized field MSE. No validation
  early stopping, checkpoint selection, coefficient fitting, or tuning.
- Preserve labels outside the prediction bands and AP labels on retained
  foreground; newly added foreground receives its nearest original AP label.
- After training and independent training-output verification, freeze all
  artifacts and evaluate all 52 validation subjects. Annotation access is only
  for scoring after predictions are fixed. The refiner has no annotation input.
- Primary endpoint: mean foreground Dice excluding background, paired against
  the baseline. Secondary: ASSD, HD95, improved/worsened/unchanged subjects,
  errors fixed/introduced, and a paired subject bootstrap (10,000, seed 0).
- Save and independently verify masks, metrics, outside-band preservation, and
  retained AP labels. Confirm frozen-artifact hashes and canonical baseline Dice.

All computation, including unit checks and verification, runs in one `sbatch`
job. The baseline stays frozen. Its training predictions are in-sample; this
run tests the requested direct training setup, without out-of-fold retraining.
The model changes distance fields, with no claim of invertibility or guaranteed
topology preservation. Validation has been used for earlier experiments and is
not a final untouched test set.

## Execution

Cluster root: `/home/3160552/pca_refiner_fold0_208_20260930_01`.
Slurm job: 677536 (training, validation, and independent checks).
Launcher: `scripts/run_pca_refiner_all.sh`.
Training: `semantic_constraints/pca_refiner/fit_eight.py --all-training`.
Evaluation: `semantic_constraints/pca_refiner/evaluate_validation.py`.
The run contains a frozen source snapshot; its baseline checkpoint dependency
is in `/home/3160552/spectral_shape_fold0_20260930_01/`.

## Training result

All 200 epochs (5,200 updates) completed. The selected checkpoint was epoch
200, chosen solely by lowest training field MSE. Mean training foreground Dice
was 0.9383332743 at baseline, 0.9390402503 for the annotation-guided oracle,
and 0.9390431141 for the learned correction. Normalized oracle-field MSE was
6.0872e-6. The learned masks fixed 1,475 errors and introduced 407; 98 cases
improved, six worsened, and 104 were unchanged. The oracle fits distance fields
rather than maximizing hard Dice, so learned Dice can slightly exceed it.

Eight unit checks passed, including evaluation consistency across batch sizes.
Independent verification of all 208 saved training masks passed: no validation
overlap, accurate Dice/error counts, exact outside-band labels, and exact AP
labels on retained foreground. The baseline and oracle means reproduced the
previous 208-case teacher audit.

Training metrics, protocol, checkpoints, basis, masks, plot, and verification
are saved locally in `results_677536/`.

## Validation result

The complete pipeline finished successfully in Slurm job 677536 in 14 minutes
9 seconds (peak host memory approximately 8.1 GiB). The checkpoint, basis, and
normalization were frozen before evaluating the 52 validation cases.

| Metric, mean across 52 validation subjects | Baseline | Refiner trained on 208 |
| --- | ---: | ---: |
| Foreground Dice, background excluded | 0.9009775184 | 0.9009775184 |
| ASSD (mm) | 0.4214657223 | 0.4214657223 |
| HD95 (mm) | 1.1415282270 | 1.1415282270 |

All 52 masks were unchanged. There were zero improved or worsened cases,
zero errors fixed or introduced, and zero outside-band changes. The paired
Dice gain and its bootstrap interval were exactly zero for these masks.
Independent verification of all saved validation masks passed; frozen artifact
hashes were unchanged, and baseline Dice matched the canonical validation run.
Saved outputs are in `validation_677536/`.

This model fits the training targets but provides no hard-mask generalization
benefit on this validation set. Continuous predicted corrections did not cross
the label threshold. The result does not identify whether target choice,
representation, training/validation error differences, or overfitting is the
main cause, and does not disprove anatomical boundary refinement in general.

BeeGFS usage rose from 90.69 to 90.87 GiB out of 93.13 GiB; no cleanup was
needed. The baseline checkpoint was reused instead of uploaded again.

## Saved-field diagnostic

Job 677565 independently reconstructs validation outputs from their saved
coefficients and the frozen basis, and reports coefficient norms and signed
distance correction magnitudes. It does not train or rerun either network.
The initial request 677560 was pending behind other jobs and was replaced with
this smaller one-CPU, 2-GiB request. Source: `audit_saved_fields.py`; launcher:
`scripts/run_pca_field_audit.sh`.

Job 677565 completed successfully in 24 seconds. All 52 validation coefficient
vectors were nonzero and distinct. Their maximum absolute band SDF correction
was 0.44044635 mm, leaving a minimum margin of 0.05955365 mm to the label
threshold. Independent reconstruction from coefficients, basis, and mean-error
offset exactly reproduced all 52 saved masks. Thus the unchanged masks are
consistent with nonzero corrections smaller than the 0.5-mm minimum boundary
voxel distance. Coefficient norms (basis-dependent units) had median 17.58 on
validation versus 37.25 on training. Details: `validation_677536/field_audit.json`.

## Correction direction and precision audit

At the user's request, sbatch job 677578 checks saved corrections against
annotations on all 208 training and 52 validation cases. No networks are run
or retrained. It reconstructs the original masks as a consistency check,
measures correction direction on existing FP/FN voxels within the bands,
and evaluates the following predeclared modifications:

- Multiply the complete correction field by 1, 2, 4, or 8.
- Replace it by `c * sign(correction)` for c = 0.25, 0.6, 0.8, 1, 1.2, 1.6,
  or 2 mm. Zero remains zero. The grid avoids half-voxel constants that place
  grid-aligned boundary voxels exactly on the threshold.
- As a direction-only control, use the sign of the fixed training mean-error
  offset alone (unit magnitude).

For each modification, report mean foreground Dice, paired gains, case counts,
errors fixed/introduced, and edit precision = fixed / (fixed + introduced).
Precision is undefined if no voxel is changed. Direction accuracy on existing
errors is a different measure: positive correction on a false positive or
negative correction on a false negative. It does not penalize incorrect edits
to previously correct voxels; edit precision does.

A preferred modification is selected by mean training Dice before validation
is read; the complete validation sweep is descriptive and must not be treated
as an independent test set or used to claim an unbiased best setting. Also
report signed-distance residual alignment and magnitude quantiles. Source:
`semantic_constraints/pca_refiner/audit_correction_accuracy.py`; launcher:
`scripts/run_pca_correction_accuracy.sh`.

Job 677578 completed successfully in 1 minute 17 seconds. All 260 original
saved masks were reproduced from saved coefficients before amplification.
The detailed per-case analysis is in `results_677536/correction_accuracy.json`.

### Direction and actual edit precision

Among baseline errors within the bands, correction direction was right for
80.72% on training (68,417 / 84,756) and 59.17% on validation
(20,121 / 34,005). Validation direction was right for 80.34% of false positives
(remove excess foreground), but only 32.31% of false negatives (add missing
foreground). Thus these predictions do not reliably recover missing tissue.
These percentages consider existing errors only; they are not edit precision.

The original training corrections fixed 1,475 voxels and introduced 407 errors:
78.37% of their edits were correct. Original validation edit precision is
undefined because no labels changed. On validation, the median absolute
correction was 0.06608 mm, the 99th percentile was 0.25722 mm, and the maximum
was 0.44045 mm. Pooled signed-distance MSE improved slightly, 0.28756 ->
0.28313, and predicted/required field cosine was only 0.13367 (training 0.43314).
This is weak continuous alignment, not zero signal or useful hard-mask repair.

### Validation magnitude diagnostics

| Modification | Mean Dice | Fixed errors | New errors | Edit precision |
| --- | ---: | ---: | ---: | ---: |
| Original field | 0.900978 | 0 | 0 | Undefined |
| 2x field | 0.899853 | 333 | 623 | 34.83% |
| 4x field | 0.869833 | 4,645 | 13,531 | 25.56% |
| 8x field | 0.771333 | 11,168 | 47,686 | 18.98% |
| Constant +/-0.25 mm | 0.900978 | 0 | 0 | Undefined |
| Constant +/-0.6 or +/-0.8 mm | 0.760322 | 18,056 | 60,317 | 23.04% |
| Constant +/-1 or +/-1.2 mm | 0.621554 | 19,772 | 103,579 | 16.03% |
| Constant +/-1.6 or +/-2 mm | 0.500812 | 20,121 | 140,290 | 12.54% |

Different constants can give identical masks because voxel distances are
discrete. At 2x, 12 validation cases improved and 40 worsened; the paired mean
gain was -0.001125, with descriptive bootstrap interval [-0.001529, -0.000738].
All 52 worsened at 4x, 8x, and every tested sign constant above 0.5 mm.
At 2x, most edits removed tissue: 328 correct removals versus 606 incorrect
removals, and five correct additions versus 17 incorrect additions.

Selecting among these modifications on training Dice alone selected the
original field (1x). Even on training, 2x lowered mean Dice to 0.935569 and
the +/-0.6-mm rule lowered it to 0.804846. The fixed mean-error sign control
also performed poorly on validation (Dice 0.614826 at unit magnitude), close
to the learned unit-sign rule (0.621554).

Conclusion: the network often points in the useful direction at existing
errors, especially excess tissue, but also recommends changes at many correct
boundary voxels. Global amplification or a constant sign replacement does not
separate those locations. The current failure is therefore not explained by
insufficient magnitude alone. This does not establish which representation or
training change would solve it.
