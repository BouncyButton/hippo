# Boundary generalization: controlled results

The experiments identify limited training diversity and residual input-coordinate sensitivity as contributors to the validation boundary gap. Mild augmentation is the replicated training remedy in this regime. The tested extra boundary losses and symmetric PCGrad do not provide a convincing additional benefit. The remaining error has not been shown to be an irreducible annotation or partial-volume floor.

## Replicated practical remedy

Use ordinary Dice training with the existing `mild_v1` augmentation and the tested early-stopping policy. On the fixed 50-case training subset and 52 validation cases:

| Initialization | Validation Dice, control → augmented | Dice gain | Unique shell errors, control → augmented | Error reduction | Cases with fewer shell errors |
|---|---:|---:|---:|---:|---:|
| Seed 0 | 86.2016% → 87.3730% | +1.1714 pp | 38,972 → 34,629 | 11.14% | 48/52 |
| Seed 1 | 86.0414% → 87.4333% | +1.3919 pp | 39,058 → 34,046 | 12.83% | 50/52 |

Both false positives and false negatives decrease in both initializations, and mean surface distance improves. Fixed-pass 30 and pass 60 comparisons also favor augmentation. The equal-seed mean gain is 1.2816 Dice points and 4,677.5 fewer shell errors per evaluation of the same 52 cases. These are two model initializations, not 104 independent validation cases.

This is specifically an outer-boundary improvement. A/P swaps increase by 371 and 439 voxels, with descriptive paired-case intervals spanning zero. Macro Dice improves in 41/52 and 31/52 cases. Do not claim that augmentation fixes the A/P partition or improves every image.

The tested recipe uses AdamW at initial LR 1e-4, weight decay 1e-5, the same ReduceLROnPlateau policy, batch size 1, and 50 updates per pass. Early stopping uses minimum 60 passes, patience 8 and minimum improvement 0.0005, with maximum 75; all seven new training runs stopped at 60. The diversity replay was additionally capped at the matched 60-pass budget. `mild_v1` combines bounded affine transformations with mild contrast, noise and blur, retains identity views, uses nearest-neighbor labels, and guards against foreground clipping. The contribution of each component was not separately identified.

## What caused part of the gap?

**Training diversity, beyond extra optimizer steps.** Repeating each of the original ten cases five times per pass gives exactly the same 3,000-update budget, actual LR trajectory and auxiliary warmup as the 50-case summed control. Selected validation Dice is 81.031% versus 86.249%; shell errors are 54,144 versus 39,077. Fifty distinct cases remove 15,067 errors (27.83%), improving shell counts and surface distance in all 52 validation cases. The advantage persists at every measured fixed budget. On the same ten training cases, the larger-data model fits less closely. Greater diversity and less repeated exposure per case change together in this intervention; this is one seed and one nested subset.

**Coordinate sensitivity.** Exact integer shifts preserve image content and label support in all tested cases, yet aligned model predictions change. On seed 0 validation cases, mild augmentation reduces mean boundary disagreement across shifted views from 303 to 168 voxels/case for Dice, and from 296 to 173 for the summed objective. A fixed average of identity plus twelve shifted views provides an additional inference treatment. It also replicates after augmentation: seed 0 improves from 87.3727% to 87.7181% Dice and removes another 632 shell errors; seed 1 improves from 87.4337% to 87.8747% and removes another 756. These are additional reductions of 1.82% and 2.22% relative to augmented identity inference. Use this fixed 13-view average when the extra inference cost is acceptable; it is distinct from learning another constraint. No input or label support was clipped in any of the six model audits, and repeated identity inference drifted by at most two shell voxels per case.

**The tested optimizer projection is unhelpful.** The matched 50-case ordinary sum reaches 86.249% validation Dice versus 85.353% for PCGrad and removes 1,658 shell errors. All 60 data orders match, with no skipped updates. The ordinary sum is already better at pass 29 while actual LRs still match; the policies diverge at pass 30. The selected-checkpoint effect includes that downstream LR response. Small auxiliary weights did not make projection a small perturbation of the total gradient.

**Extra boundary losses do not close the gap at the tested weights.** In the seed 0 factorial experiment, sum versus Dice adds only 0.047 Dice points and 105 shell errors. With augmentation, it adds 0.061 Dice points and 58 errors. Descriptive paired-case intervals span either direction. Auxiliary coefficients were fixed; these results do not rule out every alternative weight or formulation. Local constraints can be satisfied on memorized training images without locating the correct surface on new cases.

## Why approximately 36,000 outer errors versus 4,000 A/P swaps?

The original CSV sums exactly to 35,909 foreground/background errors plus 3,872 A/P swaps. These are distinct wrong voxels over 52 cases, not duplicated edge incidences: means 690.56 and 74.46 per case. Union errors occur in all 52 cases; A/P swaps in 49. The top five cases account for 15.66% and 35.18%, respectively.

The outer surface is much larger: the authoritative validation masks contain 139,916 outer faces versus 4,108 A/P faces, a 34.06× ratio. This gives geometric context for raw error totals. Faces and wrong voxels have different units, so their quotient is not a voxel error rate.

Exact crossing correctness is also unusually strict: both endpoints must represent the precise foreground/background transition. The selected 50-case Dice control scores 42.02% exact crossing correctness while its voxel-based surface Dice within one model-grid voxel is 94.18%, with mean symmetric surface distance 0.483 grid voxels. Those metrics describe different tolerances. The low exact score does not erase the 38,972 actual shell errors, and grid voxels are not millimeters.

## Evidence limits and preservation

All evaluation uses the same reused fold 0 development cohort. The new augmentation replication changes initialization, not subjects. The historical 208-case augmentation study supports the direction across three seeds, with exact source matching for seeds 1/2 and an older snapshot for seed 0. No independent held-out cohort or repeated-annotator experiment was run. Case bootstrap intervals omit training-seed, selection and possible subject-dependence uncertainty. The results establish tested contributing mechanisms, not a complete explanation of every remaining error.

All MedSAM3 runs and results were untouched. Six eligible old non-MedSAM checkpoint copies were archived locally and SHA256-verified before their remote copies were retired; the archives and scientific records remain available. Selected and final checkpoints from the new experiments remain on the cluster. A stale BeeGFS quota snapshot stopped the second-seed job before its augmented arm began; the unchanged retry reused the completed control and kept the storage guard. No additional cleanup was needed.

## Reproducible evidence

- [Detailed interpretation](INTERPRETATION.md), [original count reconstruction](COUNT_RECONCILIATION.md), and [pilot factorial results](CURRENT_RESULTS.md).
- [Diversity intervention](../boundary_diversity_20260929/REPORT.md) and [two-seed augmentation replication](../boundary_replication_20260929/REPORT.md).
- [Seed0 translation audit](../boundary_translation_20260929/REPORT.md) and [seed 1 translation replication](../boundary_translation_replication_20260929/REPORT.md).
- [Final verification manifest](VALIDATION.json) confirms seven completed training models, 21,000 optimizer updates with zero AMP skips, six completed 102-case inference audits, matching checkpoint/data bindings, and six intact archived checkpoints.
- Frozen training source, configs, per-pass order/LR records, case metrics and checkpoint hashes are retained under the corresponding `reports/boundary_*_20260929` directories. All model inference used the original CUDA runtime; local Python was used for statistics, synthetic checks and figures.
- [Summary figure](causal_summary.png), [exportable figure PDF](causal_summary.pdf), and [full pilot trajectories](trajectories.png). The summary's inference panels use repeated identity inference and can differ by one or two voxels from the original selected-checkpoint audit.

