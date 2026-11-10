# Why bands transfer poorly to the MNI transformers

7 November 2026. Diagnostic analysis only; no training or recipe changes.

## Finding

The current transformers learn MNI, but gain little additional subfield Dice from the original foreground-union bands. Earlier severe training saturation was real, especially for SwinUNETR, but it is no longer an adequate explanation on its own. The latest shorter runs retain training error and almost no late validation decline, while the bands advantage remains small.

The strongest current evidence is a change in error composition, together with a mismatch between what the constraint supervises and what the evaluation rewards. Historical transformers predominantly oversegmented foreground. MNI transformers have much more balanced foreground/background errors and substantial internal subfield confusion. Changes in base losses, augmentation and optimization plausibly explain part of this transition, but their individual causal contributions have not been isolated.

## Latest paired measurements

All scores below are percentages, final-budget checkpoints, MNI fold 0 seed 0, fresh paired runs at 63,952,901,334,180 optimization FLOPs (2× historical cap). These are pilots, not nine-run means.

| Model | Baseline train | Baseline validation | Bands train | Bands validation | Validation gain |
|---|---:|---:|---:|---:|---:|
| SwinUNETR | 81.81 | 76.07 | 82.58 | 76.53 | +0.46 pp |
| UNETR | 79.12 | 69.41 | 79.24 | 69.48 | +0.07 pp |
| nnUNet | 60.09 | 54.12 | 62.68 | 56.63 | +2.51 pp |
| UNet | 75.64 | 60.86 | 87.24 | 68.81 | +7.95 pp |

The bands checkpoints finish within 0.10 points of their best validation scores. This rules against a large late validation collapse in this pilot; it does not prove an absence of all overfitting. Swin's training/validation gap is 6.05 points, UNETR's 9.75.

## 1. The useful error signal changed

| Baseline setting | Swin internal swaps, % of wrong voxels | UNETR internal swaps, % of wrong voxels | Swin FP, % of external errors | UNETR FP, % of external errors |
|---|---:|---:|---:|---:|
| MSD historical, pooled nine runs | 1.61 | 2.37 | 99.62 | 97.57 |
| ADNI historical, pooled nine runs | 0 | 0 | 89.11 | 91.79 |
| MNI latest, fold 0 seed 0 | 24.71 | 21.25 | 54.21 | 39.72 |

External means foreground/background confusion, not a measured distance to the contour. Voxel error fractions do not decompose macro Dice. Nevertheless, the direction of change is clear: historical transformer errors were overwhelmingly excess foreground, whereas MNI UNETR now misses more foreground than it adds.

The current UNet provides a useful within-MNI comparison: 93.63% of its external errors are false positives. Bands remove 11,330 FP voxels while adding 2,078 FN voxels, and gain 7.95 Dice points. This resembles the historical oversegmentation regime. It supports the error-composition hypothesis without proving that architecture or loss alone caused the difference.

## 2. Union bands cannot supervise the subfield decision

ADNI is binary hippocampus/background. MSD distinguishes anterior/posterior but its transformer baselines had few internal swaps. MNI distinguishes CA1–3, subiculum and CA4–DG. The current bands loss combines their probabilities and supervises only hippocampus versus background.

For an inner-band voxel its objective is `-log(p_CA1_3 + p_subiculum + p_CA4_DG)`. It cannot distinguish the correct subfield from another foreground subfield. It also creates no band around a purely internal interface away from the outer surface. On a wrongly classified inner-band voxel, its direct logit gradient can strengthen the already winning wrong foreground class. Earlier zero-update probes verified this direction, but did not show reversals of the combined corrective base-loss direction at the tested low weight. This mechanism is a limitation, not proof that the constraint generally damages training.

The latest UNETR pilot exposes the practical mismatch:

| Validation outcome | Bands minus baseline |
|---|---:|
| Whole hippocampus Dice | +1.003 pp |
| CA1–3 Dice | +1.428 pp |
| Subiculum Dice | −0.342 pp |
| CA4–DG Dice | −0.879 pp |
| Mean subfield Dice | +0.069 pp |
| Foreground/background wrong voxels | −891 |
| Internal subfield swaps | +123 |

Swin improves all three classes slightly, but removes 977 FP voxels at the cost of 373 additional FN voxels. Its net effect is correspondingly modest. This is not the same failure mode as UNETR's cancellation between subfields.

## 3. The baseline recipes are substantially different

| Component | Successful historical MSD/ADNI transformer studies | Current MNI |
|---|---|---|
| Swin base loss | Standard MONAI softmax Dice, background included, unsquared denominator | Mean negative-log squared Dice, all four classes, no CE |
| UNETR base loss | Same standard Dice-only objective | Geo foreground-only squared Dice + CE, 1:1; dropout 0.1 |
| Peak LR | 0.0001 for both | Swin 0.0003; UNETR 0.0006 |
| Augmentation | None | Paired xy rotations/reflections for both |
| Compute | 1× historical cap | Latest 2×; previous 3× and 8× |
| LR horizon | Swin 610 but stopped at 206 updates; UNETR 477 but stopped at 245 | Horizon matched to cap: Swin 412; UNETR bands 491 |
| Bands calibration target | 10% for both | Swin 10%; UNETR 5% |
| Bands prefix/ramp | 50/50 updates | Swin 42/42; UNETR 50/50 |

Historical Swin training Dice was 57.07% on MSD and 75.07% on ADNI; UNETR was 62.07% and 68.27%. Bands gained 3.85/3.34 points for Swin and 3.89/2.99 for UNETR across nine runs per dataset. Historical schedules were interrupted before their low-LR tail, leaving more optimization headroom. Current MNI combines stronger recipes, augmentation and a completed schedule. Bands may consequently duplicate supervision already supplied by the baseline, especially UNETR's CE, but this is an untested causal hypothesis.

Reducing budget alone is not sufficient: going from 3× to 2× changes Swin's bands gain from +0.20 to +0.46, and UNETR's from −0.62 to +0.07. Absolute UNETR bands validation remains near 69.5%. The reduction severely harms nnUNet absolute performance. A larger relative bands gain is not useful if it comes from deliberately weakening the baseline.

## 4. What is not broken

Prior frozen-source audits established identical original band implementations across datasets, correct class grouping, nonzero gradients, correct paired transforms and approximately 80% of current MNI updates at full constraint strength. Current fresh calibration reaches exactly 10% for Swin and 5% for UNETR; weighted p95 ratios are 11.42% and 5.24%, well below the 75% guard. The guard is not suppressing the constraint.

One-time calibration matches full-volume logit gradient norms on unaugmented training crops. It does not match gradient direction, parameter updates or the later augmented training trajectory. Current anchor logit cosines are positive: approximately 0.03 for Swin and 0.35 for UNETR. These do not demonstrate destructive global conflict. UNet also has a small cosine and nevertheless benefits strongly, so small cosine alone is not an explanation.

## Ranked controlled experiments

1. **For UNETR, test class-aware boundary supervision first.** Keep the current recipe and cap. Compare original union bands against categorical boundary CE on exactly the same band support, with class/side-balanced reductions and matched gradient norm. This isolates label semantics. Separately test adding bands around internal subfield interfaces; do not change support and semantics in the first contrast. The previous categorical zero-update probe improved true-versus-wrong class margins at all measured swaps, but no trained ablation has established a Dice gain yet.
2. **For Swin, test how boundary weight is distributed.** Retain Dice-only as requested. Log inner/outer losses and train FN/FP separately, and compare the original balanced bands against a predefined training-derived inner/outer weighting or hard-voxel emphasis. The current FP reduction/FN increase makes this more targeted than simply increasing the overall gradient target. Validate any weighting rule on training labels only; current validation error counts are diagnostic evidence, not calibration inputs.
3. **Audit gradient influence through training before another target sweep.** Measure base versus constraint gradients at several fixed milestones and on the same transformed batches, including a small set of parameter-gradient checks. If the intended ratio drifts, test training-only recalibration as a separate ablation. Charge additional profiling/calibration consistently and report it. Do not assume 5–10% at one early anchor remains 5–10% of optimizer influence.
4. **Then isolate recipe redundancy.** At a fixed cap and augmentation, compare a small paired base-loss ablation: for Swin logarithmic versus linear squared Dice; for UNETR the current Dice/CE balance versus a predefined alternative that preserves learning. Every loss variant needs its own baseline and constraint arm and FLOP profiling. The historical plain-Dice recipe is an informative control, not an automatic replacement for the better baseline.

Do not start by further lowering every model's shared budget or restoring the old incomplete LR schedule simply to enlarge the relative gain. Prioritize absolute validation Dice, then consistency across the three folds and seeds, with a held-out evaluation after recipe selection.

## Evidence and limits

`analyze.py` verifies all 104 metadata files against the eight current archive manifests, recomputes confusion statistics and class-wise changes, and records source hashes in `evidence.json`. Historical comparisons come from the archived nine-run analysis and frozen protocols. Current runs have five training and five validation subjects, with two hemisphere crops each; ten crops are not ten independent subjects. Historical datasets use different splits, label definitions and recipes. This comparison identifies plausible mechanisms but cannot assign a causal percentage to each change.

Scope is MNI. COBRA was set aside earlier; no equivalent completed constrained paired study is available here to support the same diagnosis for COBRA. No jobs were submitted or recipes modified in this investigation.
