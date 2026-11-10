# MNI constraint investigation — 7 November 2026

The gradient-target search is stopped. This report distinguishes verified implementation, observed training/generalization behavior, and causal hypotheses.

## Paired evidence (three folds × three seeds)

| Dataset | Model | Baseline train Dice | Baseline validation Dice | Bands gain, validation pp | Bands gain, training pp |
|---|---|---:|---:|---:|---:|
| MSD | nnunet | 66.74 | 62.52 | +7.55 | +7.50 |
| MSD | swinunetr | 57.07 | 54.04 | +3.85 | +5.19 |
| MSD | unetr_vit_b16 | 62.07 | 54.71 | +3.89 | +6.61 |
| MSD | basic_unet16 | 78.50 | 73.83 | +4.83 | +7.09 |
| ADNI | nnunet | 84.86 | 75.66 | +0.73 | +0.87 |
| ADNI | swinunetr | 75.07 | 68.72 | +3.34 | +4.99 |
| ADNI | unetr_vit_b16 | 68.27 | 56.27 | +2.99 | +7.23 |
| ADNI | basic_unet16 | 76.69 | 69.64 | +3.81 | +7.30 |
| MNI | nnunet | 98.55 | 73.86 | +0.07 | +0.19 |
| MNI | swinunetr | 99.84 | 71.24 | -0.09 | +0.08 |
| MNI | unetr_vit_b16 | 88.06 | 70.46 | +0.45 | -0.32 |
| MNI | basic_unet16 | 98.72 | 70.92 | +0.03 | +0.33 |

MSD/ADNI: historical 1× cap, nnUNet target60%, other models10%. MNI: 8× cap, all models10% target for this complete nine-run comparison. Current stronger targets are separate one-fold/one-seed pilots. These are paired results within each dataset, not a matched cross-dataset causal ablation.

## Supported explanation

1. The optimization regime changed. MSD/ADNI models retained substantial training error and bands improved that fit. Three MNI models already fit their five training subjects at98.5–99.8% Dice while validation remains71–74%; extra annotation supervision mostly improves memorization. The historical gain should not be expected to transfer unchanged after stronger base losses and an8× training budget.
2. The metric and error composition changed. Bands supervise the foreground union; MNI evaluates three foreground subfields. Roughly27% of MNI wrong voxels are internal subfield swaps, versus2–15% for MSD and0% by definition for binary ADNI. Error fractions are voxel counts, not additive contributions to macro Dice.
3. Early benefits are often optimization acceleration. MNI nnUNet60% bands is about+1.99pp at15% of budget but−0.13pp at the final budget. The full nine-run10% bands comparison has an early mean nnUNet gain of+0.76pp, ending at+0.07pp. Swin/UNet early effects are less consistent. UNETR has a persistent positive gain and should not be described as universally failing.

## What has been ruled out

Frozen bands implementations for MNI, MSD, and ADNI are byte-identical. Correct MNI union class IDs are(1,2,3), background0. Independent loss/gradient checks, support checks and all eight paired Geo transforms passed. Constraints start after10% and are at full strength for about80% of MNI updates. nnUNet60% calibration actually reaches60%; it is not clipped to10%. There is no evidence of disabled bands, wrong label grouping, missing gradients, or a too-short activation window.

## Mechanistic distinction

For a true-foreground voxel, the outer-band term is −log(sum of foreground probabilities). Its negative gradient increases a foreground logit in proportion to that class’s current foreground probability. If the winning foreground class is already wrong, this increases its logit margin over the true subfield. This is a local tendency, not proof that full model training increases swaps. In fact, pooled internal swap counts improve modestly in all four complete MNI10% bands comparisons. The observed failure is broader: external errors barely decline for nnUNet and increase for Swin/UNet.

## Provenance

Cross-dataset numbers and paired source hashes: [cross_dataset_evidence.json](cross_dataset_evidence.json). Prior frozen implementation audit: [audit.json](../mni_bands_implementation_audit_20261107/audit.json). Matched-FLOP learning curves: [comparison.json](../mni_bands_learning_curves_20261107/comparison.json). Zero-update probes: [probe_summary.json](probe_summary.json); full per-case measurements: [interrupted_probes.json](interrupted_probes.json).

## New checkpoint measurements (zero updates)

These are the interrupted **bands+edge5%** runs, not final baseline checkpoints. Swin was at973 updates; UNETR at350, still ramping. Models were evaluated without augmentation; parameter gradients were measured on one fixed training crop only. Validation labels were used for diagnostic measurement, never calibration or optimization.

| Model | Training Dice | Validation Dice | Validation FN+FP inside band | Unramped training logit ratio | One-case parameter ratio |
|---|---:|---:|---:|---:|---:|
| swinunetr | 99.38% | 71.57% | 93.4% | 8.27% | 6.41% |
| unetr_vit_b16 | 74.36% | 67.68% | 88.5% | 5.40% | 4.56% |

Swin’s bands loss is0.0567 on training crops versus0.5692 on validation: a tenfold gap on the **same boundary objective**. Its median training logit ratio drifted from5% at calibration to8.27%; this is neither a vanished loss nor an explosion. The one-case parameter ratio is6.41% with positive cosine0.872 to the base gradient. UNETR’s unramped logit ratio is5.40%; actual ramped ratio4.20%. These measurements do not establish ratios for completed nnUNet/UNet checkpoints, which were already retired.

The bands gradient worsens the true-versus-wrong subfield margin at2923 of2924 in-band Swin validation swaps (one is numerically zero), consistent with the analytical grouped-loss limitation. However, at this5% target, adding bands+edge creates **zero reversals** of previously corrective base-loss margin directions. The weighting is too small for this mechanism to explain the overall failure by itself. A categorical boundary CE probe improves the margin at all2924 such swaps at matched raw gradient norm; that validates its local direction, not generalization or trained Dice.

The corrected distinction is therefore: the bands still see most boundary errors, but they are optimized against nearly memorized **training** boundaries. The remaining validation boundary errors do not provide a training signal. Increasing the constraint weight cannot supply missing subject variation.

## Next controlled test, specified but not launched

First isolate generalization: keep the8× cap and winning losses fixed, and compare **paired D4 augmentation baseline versus paired D4 augmentation+10% bands** for nnUNet and Swin, fold0/seed0 (four new runs). Compare to the existing corresponding no-augmentation pairs. Match augmentation sequences within pairs and log base/band losses, gradients, and spatial error breakdown over training. This separates augmentation benefit from any incremental bands benefit. The existing UNETR pipeline already exercises this exact paired transform, and all eight image/label/geometry permutations passed checks. Details: [next_experiment.json](next_experiment.json).

Class-aware boundary supervision is a separate subsequent ablation. Do not change augmentation, boundary semantics, target and cap simultaneously; otherwise a successful run would not identify the cause. No new training sweep has been submitted.

## Limits and conclusion

There is no demonstrated training-implementation defect. The best-supported explanation is **a changed optimization/generalization regime**, with an additional mismatch between union supervision and subfield evaluation. This is supported by paired results, learning curves, exact frozen source and zero-update probes. Separating the causal contribution of budget, new base losses, augmentation and dataset anatomy requires the controlled ablations described above; the present evidence does not prove any proposed replacement recipe will win. The observed error mix is specific to these trained models, not an immutable property of MNI.

## Stop and cleanup

Search job743858 stopped; continuation743861 cancelled. Diagnostic job744001 performed zero optimizer updates. Its final archive-manifest call had a duplicate-keyword error after both probes/evaluations were saved; the metadata-only recovery verified all results and unchanged checkpoint hashes before archival. No GPU computation needed repeating.

Both interrupted runs have train/latest/best validation evaluations, confusion matrices, FN/FP/internal swaps, protocol and source/checkpoint hashes archived outside BeeGFS. After local hash verification,2,190,933,554 checkpoint bytes (2.04GiB) were deleted. Quota after cleanup:44.88GiB used of93.13GiB,48.25GiB free. The search remains paused; final verification records the empty queue and checkpoint inventory. [Cleanup receipt](cleanup.json), [verification](final_verification.json).
