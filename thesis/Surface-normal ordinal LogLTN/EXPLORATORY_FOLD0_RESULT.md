# Exploratory fold-0 result from the existing saved probabilities

## Scope

This is an exploratory screening result, not a performance claim. It uses all 52 fold-0 validation cases whose labels have already informed the boundary research direction. It must not replace evaluation on untouched folds, seeds, or a final test set.

The run used the saved SwinUNETR probabilities in `datasets/Dataset101_MSD/inference_fold0_val_600236`, reconstructed equivalent logits as `log(probability)`, read the 1-mm spacing from each source NIfTI, and used:

- outer and anterior/posterior interfaces;
- ray radius 3 mm and step 0.5 mm;
- pair offset \(\delta=1\) mm;
- margin \(m=0\), temperature \(T=1\);
- localization tolerance 1 mm;
- at most 4096 surface faces per interface and case;
- frozen-logit repair RMS 0.10;
- auxiliary-to-Dice gradient RMS ratio 0.10.

## Result

The first automatic screening verdict was **NO_GO for full training of the simple pairwise ordinal formulation at this configuration**.

### Outer surface

| Quantity | Result |
|---|---:|
| Sampled rays | 139,916 |
| Erroneous rays | 27,532 (19.68%) |
| Ordinal violations | 4,066 (2.91%) |
| Error coverage | **14.77%** |
| Error precision | **100%** |
| Precision lift | **5.08×** |
| Shifted crossings | 16,787 |
| Missing crossings | 3,535 |
| Multiple crossings | 7,106 |
| Reversed crossings | 104 |

Violations are extremely specific: every violation occurred on a ray classified as erroneous. However, they cover only about one in seven erroneous rays. The majority of boundary errors already have the desired inside-to-outside ordering, so the pairwise rule cannot observe them.

### Anterior/posterior interface

| Quantity | Result |
|---|---:|
| Sampled rays | 4,108 |
| Erroneous rays | 1,762 (42.89%) |
| Ordinal violations | 240 (5.84%) |
| Error coverage | **13.62%** |
| Error precision | **100%** |
| Precision lift | **2.33×** |

The same pattern holds internally: highly specific but low-coverage supervision.

### Frozen-logit repair

An ordinal-only update improved union Dice and 1-mm surface Dice in every case, with mean changes of approximately +0.00421 and +0.01333 respectively. It also reduced mean foreground FN by 23.4 voxels, but introduced an average of 2.2 additional A/P swaps. This shows that the ordinal gradient is locally meaningful when applied strongly.

At the intended 10% auxiliary/Dice gradient ratio, however, `Dice + outer ordinal` versus the equal-RMS `Dice only` update produced:

- mean union-Dice difference: +0.000024;
- mean 1-mm surface-Dice difference: +0.000148;
- cases with improved 1-mm surface Dice: 23.1%;
- unchanged cases: 75.0%;
- mean foreground FP difference: +0.06 voxels;
- mean foreground FN difference: −0.19 voxels;
- mean A/P swap difference: +0.06 voxels.

The existing grouped band BCE was more effective under the same diagnostic: `Dice + bands` improved 1-mm surface Dice in 75% of cases, with a mean difference of +0.000403 relative to `Dice only`.

## Interpretation

The simple normal pair is not useless: it identifies a small, clean subset of malformed transitions and its direct gradient moves boundaries beneficially. The problem is **coverage**, not gradient direction. Most observed errors are shifted, missing, or multiply crossing while still satisfying the local ordering at \(\pm1\) mm.

Therefore the evidence argues against spending a full training run on this pairwise loss as the primary boundary solution. The most defensible next step is to retain it as a diagnostic or targeted term and develop the tolerance-aware one-cut ray formula, which can supervise transition existence, multiplicity, and allowed location rather than orientation alone.

The counterfactual result is only a local logit-space experiment. Network parameterization and optimization can behave differently, so it cannot prove that training would fail. It does, however, provide a concrete reason to prefer a more expressive logical constraint before allocating a full cluster run.

## Improved one-cut audit

The audit was then extended without changing the saved model outputs. The improved formula evaluates the whole ray and existentially marginalizes all single foreground-to-background cuts whose midpoint is within 1 mm of the GT surface.

### Structural observability

| Interface | ROC AUC | Error coverage at 5% correct-ray FPR | Precision | Lift |
|---|---:|---:|---:|---:|
| Outer H/background | 0.99996 | 100.0% | 83.05% | 4.22× |
| Anterior/posterior | 0.99395 | 98.24% | 93.62% | 2.18× |

This large increase over the ordinal pair confirms that the one-cut score responds to displaced, missing, reversed, and multiple transitions. It is a structural sanity check, not independent predictive evidence: both the error categories and the formula are defined from closely related ray-crossing conditions.

### Outer one-cut frozen-logit repair

`Dice + outer one-cut` versus the equal-RMS `Dice only` update produced:

- mean union-Dice difference: **+0.000196**;
- mean 1-mm surface-Dice difference: **+0.000533**;
- mean 2-mm surface-Dice difference: **+0.000097**;
- mean ASSD difference: **−0.000670 mm**;
- mean HD95 difference: **−0.01249 mm**;
- cases with improved 1-mm surface Dice: **61.5%**;
- worse cases: 5.8%;
- unchanged cases: 32.7%;
- mean foreground FP difference: −0.10 voxels;
- mean foreground FN difference: −1.12 voxels.

The outer one-cut gradient had a positive Dice-gradient cosine in every case, with mean approximately 0.107. The per-case weight required for the 10% RMS match had median approximately 0.0198.

This passes the pre-specified screen for a **short outer-surface pilot**.

### Comparison with the existing grouped band BCE

Under the same matched repair, one-cut versus bands produced:

- 1-mm surface-Dice mean difference: +0.000130 in favour of one-cut;
- union-Dice mean difference: −0.000639 in favour of bands;
- ASSD mean difference: +0.002724 mm in favour of bands;
- one-cut had higher 1-mm surface Dice in 36.5% of cases and lower in 44.2%;
- bands reduced FP and FN more consistently.

Therefore the audit does **not** show that one-cut is already superior to the band loss. It shows a distinct trade-off: one-cut has a somewhat larger mean response in the strict 1-mm surface metric, while bands remain stronger and more consistent for overlap, ASSD, FP, and FN. The existing BCE band run must remain the primary comparator in the pilot.

### A/P one-cut result

The direct A/P one-cut update reduced swaps strongly, but the 10%-matched `Dice + A/P one-cut` update had:

- mean A/P swap difference: −0.423 voxels;
- mean union-Dice difference: −0.000044;
- mean 1-mm outer surface-Dice difference: −0.000038.

This is a real objective trade-off. The A/P term should **not** be mixed into the first outer-surface pilot. It should be deferred to a separate class-interface ablation with swap- and interface-specific endpoints.

## Updated decision

- **Outer one-cut:** `GO_TO_SHORT_PILOT`.
- **Simple ordinal pair:** retain only as a baseline/ablation.
- **A/P one-cut:** defer to a separate trade-off experiment.
- **Full training campaign:** not yet authorized; the frozen-logit audit cannot establish parameter-space trainability or generalization.

The recommended first pilot is outer one-cut only, from the same initialization and exposure as Dice-only and grouped-band controls, with its weight calibrated to the same 10% logit-gradient RMS target. The pilot should use a pre-specified early gate before any full multi-seed run.
