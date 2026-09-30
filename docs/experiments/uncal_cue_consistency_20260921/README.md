# Broad MRI/shape cue audit: fold 0 training and validation

2026-09-21. Exploratory analysis of 208 training and 52 validation **volumes**. Case-level splits cannot be assumed to separate subjects: the released crops lack a verified left/right subject-pair mapping. Both sets have informed earlier research, so this is a consistency audit, not a fresh holdout experiment.

[Read the interpretation and LTN implications](interpretation.md).

## What was tested

170 scalar measurements cover separate profiles, disappearance of small profiles, vertical/horizontal tissue runs, holes, superior notches and protrusions, adjacent-slice area/contour changes, sagittal/axial anterior bridges, and local T1 intensity/edge/context changes. Current values and adjacent-slice differences are tested. This is a broad explicit candidate set, not every possible anatomical cue.

The sagittal return proxy finds vertically separated tissue runs at a candidate y and requires the gap interval to be bridged at the same x within 1–4 mm anteriorly. The analogous axial proxy works along x at a fixed z. Minimum tissue-run lengths of one and two voxels test sensitivity to thin profiles. These axis-aligned bridge tests can miss curved folds and can flag unrelated clefts. MRI darkness inside these gaps is also measured; no CSF or named structure is segmented.

Features receive MRI and the union of anterior/posterior reference masks, never the class identity. This is oracle foreground support. Targets are fitted class interfaces; they are not independent anatomical apex annotations. Raw MRI is robustly scaled per volume. Candidate cuts cover all occupied native coronal planes except the most posterior one, because a cut requires tissue on both sides. Increasing y is anterior; all coordinates are native zero-based indices.

## Discrete cue coverage

Counts are volumes, not slices. “At cut” is the fitted interface. “Near” means within 2 mm. “Elsewhere” means the cue also occurs more than 3 mm from the interface. Near-cut coverage alone is not localisation accuracy.

| Cue | Train at cut /208 | Val at cut /52 | Train near /208 | Val near /52 | Train elsewhere /208 | Val elsewhere /52 |
|---|---:|---:|---:|---:|---:|---:|
| two_components | 4 | 2 | 4 | 2 | 20 | 9 |
| secondary_disappears | 4 | 2 | 4 | 2 | 7 | 3 |
| vertical_double_min2 | 4 | 0 | 10 | 7 | 5 | 2 |
| sagittal_return_min1 | 34 | 9 | 76 | 23 | 51 | 13 |
| sagittal_return_min2 | 27 | 6 | 52 | 16 | 23 | 5 |
| axial_return_min1 | 85 | 23 | 155 | 45 | 169 | 42 |
| axial_return_min2 | 49 | 19 | 112 | 41 | 130 | 33 |

Two components require eight-neighbour separation and at least two voxels in each component. The disappearance flag requires a secondary component to have no same-coordinate overlap with the immediately posterior mask; shape displacement can also produce this flag. Neither condition establishes anatomical identity.

## Strongest individual measurements on training data

Direction and threshold are chosen using training only. Ranking below uses training within-volume AUC: how often the cut scores above a distant candidate, with ties worth 0.5. Thresholds maximise training sensitivity minus mean within-volume distant-slice false-positive rate over 101 quantiles. Training values are fitted/descriptive and optimistic; validation values are not used to choose direction, threshold, or table order. Multiple candidates were explored; no confirmatory significance claim is made.

| Measurement | Direction | Train distant AUC | Val distant AUC | Val local AUC | Train cut sensitivity | Val cut sensitivity | Val distant-slice flag rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| next_delta__shape__area | +1 | 0.867 | 0.904 | 0.745 | 91.3% | 96.2% | 29.1% |
| current__image__superior_band_left_right_difference_abs | +1 | 0.860 | 0.857 | 0.673 | 80.3% | 76.9% | 23.3% |
| current__image__superior_band_intensity_std | +1 | 0.828 | 0.807 | 0.551 | 76.0% | 71.2% | 30.0% |
| delta__shape__area | +1 | 0.796 | 0.846 | 0.686 | 82.2% | 90.4% | 33.5% |
| transition__relative_area_change | +1 | 0.794 | 0.833 | 0.720 | 82.2% | 90.4% | 31.9% |
| transition__removed_fraction | -1 | 0.747 | 0.773 | 0.535 | 89.9% | 92.3% | 47.8% |
| transition__superior_boundary_rise_max | +1 | 0.732 | 0.741 | 0.649 | 53.4% | 53.8% | 11.4% |
| delta__image__central_superior_band_intensity_mean | -1 | 0.712 | 0.744 | 0.635 | 67.3% | 75.0% | 34.8% |
| next_delta__shape__width | +1 | 0.707 | 0.747 | 0.638 | 96.6% | 92.3% | 65.9% |
| transition__superior_boundary_rise_mean | +1 | 0.699 | 0.718 | 0.671 | 59.6% | 65.4% | 23.1% |
| transition__novel_fraction_after_dilation | +1 | 0.696 | 0.728 | 0.705 | 69.2% | 71.2% | 31.0% |
| delta__shape__width | +1 | 0.692 | 0.719 | 0.670 | 61.5% | 67.3% | 32.2% |
| transition__relative_width_change | +1 | 0.683 | 0.711 | 0.676 | 61.5% | 65.4% | 29.0% |
| transition__novel_superior_side_difference | +1 | 0.681 | 0.721 | 0.697 | 50.5% | 55.8% | 13.3% |
| transition__novel_superior_side_max | +1 | 0.679 | 0.719 | 0.697 | 51.4% | 57.7% | 13.9% |

Local AUC compares the cut with the other candidates within 3 mm, using the same training-fitted score direction. A value near 0.5 means the scalar does not distinguish the exact cut from its neighbours, even if it separates this broad region from distant slices.


## Can combinations locate the cut?

A fixed logistic ranker (C=1) combines each feature family, with and without a relative-position prior. Four-fold case-wise out-of-fold predictions are computed inside the 208 training volumes; scalers and position statistics are fitted separately within each fold. One final fit on all 208 predicts validation. Each training case has equal total weight and equal positive/negative mass. The highest score selects the cut; exact ties use the middle tied candidate. No segmentation network was trained.

| Inputs | Train OOF MAE, mm | Val MAE, mm | Val exact | Val within 1 mm | Val within 2 mm | Val maximum, mm |
|---|---:|---:|---:|---:|---:|---:|
| position_only | 1.159 | 1.173 | 23.1% | 71.2% | 92.3% | 4 |
| geometry | 2.356 | 3.269 | 38.5% | 67.3% | 76.9% | 22 |
| geometry_position | 1.091 | 1.327 | 48.1% | 73.1% | 86.5% | 20 |
| mri | 3.784 | 2.654 | 34.6% | 63.5% | 76.9% | 24 |
| mri_position | 1.293 | 0.923 | 51.9% | 75.0% | 88.5% | 4 |
| crossplane | 4.279 | 3.288 | 23.1% | 48.1% | 67.3% | 21 |
| crossplane_position | 1.447 | 1.269 | 32.7% | 61.5% | 86.5% | 5 |
| all | 1.510 | 1.827 | 46.2% | 76.9% | 86.5% | 20 |
| all_position | 0.933 | 1.250 | 42.3% | 78.8% | 90.4% | 20 |

Lowest training OOF MAE selects **all_position**. The comparison against position-only is required: a cue that merely recognises the usual location is not enough evidence for an anatomical landmark detector.

Validation MAE reduction versus position-only: -0.077 mm; descriptive volume-bootstrap 95% interval [-1.000, 0.500] mm. Better/equal/worse volumes: 24/17/11. This interval does not account for model selection, prior validation exploration, or unknown paired hippocampi.

- train, planar reference split: n=161, selected-model MAE 0.938 mm.
- train, nonplanar reference split: n=47, selected-model MAE 0.915 mm.
- val, planar reference split: n=41, selected-model MAE 1.122 mm.
- val, nonplanar reference split: n=11, selected-model MAE 1.727 mm.

Using the actual most-posterior anterior voxel plane instead of the fitted plane gives selected-model validation MAE 1.250 mm.

## Post-hoc diagnostic sensitivity

Inspection of the original largest validation error (033, predicted y=6 versus target y=26) showed that the closing-based solidity proxy is nearly constant in training but extreme at that posterior tip. Its three standardised terms dominate the score. This is a descriptor/extrapolation failure, not evidence of absent uncal anatomy.

Removing only the current/delta/next-delta solidity proxy and retaining the same four training folds, C=1 and other inputs yields training OOF MAE **0.938 mm** and validation MAE **0.865 mm**, within 1 mm **80.8%**, maximum **4 mm**. The position-only comparator remains 1.159/1.173 mm on training OOF/validation.

This exclusion was motivated by a validation failure. Its improved result is explicitly exploratory and requires independent evaluation. Original results and original-model gallery predictions remain above; they have not been replaced by the corrected variant.
[Sensitivity results](../../../experiments/uncal_cue_consistency_20260921/sensitivity.json) and [033 score contributions](../../../experiments/uncal_cue_consistency_20260921/case033_failure.json).

## Evidence gallery and every-case results

- [hippocampus_185](../../../experiments/uncal_cue_consistency_20260921/hippocampus_185.png): Previously discussed validation example.
- [hippocampus_205](../../../experiments/uncal_cue_consistency_20260921/hippocampus_205.png): Previously discussed validation example.
- [hippocampus_164](../../../experiments/uncal_cue_consistency_20260921/hippocampus_164.png): Previously discussed validation example.
- [hippocampus_327](../../../experiments/uncal_cue_consistency_20260921/hippocampus_327.png): Previously discussed validation example.
- [hippocampus_145](../../../experiments/uncal_cue_consistency_20260921/hippocampus_145.png): Training positive: secondary_disappears.
- [hippocampus_160](../../../experiments/uncal_cue_consistency_20260921/hippocampus_160.png): Training positive: secondary_disappears.
- [hippocampus_023](../../../experiments/uncal_cue_consistency_20260921/hippocampus_023.png): Training positive: sagittal_return_min1.
- [hippocampus_042](../../../experiments/uncal_cue_consistency_20260921/hippocampus_042.png): Training positive: sagittal_return_min1.
- [hippocampus_001](../../../experiments/uncal_cue_consistency_20260921/hippocampus_001.png): Training positive: axial_return_min2.
- [hippocampus_011](../../../experiments/uncal_cue_consistency_20260921/hippocampus_011.png): Training positive: axial_return_min2.
- [hippocampus_236](../../../experiments/uncal_cue_consistency_20260921/hippocampus_236.png): Largest training OOF localisation errors.
- [hippocampus_226](../../../experiments/uncal_cue_consistency_20260921/hippocampus_226.png): Largest training OOF localisation errors.
- [hippocampus_033](../../../experiments/uncal_cue_consistency_20260921/hippocampus_033.png): Largest validation localisation errors.

These examples are selected for evidence inspection, not sampled to estimate expert visibility. Automated extraction covers every volume; detailed manual review does not.

[All 260 case results](cases.md), [full numerical results](../../../experiments/uncal_cue_consistency_20260921/summary.json), and [summary plot](../../../experiments/uncal_cue_consistency_20260921/summary.png).

## Boundaries on the conclusion

- Disconnected components, double runs, or bridges are geometric cues. Their absence cannot establish absence of the uncus; their presence cannot establish uncal identity.
- Reliable hemisphere/medial direction is not supplied by the crop metadata. Side descriptors are symmetrised; no medial-versus-lateral anatomical claim is made.
- Surrounding named structures such as crus cerebri and ventricular recess are not independently labelled. Local context intensity is only a proxy; the complete Woolard procedure is not automated here.
- Reference union masks encode expert delineation information. Results must be repeated on predicted foreground before using a deployable MRI/shape predicate.
- Cue/annotation agreement does not establish generalisation gains from an LTN loss. Independent landmarks and a new untouched evaluation split are needed for that claim.
- No labels, baseline models, or training losses were modified.

## Reproduction

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cue_consistency
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cue_sensitivity
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.report_cue_consistency
rtk proxy .venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_cue_consistency.py -q
```

Split SHA256: `1d6a3fe993359ad85e9c28a29901162251c3359449b710c09cb9e92d66dda472`.
