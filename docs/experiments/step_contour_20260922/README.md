# Descending contour audit — 22 September 2026

## Question

Does the superior outline of the segmented hippocampus follow an approximately
decreasing step function along its posterior-to-anterior axis? Can either a
violation or the patient-specific *position* of that contour locate voxel errors?

## Data and measurement

- All 260 native MSD reference volumes were measured. They are axis-aligned,
  with 1 mm voxels, increasing `y` anteriorly, and increasing `z` superiorly.
- The foreground is the union of anterior and posterior labels. On each coronal
  slice with area at least `max(5 voxels, 10% of that volume's peak slice area)`,
  the upper and lower outlines are the 90th and 10th percentiles of foreground
  `z` coordinates. The center is their voxel-weighted mean `z`.
- A decreasing isotonic regression fits a step function to each profile. Its
  root-mean-square residual measures departure from monotonicity. An adjacent
  upward change greater than one voxel is counted as a local reversal.
- For error association, the 52 fold-0 validation predictions from the matched
  early-stopped unaugmented and augmented Swin runs were compared with reference
  masks. Every saved 64³ reference mask was verified to be the exact centered
  padding of its native annotation before errors were counted. These are two
  predictions of the **same** 52 volumes, not independent validation cohorts.

## Results

The visual observation holds remarkably well in the reference labels:

| Upper-contour measure | 260 references | 52 unaugmented predictions | 52 augmented predictions |
| --- | ---: | ---: | ---: |
| Median Spearman correlation with anterior position | −0.995 | −0.996 | −0.996 |
| Median posterior-to-anterior height change | −17.0 voxels | −16.5 voxels | −17.0 voxels |
| Median decreasing step-fit residual | 0.141 voxels | 0.135 voxels | 0.136 voxels |
| Volumes with an upward jump greater than 1 voxel | 33/260 | 8/52 | 6/52 |

All 260 reference upper contours have a negative position correlation; 258/260
are at or below −0.7. The lower outline and center also descend (median
correlations −0.997 and −0.999). Cross-sectional **area** does not follow a
decreasing profile (median correlation +0.442). Thus the image mostly shows a
sloping structure, rather than monotonic thinning. The literal stair steps
reflect voxel sampling; a decreasing isotonic fit can also use as many plateaus
as needed and is not proof of a biological step law.

The prediction already reproduces this pattern, so a violation is a weak
case-level error signal:

| Fold-0 score | Unaugmented: correlation with boundary-error count | Unaugmented: error capture in top 20% | Augmented: correlation | Augmented: capture |
| --- | ---: | ---: | ---: | ---: |
| Upper step-fit residual | −0.011 | 22.5% | −0.030 | 19.9% |
| Upper upward-jump excess | +0.155 | 21.4% | −0.011 | 21.0% |
| Predicted foreground volume | +0.095 | 23.8% | +0.088 | 21.1% |

The top 20% is 11 of 52 volumes; a score unrelated to errors would capture
about 21.2% of error voxels on average. The contour scores do not consistently
beat that reference or the simple volume comparator.

There is a limited local effect. In the unaugmented run, the two slices adjacent
to each greater-than-one-voxel upper jump select 16 of 1,976 supported slices
and contain 0.83% of boundary errors while containing 0.45% of reference
foreground voxels. In the augmented run, 12 of 1,980 slices contain 0.51% of
boundary errors and 0.26% of foreground. Those slices have roughly 1.8–1.9×
the average error density by this denominator, but capture very little of the
total error. Twelve of the same 52 **reference** volumes also have a valid
greater-than-one-voxel upper jump, so removing every reversal would contradict
the annotations.

## Voxel-level contour overlap

The case-level reversal score above does **not** test where the correct contour
lies. For this second audit, each reference and predicted mask was projected
across the lateral `x` axis. The exact upper and lower extent of that sagittal
silhouette was measured on every occupied `y` slice and separately fitted with
decreasing isotonic steps. We then crossed one- and two-voxel `z` bands around
those curves with the actual extra and missed foreground voxels. The candidate
denominator is the union of reference and predicted foreground voxels, so the
reported error fractions are descriptive and label-informed.

| Run and fitted contour | Error recall within 1 voxel | Candidate foreground covered | Error recall within 2 voxels | Candidate foreground covered |
| --- | ---: | ---: | ---: | ---: |
| Unaugmented, reference contour | **51.8%** | 30.1% | 73.1% | 54.8% |
| Unaugmented, predicted contour | 52.8% | 32.2% | 74.3% | 57.7% |
| Augmented, reference contour | **52.6%** | 29.6% | 73.4% | 54.3% |
| Augmented, predicted contour | 55.1% | 32.7% | 76.1% | 58.4% |

Thus the correct patient-specific contour **does contain useful spatial
information**: for the unaugmented run, a one-voxel band contains 18,793 of
36,262 boundary-error voxels. The error fraction among union-foreground voxels
in that band is 32.1%, versus 18.7% over the whole union. However, a band
around the predicted contour captures roughly the same errors at slightly
wider coverage. This experiment does not show that an independently estimated
reference contour would improve inference.

The reference and predicted fitted upper/lower steps are within one voxel at
95.2% of matched positions in the unaugmented run (96.1% augmented). With the
*exact* reference silhouette extrema, voxels lying beyond the correct contour
account for 17.4% of all extra or missed foreground in the unaugmented run
(15.6% augmented). These are guaranteed errors by construction, but locating
them requires the reference boundary or a reliable independent estimate. Most
remaining errors occur inside the same sagittal silhouette, where the lateral
`x` shape still matters. The union contour cannot distinguish the 4,292
unaugmented and 3,702 augmented anterior/posterior class swaps.

The illustrative [voxel overlay](../../../evaluation_output/step_contour_20260922/voxel_overlay_164.png)
under the generated output directory
plots extra and missed foreground projected over `x`, with both fitted steps.
It shows errors throughout the ribbon even where the reference and predicted
steps nearly overlap.

## Interpretation

The descending contour is real and easy to measure in this MSD coordinate
system. It is already present in the tested model predictions and does not
provide a useful **global violation score** on these 52 volumes. Its
patient-specific position does locate many errors near the upper and lower
surface. The unresolved question is how to estimate a more accurate contour
from the MRI independently of the mask; the reference-derived result is an
oracle. A hard monotonicity constraint is not supported: it would penalize
valid reference shapes, while predicted curves already follow that rule.

This is an exploratory audit of one previously studied validation fold. It
does not test how an image-derived contour head or contour loss would affect
retraining, and it does not establish transfer to a differently oriented
dataset.

A follow-up [sagittal step-function fit](sagittal_step_fit.md) implements the
patient-specific, section-level constraint and reports a frozen-gradient
screen. It fits whole upper and lower contour functions within each sagittal
slice, with a gradient through the predicted step fit.

## Reproduce

Implementation: [audit_step_contour.py](../../../scripts/audit_step_contour.py).
Generated JSON and figure are in ignored
`evaluation_output/step_contour_20260922/`.

```bash
rtk proxy .venv/bin/python scripts/audit_step_contour.py \
  --output evaluation_output/step_contour_20260922/early_baseline.json

rtk proxy .venv/bin/python scripts/audit_step_contour.py \
  --predictions experiments/uncal_fold_early_stopping_20260921/voxel_audit/augmentation_seed0/error_maps \
  --output evaluation_output/step_contour_20260922/early_augmented.json
```
