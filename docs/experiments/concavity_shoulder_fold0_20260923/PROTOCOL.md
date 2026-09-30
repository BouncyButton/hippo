# Concavity-shoulder localization: fold 0

Protocol fixed on 23 September 2026 before computing localization results.

## Hypothesis and geometry

The superior sagittal outline changes from a steep posterior segment to a flatter
anterior segment near the A/P annotation cut. This is a shoulder of a concavity,
not necessarily a local height minimum. The screenshot camera pose is unknown;
use canonical RAS coordinates, never camera coordinates or A/P colours.

Use the largest 26-connected component of the binary hippocampus union. Extract
the upper sagittal silhouette (maximum z over x), and separate upper profiles in
individual sagittal x slices. Smooth each contiguous profile with sigma 1 mm.
At each candidate separator c-0.5, fit straight lines to the four immediately
posterior and four immediately anterior samples. Score the positive slope change
only when the posterior slope is descending with increasing y. A linear sloping
outline must score zero. No target cut enters extraction or candidate selection.

Three fixed geometric scores: silhouette; mean sagittal-slice score times the
fraction of eligible slices with a positive shoulder; geometric mean of these two
(primary consensus). Require complete local windows; no interpolation over gaps.
This tests a local change in slope, not recognition of a named anatomical structure.

## Fitting and evaluation

- Use all 208 training and 52 validation crops from the saved fold-0 split.
- Raw score maxima require no fitting. Report both uncorrected and training-median
  offset-corrected primary shoulder estimates.
- A hybrid ranks candidates by within-case standardized consensus score minus a
  training-derived quadratic relative-position penalty. Choose penalty strength
  from [0, 0.25, 0.5, 1, 2, 4, 8] using internal cross-validation MAE only.
- Report training estimates out of fold (four folds). Hybrid strength selection
  occurs inside each outer training fold using three inner folds. Final validation
  model selects strength by four-fold CV within the full 208 training cases.
- Compare with the training-median relative-position baseline. Freeze everything
  before evaluating validation or its predicted foreground supports.
- Transfer the same fitted detector to available saved validation predictions,
  including the viewer's baseline and early-stopped unaugmented/augmented models.
  Verify exact cached-label alignment. Preserve full prediction extent, including
  predictions outside the acquired native crop; convert padded y back to native y.
- Targets are best-fit A/P planes: class 2 for y<c, class 1 for y>=c. Original mixed
  labels are retained; report nonplanarity and ties separately. Native cut MAE is
  a plane-summary metric, not full segmentation quality.
- Report MAE, within 1/2 mm, p90/max errors, paired helped/equal/harmed and bootstrap
  uncertainty, and support-transfer displacement. Export every case and a gallery
  of all reference masks plus representative predicted-mask successes/failures.
- A fixed 90th-percentile training-OOF absolute-error radius gives descriptive
  interval coverage only. It is not a conformal guarantee or calibrated confidence.

## Interpretation boundaries

Reference union is oracle foreground. Validation fold 0 has been examined repeatedly
in this project; this is a development audit. Original participant pairing is
unverified, so crops are not claimed to be independent subjects. Reference labels
supervise calibration but do not independently annotate anatomical landmarks.
No segmentation retraining or loss integration is part of this audit.

## Exploratory amendment after the first result

The fixed local detector selected posterior micro-bends and required a +12 mm
training-median correction. Its initial validation results were already visible
when this amendment was made. Training outlines suggested that the visual
hypothesis concerns a broad bend rather than the strongest four-sample bend.

A second, fixed detector fits a continuous two-segment line to the smoothed upper
sagittal silhouette, excluding 3 mm at each end to reduce cap effects. Each segment
requires four samples. Require a descending posterior slope and a positive slope
change. Score the fractional reduction in squared residual relative to one straight
line, and choose the strongest fit. Repeat the same nested training calibration
and frozen validation transfer; no search over smoothing or tip-exclusion scales.
Keep both detector families and their full results. This amendment is explicitly
exploratory and cannot restore an untouched validation test.
