# Concavity-shoulder localization audit — fold 0

**Decision: the tested shoulder, signed-distance, and extrema detectors are not
reliable enough to enforce as cut-location constraints.** The observation motivated a useful hypothesis, but
these explicit geometric definitions do not add localization accuracy beyond a
simple training-derived position prior. This does not rule out a different 3-D
landmark detector or an image-based landmark with expert annotation.

Completed 23 September 2026 on **208 training and 52 validation crops**. Training
estimates use four-fold out-of-fold prediction, with an inner three-fold search for
position weighting. The final detector uses all training cases and training-only
four-fold selection, then stays frozen for validation. No segmentation network was
retrained and no loss was enabled.

## What was measured

In canonical sagittal geometry, increasing native y is anterior. The upper outline
often flattens from a descending posterior segment into the anterior region. We
tested its shoulder without supplying A/P class colours: all shape extraction sees
only the binary hippocampus union. Target cuts are the best-fit A/P separator planes
at c−0.5; native voxel spacing was verified as 1 mm RAS.

The original fixed detector compares slopes in two four-sample windows after 1-mm
smoothing, combining the whole sagittal silhouette with persistence in individual
sagittal slices. It often chooses posterior local bends. A broader two-line fit to
the overall silhouette was added after inspecting that result and training outlines.
This second definition is **explicitly exploratory**; both families are retained.
See the [protocol and amendment](PROTOCOL.md).

## Reference-mask results

Mean absolute cut error in millimetres; lower is better. Reference foreground is
oracle support, not a mask available independently at deployment.

| Method | Training OOF, n=208 | Validation, n=52 |
|---|---:|---:|
| Training-median relative position | **1.183** | **1.173** |
| Raw local shoulder | 10.942 | 12.577 |
| Local shoulder + training-derived offset | 4.510 | 4.481 |
| Local shoulder + offset + position prior | 1.284 | 1.308 |
| Raw broad shoulder (exploratory) | 10.288 | 10.712 |
| Broad shoulder + training-derived offset | 2.731 | 2.404 |
| Broad shoulder + offset + position prior | 1.192 | 1.212 |

The final training offsets were **+12 mm for the local detector** and **+10 mm for
the broad detector**. These large corrections show that the raw geometric maxima
usually identify a different part of the contour, not the annotated cut itself.
Both hybrids selected the strongest tested positional penalty (8). Their near-1-mm
accuracy should therefore not be attributed to independent anatomical detection.

On reference validation masks, the local hybrid helps/equal/harms 11/23/18 crops
relative to position alone; the broad hybrid gives 3/44/5. Their paired MAE changes
are +0.135 mm (crop-bootstrap 95% interval −0.077 to +0.346) and +0.038 mm
(−0.077 to +0.154), respectively. Neither shows an improvement.

## Transfer to predicted foreground

All three saved model sets contained all 52 validation crops and no training caches.
Thus the training audit uses reference masks; predicted-support transfer is measured
on validation. Cached labels were checked for exact equality with center-padded
native labels. Full prediction volumes were preserved, including predictions outside
the acquired crop; only the shape extractor uses the largest connected component.

| Validation foreground | Model's own cut | Position only | Local hybrid | Broad hybrid (exploratory) |
|---|---:|---:|---:|---:|
| Baseline used by the 3-D viewer | **0.962** | 1.058 | 1.308 | 1.096 |
| Early-stopped unaugmented | **1.019** | 1.096 | 1.288 | 1.077 |
| Early-stopped augmented | **0.904** | 1.038 | 1.288 | 1.038 |

On the augmented model, the local hybrid helps/equal/harms 11/15/26 crops relative
to the model cut; the broad hybrid gives 9/29/14. The broad hybrid's +0.135-mm MAE
change has a crop-bootstrap interval of −0.115 to +0.366 mm; this is not evidence
of a gain. No corrected segmentation was generated or evaluated.

## Signed-distance and extrema follow-up

At the user's request, four additional fixed feature families were tested using
the earlier first/last occupied-z representation. At each upper half-voxel face,
sample curvature from a smoothed signed-distance field (negative inside, positive
outside). Test both the AP level-curve curvature and 3-D normal divergence. For
each sagittal upper-extrema curve, also measure its perpendicular depth below its
upper convex-hull envelope; combine this depth with AP concavity curvature by a
geometric mean. Lower extrema and thickness are diagnostic only. The [SDF protocol](SDF_PROTOCOL.md)
was fixed before evaluating these feature families.

The SDF value itself is approximately zero at the surface; derivatives, rather
than the surface value, provide the concavity measurement. Rasterization can create
a flat notch whose depth maximum is central but whose curvature maxima lie on its
shoulders. This was explicitly checked with a synthetic shape.

All offsets below were fitted within training. The selected variant uses nested
selection over feature family and position strength; individual rows were not
chosen using validation performance.

| Method (training-offset corrected) | Training OOF MAE | Validation reference MAE |
|---|---:|---:|
| Position only | **1.183** | **1.173** |
| SDF AP curvature | 3.740 | 4.269 |
| SDF 3-D curvature | 7.322 | 8.462 |
| Extrema concavity depth | 2.591 | 2.673 |
| SDF AP curvature × depth (geometric mean) | 3.510 | 3.404 |
| Training-selected feature + position | 1.202 | **1.173** |

All four outer training folds selected **extrema depth**, with position weights
4, 8, 4, 8. The full-training fit also selected depth with weight 8 and a +7-mm
offset. No SDF curvature family won selection. Relative to position alone, the
selected method helps/equal/harms 8/188/12 training crops and 3/46/3 validation
crops. Validation MAE change is 0.000 mm, with crop-bootstrap 95% interval
−0.096 to +0.096 mm. Thus the geometry changes some individual cuts but adds no
net validation accuracy in this audit.

| Predicted validation support | Selected depth + position | Model's own cut |
|---|---:|---:|
| Viewer baseline | 1.115 | **0.962** |
| Early-stopped unaugmented | 1.058 | **1.019** |
| Early-stopped augmented | 1.038 | **0.904** |

For the augmented model, selected geometry helps/equal/harms 9/29/14 crops versus
the native cut. Its +0.135-mm mean error change has a crop-bootstrap interval
−0.115 to +0.385 mm. The four descriptors all produced nonzero scores on every
case/support pair, so this outcome is not driven by absent score curves.

The SDF AP measurement improves substantially over the original uncalibrated local
slope detector, but remains much worse than the relevant position/model controls.
These results do not justify using it as a trusted cut anchor. They also do not
test whether supervised SDF prediction could improve outer-boundary training;
that is a different target and requires a matched training experiment.

[Detailed SDF results](../../../evaluation_output/concavity_shoulder_fold0_20260923/SDF_RESULTS.md)
· [SDF per-case CSV](../../../evaluation_output/concavity_shoulder_fold0_20260923/sdf_case_results.csv)
· [SDF candidate scores](../../../evaluation_output/concavity_shoulder_fold0_20260923/sdf_candidate_scores.csv)

## Interpretation and limits

- The screenshot viewpoint is unknown. The audit tests canonical sagittal geometry,
  not the exact screen-space indentation under an arbitrary camera rotation.
- A shape can have several plausible shoulders. The largest local slope change and
  the best global hinge do not reliably select the particular transition of interest.
- There are 47 nonplanar reference labels in training and 11 in validation; all have
  a unique best-fit plane. Original labels were retained. Cut error summarizes the
  A/P boundary but is not a full segmentation metric.
- Every reference cut lies inside the local detector's complete-window candidate
  range. The local detector produced a positive score in every case, so failure is
  not explained by excluded targets or missing local candidates.
- The broad-fit detector has one zero-score training case. Its forced midpoint
  tie-break is counted in the metrics but is not evidence of detecting a shoulder.
- Validation fold 0 has been inspected repeatedly, and the broad fit was added
  after seeing initial validation results. Treat all findings as development evidence.
  Left/right participant pairing is unverified; counts and bootstrap intervals refer
  to crops, not known independent participants.

Before further training, the most informative next step is to mark the intended
shoulder on a blinded set of training volumes in 3-D and measure inter-rater agreement
and cut offset. That would distinguish failure of our geometric detector from
failure of the anatomical hypothesis.

## Outputs and verification

- [Visual gallery: all 260 reference cases and selected prediction examples](../../../evaluation_output/concavity_shoulder_fold0_20260923/index.html)
- [Detailed local-detector results](../../../evaluation_output/concavity_shoulder_fold0_20260923/RESULTS.md)
- [Detailed broad-detector follow-up](../../../evaluation_output/concavity_shoulder_fold0_20260923/BROAD_RESULTS.md)
- [Local per-case CSV](../../../evaluation_output/concavity_shoulder_fold0_20260923/case_results.csv)
- [Broad per-case CSV](../../../evaluation_output/concavity_shoulder_fold0_20260923/broad_case_results.csv)
- [All local candidate scores](../../../evaluation_output/concavity_shoulder_fold0_20260923/candidate_scores.csv)

Sixteen tests passed: synthetic local and broad bends, SDF sign and curvature,
upper-envelope depth, straight/convex controls, missing
tissue, empty/short inputs, padding/native coordinates, left/right mirroring,
independence from A/P labels, and metric direction. All three audits completed with
416 case/support rows each. All 416 input file hashes were rechecked and unchanged.
Representative figures were rendered and inspected.
Frozen calibration, exact case IDs, input hashes and split provenance are saved
alongside the results.

```bash
MPLCONFIGDIR=/tmp/hippo-mpl .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_concavity_shoulder
MPLCONFIGDIR=/tmp/hippo-mpl .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_broad_shoulder
MPLCONFIGDIR=/tmp/hippo-mpl .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_sdf_shoulder
.venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_concavity_shoulder.py thesis/new_constraints/uncal_fold/test_sdf_shoulder.py -q
```
