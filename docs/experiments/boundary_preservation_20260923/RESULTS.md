# Boundary preservation: completed 208-case audit

The matched saved models support the original premise that deep foreground
is already well recovered. E raw is worse overall while predicting more
foreground, not because it broadly loses the deep interior. Rendering removes
many false positives, but also removes real foreground near the surface.

Job 666692 completed successfully in 3:05. All 1,248 case/arm/metric reproduction
checks matched the frozen completed component audit exactly. No training,
checkpoint selection, or renderer modification was performed.

## The deep foreground is already recovered

Across 208 cases, the fixed reference foreground farther than 2 mm from the
6-neighbor inner reference surface contains 65,015 voxels.

| Model | Missed deep foreground voxels | Pooled deep foreground recall | Deep A/P swaps |
|---|---:|---:|---:|
| A | 41 | 99.9369% | 2,088 |
| E raw | 34 | 99.9477% | 2,150 |
| E final | 43 | 99.9339% | 2,152 |

This supports preservation of deep foreground occupancy, not perfect anterior/
posterior labeling or anatomical validity. Thin anatomy is generally outside
this deep-interior definition. Net counts do not imply identical masks: A to E
final fixes 21 deep foreground misses but introduces 23, in five cases with at
least one new miss (maximum eight new deep misses in a case). Rendering alone
fixes two and introduces eleven, also across five cases with new misses.

## Where the raw system changes

| Pooled voxel error | A | E raw | E final |
|---|---:|---:|---:|
| False-positive foreground | 87,880 | 125,859 | 81,778 |
| Missed foreground | 71,949 | 49,281 | 73,988 |
| Anterior/posterior swaps | 17,476 | 18,390 | 17,637 |

Relative to A, E raw has 37,979 extra FP voxels and 22,668 fewer FN voxels.
Its signed volume error is higher in every case. This is consistent with an
expansive raw prediction followed by renderer suppression. It does not isolate
which loss or gradient path caused that behavior.

For A, 97.9603% of foreground/background errors lie within 2 mm of the reference
surface under this definition. This is a reference-label analysis; the same
coverage cannot be assumed for a band around a predicted surface.

## Exact changes caused by the renderer

| Reference region | FP voxels removed | New FP voxels | Missed true voxels recovered | Previously present true voxels deleted |
|---|---:|---:|---:|---:|
| 0–1 mm, both sides | 28,893 | 281 | 245 | 24,828 |
| >1–2 mm, both sides | 10,328 | 77 | 7 | 122 |
| Foreground >2 mm | 0 | 0 | 2 | 11 |
| Background >2 mm | 5,246 | 28 | 0 | 0 |
| All | 44,467 | 386 | 254 | 24,961 |

Foreground preservation is measured independently of whether the raw anterior/
posterior label was correct. Of the 24,961 deleted true foreground voxels,
24,181 had the exactly correct raw three-class label and 780 were raw A/P swaps.
There are no direct anterior-to-posterior or posterior-to-anterior switches
between E raw and E final, as required by the renderer architecture.

Thus rendering performs substantial useful cleanup, particularly away from the
immediate surface, but pays for it with substantial near-surface foreground
loss. These counts include all ray lengths; they do not establish that thin
rays account for most deletions or most ASSD error.

## Final benefit, churn, and heterogeneity

Relative to A, E final corrects 32,335 foreground/background errors and creates
28,272, for a net reduction of 4,063 (19.5337/case). It reduces FP by 6,102 but
increases FN by 2,039. Within 2 mm, it makes 31,055 union fixes and 27,407 union
breaks. Exact three-class errors fall by 3,902 despite 161 more A/P swaps.

Case-mean ASSD changes from 0.495716 to 0.483148 mm; union Dice changes from
0.883713 to 0.886133. ASSD worsens in 69/208 cases. Foreground/background error
count worsens in 73 cases, improves in 134, and ties in one.

| Fold | Final minus A union errors, pooled | Final minus A mean ASSD, mm |
|---|---:|---:|
| 1 | +64 | -0.002029 |
| 2 | -405 | -0.003922 |
| 3 | -1,465 | -0.019021 |
| 4 | -2,257 | -0.025300 |

Fold 1 illustrates why a better average ASSD does not imply fewer voxel errors.
Voxel-count changes also do not attribute ASSD changes: the surface geometry
and nearest-point distances change jointly.

The conditional paired case-bootstrap interval for the ASSD difference is
[-0.017376, -0.008201] mm. For the mean net union-error difference it is
[-27.4038, -12.3940] voxels/case. These intervals do not cover seed uncertainty,
overlapping training sets, or prior method selection. This remains internal
development evidence with one trained backbone seed.

## Implication for the next experiment

Broad deep-interior damage is not supported as the main failure here. The
useful target is selective near-surface correction: preserve the background
cleanup while avoiding deletion of true foreground. Increasing global rescue
strength is not sufficient evidence of progress.

The signed-gradient audit says a preservation signal already exists at sampled
E outputs. Its shared-head SGD responses show that stronger rescue can raise
thin and empty presence together. The proposed H route is an appropriate
mechanism test, but the additional subtraction in GRADIENT_FOLLOWUP_REVIEW.md
predicts a similarly shared increase for its head-only SGD direction. Check
actual AdamW updates and thin-versus-empty separation before interpreting H as
a remedy. Retain matched FP-voxel and FP-ray guards and a scalar-bias control.

H does not freeze the backbone: it retains indirect rendered-loss gradients
through the head's inputs, as well as the original explicit-loss routes. V
removes the indirect rendered route. Neither is automatically a preservation
guarantee or evidence of better learned edge positions.

## Artifacts and verification

- `results_666692/REPORT.md`: automatically generated comparison tables.
- `results_666692/summary.json`: pooled transitions, case means, intervals.
- `results_666692/cases.json` and `case_transitions.csv`: paired case records.
- `results_666692/provenance.json`: source/data/checkpoint/script hashes.
- `results_666692/reproduction.json`: all 1,248 comparison checks plus original
  historical metrics. The small pre-existing rendered-E discrepancy from the
  original historical mean 0.483144 is retained, not silently discarded.

Four local implementation tests passed, plus a separate 52-case baseline-mask
integration check. Matrix accounting and partition identities are asserted
during the audit. The final matrices independently passed the check that the
renderer makes no direct A/P switches.

The first proposed transfer containing per-case reference data was rejected by
automatic approval review. Execution used an approved code-only transfer and
built its reference from data already on the user's cluster. Numerical outputs
were retrieved into the workspace after successful completion.
