# Does anything besides the label locate the one-voxel errors? — 22 September 2026

Question: can the prediction's own sagittal step fit (or the image) tell which
side of the boundary a one-voxel error belongs to? This decides whether the
step-fit idea can supply a training signal beyond the labels.
Data: 52 fold-0 validation cases, hard masks only (no probabilities saved).
Baseline = `baseline_seed0`, Aug = `augmentation_seed0`.

| Check | Result |
| --- | --- |
| Error voxels at a column's top/bottom z extreme | 80.3% (baseline), 83.1% (Aug) |
| Snap prediction to its own decreasing step fit: changed voxels that fix an error | 34.5% / 31.5%; Dice 0.8961→0.8916, 49/52 cases worse |
| Same, only deviations ≥2 / ≥3 voxels | precision 33.8% / 39.7%; still net negative |
| Same with 3-section (x±1) smoothing before the fit | fixes 2,919, breaks 4,045 column extremes |
| Wrong column extremes where the fit does not move the extreme | 86% (19,865 / 23,099) |
| Reference itself changed by snapping to its own fit | 3.95% of foreground voxels |
| Wrong columns adjacent to a reference step transition | 78.6%, vs 73.9% of all columns |
| Image edge strength at wrong extremes: reference edge stronger than predicted edge | 49.0% |
| Error voxels whose intensity is closer to the class the reference assigns | 52.7% |
| Signed top/bottom error (mean, voxels) | +0.02 / −0.09 (baseline), +0.05 / +0.03 (Aug) |
| Dice between the two models vs Dice with the reference | 0.953 vs 0.896 / 0.908 |
| Baseline error voxels where Aug makes the same call | 71.3% |

## Reading

Most errors are a shifted but still valid staircase: the predicted plateau is
one voxel off, or a step sits one y position off. These contours satisfy the
decreasing-step prior, so fitting the prediction to that prior cannot detect
them. Where the fit does move a contour, it is wrong about two times in three,
partly because the reference itself is not monotone at the one-voxel scale.

At the disputed voxels, local intensity is at chance: edge strength 49% and
region likeness 53%. The errors show no side bias. Two models trained
differently agree with each other far better than either agrees with the
annotation, and they share 71% of their errors. Both models used the same
training labels and fold, so this is not an inter-rater study. The pattern is
still the signature of an annotation/partial-volume floor rather than a
learnable, prior-detectable mistake.

Limits: hard masks only; simple intensity features, not the network's context;
one fold; the two models are not independent raters.

## Reproduce
```bash
python3 scripts/audit_self_step_correction.py experiments/uncal_fold_early_stopping_20260921/voxel_audit/baseline_seed0/error_maps 1
python3 scripts/audit_boundary_intensity.py experiments/uncal_fold_early_stopping_20260921/voxel_audit/baseline_seed0/error_maps
```

## Follow-up: better step models (smooth surface, rasterised)

Hypothesis: the staircase is the rasterisation of a smooth sloping surface, so
a smooth fit (pooling many columns and neighbouring sections) could place
steps correctly where isotonic regression cannot. Test: fit each model to the
upper/lower height maps and round. "Ref reproduced" fits the **reference to
itself**, which is an upper bound on what the model can represent. Fixes and
breaks come from fitting the prediction. Baseline run; 23,165 wrong predicted
column extremes.

| Model | Ref reproduced | Fixes | Breaks | Net |
| --- | ---: | ---: | ---: | ---: |
| isotonic (1D) | 0.896 | 1,058 | 1,812 | −754 |
| line per section | 0.445 | 5,814 | 16,650 | −10,836 |
| quadratic per section | 0.543 | 4,766 | 13,566 | −8,800 |
| cubic per section | 0.629 | 4,580 | 10,751 | −6,171 |
| Gaussian 2D σx=1, σy=1 | 0.770 | 2,750 | 3,366 | −616 |
| Gaussian 2D σx=1, σy=2 | 0.657 | 4,352 | 6,920 | −2,568 |
| Gaussian 2D σx=2, σy=2 | 0.563 | 5,694 | 10,314 | −4,620 |
| Gaussian 1D along y σ=1.5 | 0.796 | 2,138 | 3,205 | −1,067 |

No tested smooth model reproduces the reference contour from the reference
itself better than 80% of column extremes. The annotation has one-voxel
irregularities that are not explained by a smooth surface, so these models
cannot recover them from the prediction either. Criterion for any future step
model: reference self-reproduction ≥ about 0.97 **and** positive net fixes on
predictions. Script: `scripts/audit_smooth_surface_steps.py`.
