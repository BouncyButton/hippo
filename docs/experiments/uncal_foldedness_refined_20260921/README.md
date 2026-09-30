# Refined uncal-fold descriptor audit

Five-fold subject-held-out results on all 260 Task04 training masks/images.
The refined geometry is mirror-invariant because the release mixes left/right
hippocampus crops without reliable laterality metadata.

| method | MAE | exact | within 1 | within 2 | p90 |
|---|---:|---:|---:|---:|---:|
| compact_geometry_image_plus_soft_position | 0.969 | 40.0% | 79.2% | 89.2% | 3.0 |
| clear_anatomical_plus_soft_position | 1.081 | 39.6% | 73.8% | 87.3% | 3.0 |
| median_relative | 1.162 | 25.8% | 70.0% | 90.4% | 2.0 |
| compact_geometry_plus_soft_position | 1.212 | 35.4% | 70.8% | 84.6% | 3.0 |
| compact_fixed_formula | 1.396 | 35.0% | 67.7% | 81.5% | 4.0 |
| refined_shape_plus_image | 1.888 | 34.2% | 75.4% | 86.9% | 3.0 |
| clear_anatomical | 3.815 | 29.6% | 58.5% | 70.8% | 18.0 |
| refined_shape | 3.877 | 27.7% | 58.5% | 71.2% | 18.0 |
| image_only | 5.735 | 24.6% | 57.3% | 67.3% | 21.0 |
| compact_geometry | 7.323 | 22.7% | 46.2% | 57.3% | 21.0 |
| clear_anatomical_sparse | 12.077 | 14.6% | 30.8% | 35.8% | 23.0 |

## Most atypical third

| method | MAE | exact | within 1 | p90 |
|---|---:|---:|---:|---:|
| compact_geometry_image_plus_soft_position | 1.140 | 33.7% | 75.6% | 3.0 |
| clear_anatomical_plus_soft_position | 1.267 | 36.0% | 69.8% | 3.0 |
| median_relative | 2.267 | 0.0% | 9.3% | 3.0 |
| compact_geometry_plus_soft_position | 1.372 | 30.2% | 70.9% | 3.5 |
| compact_fixed_formula | 1.500 | 32.6% | 70.9% | 4.0 |
| refined_shape_plus_image | 1.965 | 30.2% | 74.4% | 3.5 |
| clear_anatomical | 3.791 | 30.2% | 60.5% | 16.0 |
| refined_shape | 3.884 | 26.7% | 58.1% | 17.0 |
| image_only | 5.686 | 18.6% | 57.0% | 21.0 |
| compact_geometry | 6.302 | 20.9% | 51.2% | 20.5 |
| clear_anatomical_sparse | 11.547 | 15.1% | 29.1% | 23.0 |

## Stable coefficients in the best compact model

| feature | mean coefficient | mean absolute | nonzero folds |
|---|---:|---:|---:|
| `position__soft_log_prior` | 1.359 | 1.359 | 5/5 |
| `clear_shape_next_delta__area` | 0.803 | 0.803 | 5/5 |
| `image_current__superior_band_left_right_difference_abs` | 0.678 | 0.678 | 5/5 |
| `image_current__superior_band_intensity_mean` | -0.422 | 0.422 | 5/5 |
| `image_current__mask_gradient_p90` | 0.383 | 0.383 | 5/5 |
| `transition__superior_boundary_rise_max` | 0.313 | 0.313 | 5/5 |
| `clear_shape_delta__area` | 0.312 | 0.312 | 5/5 |
| `image_current__superior_band_intensity_std` | 0.227 | 0.227 | 5/5 |
| `clear_shape_delta__superior_notch_depth` | 0.208 | 0.208 | 5/5 |
| `transition__novel_superior_side_difference` | 0.193 | 0.193 | 5/5 |
| `image_current__central_superior_band_intensity_mean` | -0.124 | 0.124 | 5/5 |
| `transition__slice_dice` | 0.109 | 0.109 | 5/5 |
| `image_delta__superior_band_intensity_mean` | 0.077 | 0.077 | 5/5 |
| `clear_shape_current__multirun_side_max` | 0.038 | 0.038 | 5/5 |

## Interpretation

The repeatable evidence is a transition pattern, not one static foldedness
number: cross-sectional area grows across the current and following slice; the
superior boundary rises or changes notch configuration; and the T1 band
immediately above the hippocampus becomes darker, more asymmetric, and more
edge-rich. Shape-only and image-only scores have catastrophic end-slice
outliers. The compact combination works only with a soft position prior, which
must remain defeasible rather than becoming a hard anatomical axiom.

The failed sparse and fixed-formula controls show that the current result does
not justify calling any single scalar an uncal-apex detector. The compact score
is the appropriate candidate for a fuzzy LTN predicate, followed by a baseline
comparison on predicted rather than ground-truth foreground masks.

## Reproduction

```bash
.venv/bin/python -m thesis.new_constraints.uncal_fold.audit_refined
```
