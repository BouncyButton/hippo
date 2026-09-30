# Uncal-foldedness discovery audit

This is a five-fold, subject-held-out audit on all 260 MSD Task04 labels. Every
shape feature is calculated from the undivided foreground union `(label > 0)`.
Class 1/2 labels are used only to derive the best-fitting first-anterior slice
against which predictions are scored. No network predictions enter this audit.

## Results

| method | MAE (slices) | exact | within 1 | within 2 | p90 error |
|---|---:|---:|---:|---:|---:|
| logistic_shape_only_gated | 0.877 | 38.5% | 81.2% | 93.8% | 2.0 |
| logistic_shape_plus_coordinate_gated | 0.881 | 37.3% | 82.3% | 93.8% | 2.0 |
| median_relative | 1.162 | 25.8% | 70.0% | 90.4% | 2.0 |
| selected_single_descriptor_gated | 1.523 | 26.2% | 58.8% | 77.7% | 3.0 |
| handcrafted_fold_transition_gated | 1.638 | 22.7% | 54.6% | 76.2% | 4.0 |
| logistic_shape_plus_coordinate | 1.700 | 35.0% | 78.1% | 89.6% | 3.0 |
| logistic_shape_only | 2.285 | 35.0% | 74.2% | 86.2% | 4.0 |
| logistic_coordinate_only | 2.404 | 8.1% | 28.5% | 51.9% | 4.0 |
| selected_single_descriptor | 6.662 | 4.6% | 18.8% | 28.1% | 15.0 |
| handcrafted_fold_transition | 9.519 | 6.9% | 19.2% | 23.5% | 18.1 |

## Most atypical third

Atypicality is the absolute distance between a validation subject's relative
cut and the median relative cut of that fold's training subjects.

| method | MAE | exact | within 1 | p90 error |
|---|---:|---:|---:|---:|
| median_relative | 2.267 | 0.0% | 9.3% | 3.0 |
| logistic_shape_only | 2.105 | 34.9% | 74.4% | 3.0 |
| logistic_shape_only_gated | 0.907 | 39.5% | 80.2% | 2.0 |

`median_relative` is the no-shape control. `handcrafted_fold_transition` is a
fixed mean of superior-contour asymmetry, height asymmetry, notch depth,
roughness and multi-run evidence. `selected_single_descriptor` chooses one
human-readable descriptor/rule using only the training subjects of each fold.
The logistic models are diagnostics: they test whether the descriptor vector
contains recoverable held-out signal, not whether it is already an LTN rule.
Methods ending in `_gated` restrict candidates to the 1st--99th percentile
of relative cut positions measured on that fold's training subjects. This
broad, training-only guard tests whether catastrophic end-slice extrema were
hiding useful local shape evidence; it is not a foldedness measurement.

The shape-only gated model has lower error than the median baseline in 113 cases, equal error in 79, and higher error in 68. The training-derived gate
contains the true validation cut in 253/260 cases (97.3%).
It must therefore remain a soft/support diagnostic rather than a hard anatomical
axiom. In particular, `hippocampus_164` lies outside that gate.

## Fold-wise selected single rules

- Fold 0: `max_current(top_left_minus_right)`
- Fold 1: `max_current(top_left_minus_right)`
- Fold 2: `max_current(top_left_minus_right)`
- Fold 3: `max_current(top_left_minus_right)`
- Fold 4: `max_current(top_left_minus_right)`

## Strongest standardized logistic coefficients

These are averaged across folds and are descriptive only.

| feature | mean coefficient | mean absolute coefficient |
|---|---:|---:|
| `current__top_left_minus_right` | 1.361 | 1.361 |
| `next_delta__area` | 0.759 | 0.759 |
| `delta__area` | 0.622 | 0.622 |
| `current__column_multirun_fraction` | 0.476 | 0.476 |
| `delta__top_left_minus_right` | -0.407 | 0.407 |
| `current__solidity` | 0.391 | 0.391 |
| `delta__eccentricity` | 0.322 | 0.322 |
| `current__height_asymmetry_abs` | -0.319 | 0.319 |
| `next_delta__top_asymmetry_abs` | -0.246 | 0.246 |
| `next_delta__eccentricity` | 0.230 | 0.230 |
| `delta__top_asymmetry_abs` | -0.230 | 0.230 |
| `current__perimeter_over_sqrt_area` | -0.221 | 0.221 |

## Interpretation rule

A foldedness formulation should move forward only if a shape-based method
beats the held-out median-relative baseline, with particular attention to
within-one-slice accuracy. A good logistic result with a poor fixed descriptor
means that shape contains signal but the proposed hand equation is not yet the
right grounding. Failure of both argues for examining MRI encoder features.

## Reproduction

```bash
.venv/bin/python -m thesis.new_constraints.uncal_fold.audit_foldedness
```

Cases: 260. All inputs were verified as stored RAS.
