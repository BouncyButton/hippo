# Fold-0 predicted-foreground uncal-cut audit

The compact geometry + T1-image + soft-position candidate was fitted on
the 208 fold-0 training ground-truth unions. The fitted coefficients were
frozen before application to the 52 validation cases. Validation A/P labels
were used only to define and score the target cut.

| foreground supplied to locator | MAE | exact | within 1 | within 2 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| Ground-truth union (reference) | 0.769 | 55.8% | 80.8% | 90.4% | 2.0 | 4 |
| Unaugmented predicted union | 1.212 | 38.5% | 75.0% | 86.5% | 4.0 | 6 |
| Augmented predicted union | 1.115 | 46.2% | 86.5% | 92.3% | 2.0 | 19 |

## Relative-position control

| foreground supplied to locator | median-prior MAE | descriptor minus prior MAE | descriptor better/equal/worse | paired p |
|---|---:|---:|---:|---:|
| Ground-truth union (reference) | 1.173 | -0.404 | 26/17/9 | 0.02076 |
| Unaugmented predicted union | 1.096 | +0.115 | 17/18/17 | 0.7071 |
| Augmented predicted union | 1.038 | +0.077 | 21/20/11 | 0.147 |

## Paired augmented-versus-unaugmented result

The augmented foreground gives a smaller/equal/larger cut error in `17/30/5` of the 52 paired cases.
Mean error change (augmented minus unaugmented): `-0.096` slices.
Exploratory paired Wilcoxon p-value: `0.0208`.

## Predicted-support quality and cleanup

The locator receives the largest 26-connected foreground component. This
is a prespecified hippocampus-support cleanup, not use of the A/P labels.

| model | mean union Dice | cases with islands | discarded voxels |
|---|---:|---:|---:|
| Unaugmented | 0.8962 | 3/52 | 27 |
| Augmented | 0.9080 | 3/52 | 16 |

## Interpretation

The ground-truth-union row measures the descriptor's localization ceiling on
this exact fold. The two predicted-union rows measure transfer through the
segmentation model's outer-boundary errors. A comparison between the augmented
and unaugmented rows therefore answers whether augmentation produces a support
on which this fixed anatomical descriptor can localize the annotated cut more
reliably; it does not retrain or tune the locator on validation predictions.

The descriptor improves materially over the relative-position control when its
input is the clean ground-truth union. That advantage disappears on predicted
foreground: it is slightly worse than the control for the unaugmented model and
ties it in mean error for the augmented model. The current candidate is therefore
appropriate as soft, quality-gated auxiliary evidence, but not as a hard or
self-sufficient uncal-apex rule.

## Artifacts

- `summary.json`: aggregate and paired metrics.
- `case_results.csv`: target, predicted cuts, errors, Dice, and cleanup per case.
- `absolute_error_comparison.png`: paired per-case errors.

## Reproduction

```bash
.venv/bin/python -m thesis.new_constraints.uncal_fold.audit_predicted_foreground
```
