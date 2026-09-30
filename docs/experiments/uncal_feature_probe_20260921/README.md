# Frozen feature probe for the uncal-apex cut

The segmentation models were frozen. Layer, pooling region, and logistic
regularisation were selected by four-fold subject-wise cross-validation on
the 208 fold-0 training subjects. The 52 validation subjects were evaluated
once after selection. `feature_only` contains no explicit A/P coordinate.

| model | probe | MAE | exact | within 1 | within 2 | p90 | max | endpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unaugmented | feature_only | 0.981 | 0.346 | 0.788 | 0.942 | 2.0 | 5 | 0.000 |
| unaugmented | feature_plus_position | 0.981 | 0.346 | 0.788 | 0.942 | 2.0 | 5 | 0.000 |
| unaugmented | position_only | 1.212 | 0.288 | 0.635 | 0.923 | 2.0 | 5 | 0.000 |
| augmented | feature_only | 0.904 | 0.365 | 0.846 | 0.923 | 2.0 | 4 | 0.000 |
| augmented | feature_plus_position | 0.904 | 0.365 | 0.846 | 0.923 | 2.0 | 4 | 0.000 |
| augmented | position_only | 1.173 | 0.212 | 0.673 | 0.962 | 2.0 | 4 | 0.000 |

## Selected hidden representation

- **unaugmented:** `decoder1`, `union`, C=0.01. Feature-only was better/equal/worse than position-only in 19/26/7 cases (exploratory paired Wilcoxon p=0.0314).
- **augmented:** `decoder1`, `all_regions`, C=0.1. Feature-only was better/equal/worse than position-only in 22/22/8 cases (exploratory paired Wilcoxon p=0.0316).

For both checkpoints the selected layer was the final full-resolution
decoder. Position contributed less than 1.4% of total absolute coefficient
mass and did not change any validation prediction when added to the hidden
features.

## Handcrafted predicted-foreground comparator

| model | MAE | exact | within 1 | within 2 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| unaugmented | 1.212 | 0.385 | 0.750 | 0.865 | 4.0 | 6 |
| augmented | 1.115 | 0.462 | 0.865 | 0.923 | 2.0 | 19 |

## Interpretation rule

A feature-only improvement over position-only is evidence that the frozen
network contains local visual information about the fold. Improvement only
after adding position means the representation is useful mainly when anchored
by a dataset prior. Validation results must not be used to retune the probe.

The handcrafted comparator is reported in the adjacent predicted-foreground
audit; it was not used to select this probe.

## Limitations

The segmentation checkpoints were trained on the same 208 subjects used for
inner probe selection, so the very low training-CV errors are optimistic and
are not performance estimates. Only the untouched 52-case validation metrics
should be interpreted as generalisation. The logistic peak values are ranking
scores, not calibrated probabilities. Finally, pooling still depends on the
baseline's predicted foreground support.
