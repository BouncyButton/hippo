# MedSAM3 three-fold, three-seed replication

All 27 fine-tuning arms completed and each run passed the independent saved-mask/training audit.

**The strict cross-seed calibration gate did not pass.** This is an explicitly requested descriptive report of all observed runs, with the original threshold retained. No coefficients, predictions, protocols or runs were changed or excluded. The cause and effect of the calibration differences remain unresolved; this is not a fully compliant replication claim.

| Fold | Constraint | Maximum relative coefficient difference | Original check |
|---|---|---:|---|
| 0 | bands | 0.002946% | failed |
| 0 | edge | 0.024488% | failed |
| 1 | bands | 0.105425% | failed |
| 1 | edge | 0.030773% | failed |
| 2 | bands | 0.004593% | failed |
| 2 | edge | 0.009522% | failed |

| Fold | Seed | Validation volumes | Baseline Dice | Bands Dice | Bands + edge Dice |
|---|---:|---:|---:|---:|---:|
| 0 | 17 | 46 | 0.835453 | 0.849107 | 0.865451 |
| 0 | 83 | 46 | 0.823506 | 0.834957 | 0.851188 |
| 0 | 191 | 46 | 0.821586 | 0.820149 | 0.845669 |
| 1 | 17 | 52 | 0.835619 | 0.838681 | 0.844303 |
| 1 | 83 | 52 | 0.821328 | 0.832596 | 0.850066 |
| 1 | 191 | 52 | 0.834704 | 0.836706 | 0.825847 |
| 2 | 17 | 47 | 0.812242 | 0.803372 | 0.806707 |
| 2 | 83 | 47 | 0.839457 | 0.833626 | 0.840399 |
| 2 | 191 | 47 | 0.837791 | 0.808829 | 0.829483 |

| Contrast | Mean change (Dice points) | Exploratory 95% interval | Positive runs |
|---|---:|---|---:|
| bands_edge_minus_baseline | +1.083 | [-0.610, +2.757] | 6/9 |
| bands_minus_baseline | -0.041 | [-1.575, +1.121] | 5/9 |
| bands_edge_minus_bands | +1.123 | [+0.074, +2.078] | 8/9 |

Repeated case/seed predictions are not independent. With three fold blocks, the minimum two-sided exact sign-flip p-value is 0.25. Bootstrap intervals are exploratory, not a significance declaration.

Only three fold blocks; bootstrap tails are approximate and no patient-independent claim is possible.

The original pilot is excluded. All training volumes are excluded from every evaluation set. Training support sets differ across folds; each is held fixed across three new optimization seeds. Coefficients are calibrated from training data only with a fixed calibration seed per fold. No validation-based tuning or early stopping is used.
