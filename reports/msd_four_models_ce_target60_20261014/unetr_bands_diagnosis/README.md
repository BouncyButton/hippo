# UNETR bands60 diagnosis

Both runs use the same initialization, 10 training/52 validation crops, no augmentation, 245 updates, optimizer/LR, precision and dataset/source hashes. Logged loss and Dice match exactly through updates [12, 24, 36, 49]. The difference begins after bands activate at update51 and ramp to full weight by update100.

| Metric | Dice+CE | +bands60 |
|---|---:|---:|
| validation_macro_dice_percent | 43.20 | 33.51 |
| training_macro_dice_percent | 57.88 | 39.00 |
| best_macro_dice_percent | 45.02 | 35.12 |
| validation_anterior_dice_percent | 21.14 | 12.49 |
| validation_posterior_dice_percent | 65.26 | 54.53 |
| binary_foreground_dice_percent | 50.95 | 57.30 |
| foreground_fn | 71610.00 | 56244.00 |
| foreground_fp | 168000.00 | 169560.00 |
| anterior_to_posterior | 11584.00 | 43282.00 |
| posterior_to_anterior | 203.00 | 117.00 |

The bands objective supervises the union of anterior and posterior against background. Its loss is invariant to exchanging anterior/posterior logits or labels, verified using the archived implementation. Thus better external foreground segmentation can coexist with worse anatomical classification. Observed binary foreground Dice rises, while A/P swaps increase and both class Dice scores fall. The degradation also occurs on training crops, so it is not solely validation overfitting.

Calibration achieved median 60% and p95 63.82%, below the 75% ceiling, on all ten training crops at update 50. The coefficient then remains fixed; this is not a guarantee of a 60% parameter-gradient contribution during later training. The learning curves initially improve with bands, then lose to baseline after about 184 updates. Best-checkpoint Dice is also lower, so final-checkpoint selection alone does not explain the outcome.

Interpretation: a plausible explanation is excessive emphasis on the external boundary before A/P discrimination is stable. The four requested 10/20% runs test strength while preserving all other settings. This single pair does not prove an architecture-wide weakness or establish an optimal weight.
