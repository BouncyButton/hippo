# Uncal-cut result under matched early stopping

The descriptor and its fitted coefficients are identical between audits. Only
the segmentation checkpoints and their predicted foreground masks changed.

| model foreground | policy | MAE | exact | within 1 | within 2 | p90 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| Unaugmented | Fixed 50 | 1.231 | 34.6% | 73.1% | 84.6% | 3.0 | 6 |
| Unaugmented | Early-stopped best | 1.212 | 38.5% | 75.0% | 86.5% | 4.0 | 6 |
| Augmented | Fixed 50 | 0.962 | 40.4% | 76.9% | 92.3% | 2.0 | 5 |
| Augmented | Early-stopped best | 1.115 | 46.2% | 86.5% | 92.3% | 2.0 | 19 |

## Matched early-stopping comparison

Augmentation is better/equal/worse in `17/30/5` cases (paired exploratory p=`0.0208`).
It improves exact and within-one localization, but one catastrophic endpoint
selection in `hippocampus_317` chooses slice `6` instead of `25`. That single `19`-slice
error inflates the augmented MAE. It cannot be removed or tuned away after
looking at validation labels; it is evidence that the soft position prior is
not a sufficient safety guard.

Without that case (reported only as a sensitivity analysis), augmented
early-stopping MAE is `0.765`
versus `1.157`
for the matched non-augmented checkpoint and
`0.922` for the historical
augmented fixed-50 checkpoint. The primary all-52 result remains the table above.

## Scientific conclusion

The anatomical evidence discovered earlier does not change: area transition,
superior-boundary rise/notch change, asymmetric new superior tissue, and the
superior T1 band retain exactly the same fitted weights. Early stopping and
augmentation improve the quality of the foreground supplied to those descriptors,
but the locator remains vulnerable to a high-scoring endpoint. It should remain
a soft, quality-gated LTN predicate rather than a hard cut rule.
