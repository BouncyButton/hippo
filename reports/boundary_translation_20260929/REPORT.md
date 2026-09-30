# Frozen translation intervention

Fixed intervention; descriptive paired-case intervals on the same reused cohort. No new training. No validation-tuned views or thresholds.

| Model | Identity val Dice% | TTA val Dice% | Change pp | Net shell errors removed | Corrected / introduced | Input unclipped cases |
|---|---:|---:|---:|---:|---:|---:|
| dice | 86.2012 | 86.6888 | +0.4876 | 1631 | 4882 / 3251 | 52/52 |
| dice_aug | 87.3727 | 87.7181 | +0.3454 | 632 | 2732 / 2100 | 52/52 |
| sum | 86.2487 | 86.7759 | +0.5273 | 1655 | 4740 / 3085 | 52/52 |
| sum_aug | 87.4342 | 87.7930 | +0.3588 | 634 | 2822 / 2188 | 52/52 |

The geometry ratio counts GT faces, whereas the error totals count unique voxels; it provides surface-size context, not a directly comparable error rate.

train: 133114 outer faces and 3986 A/P faces, ratio 33.40; 146554 outer endpoint voxels and 7830 A/P endpoints.
validation: 139916 outer faces and 4108 A/P faces, ratio 34.06; 155343 outer endpoint voxels and 8041 A/P endpoints.
