# Frozen translation intervention

Fixed intervention; descriptive paired-case intervals on the same reused cohort. No new training. No validation-tuned views or thresholds.

| Model | Identity val Dice% | TTA val Dice% | Change pp | Net shell errors removed | Corrected / introduced | Input unclipped cases |
|---|---:|---:|---:|---:|---:|---:|
| dice | 86.0414 | 86.6826 | +0.6412 | 1944 | 5314 / 3370 | 52/52 |
| dice_aug | 87.4337 | 87.8747 | +0.4410 | 756 | 2765 / 2009 | 52/52 |

The geometry ratio counts GT faces, whereas the error totals count unique voxels; it provides surface-size context, not a directly comparable error rate.

train: 133114 outer faces and 3986 A/P faces, ratio 33.40; 146554 outer endpoint voxels and 7830 A/P endpoints.
validation: 139916 outer faces and 4108 A/P faces, ratio 34.06; 155343 outer endpoint voxels and 8041 A/P endpoints.
