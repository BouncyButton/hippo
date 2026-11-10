# MNI shared-cap headroom confirmation

Completed: 26/36. Shared optimization FLOP cap: 95,929,352,001,270.

| Model | Cells | Training mean | Training range | Validation >50% | Validation mean | Train/val gap |
|---|---:|---:|---:|---:|---:|---:|
| nnunet | 7/9 | 85.35% | 80.87–89.40% | 7/7 | 70.55% | 14.80 pp |
| swinunetr | 7/9 | 86.05% | 85.30–86.82% | 7/7 | 75.11% | 10.94 pp |
| unetr_vit_b16 | 6/9 | 79.76% | 78.65–81.86% | 6/6 | 70.78% | 8.98 pp |
| basic_unet16 | 6/9 | 75.79% | 54.88–88.46% | 5/6 | 58.95% | 16.84 pp |

Fixed recipes selected on fold 0 seed 0 using training headroom, a validation floor and gap diagnostics. Validation is development data. Folds can overlap; SD is descriptive, not an independent-sample confidence interval.
