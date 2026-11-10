# Calibrated compute-matched validation results

| Model/objective | Mean Dice | SD | Runs |
|---|---:|---:|---:|
| unet/dice | 0.7699 | 0.0000 | 1 |
| unet/dice_bands | 0.7789 | 0.0000 | 1 |
| unet/dice_sdf | 0.7744 | 0.0000 | 1 |
| unet/dice_bands_sdf | 0.7785 | 0.0000 | 1 |
| unet/dice_bands_edge | 0.7779 | 0.0000 | 1 |
| unet/dice_bands_edge_sdf | 0.7794 | 0.0000 | 1 |

Selected validation scores, not independent held-out test performance.
Fold/seed standard deviations are descriptive; runs are not independent datasets.
Warm-start training, calibration and all attempted updates are charged to each run.
