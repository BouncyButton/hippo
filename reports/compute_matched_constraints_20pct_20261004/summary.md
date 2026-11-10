# Calibrated compute-matched validation results

| Model/objective | Mean Dice | SD | Runs |
|---|---:|---:|---:|
| unet/dice | 0.4214 | 0.0000 | 1 |
| unet/dice_bands | 0.5474 | 0.0000 | 1 |
| unet/dice_bands_edge | 0.5280 | 0.0000 | 1 |
| unet/dice_edge | 0.4133 | 0.0000 | 1 |
| unet/dice_sdf | 0.6115 | 0.0000 | 1 |
| unet/dice_bands_sdf | 0.6527 | 0.0000 | 1 |
| unet/dice_bands_edge_sdf | 0.6497 | 0.0000 | 1 |
| unet/dice_edge_sdf | 0.5725 | 0.0000 | 1 |

Selected validation scores, not independent held-out test performance.
Fold/seed standard deviations are descriptive; runs are not independent datasets.
Warm-start training, calibration and all attempted updates are charged to each run.
