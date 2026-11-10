# Calibrated compute-matched validation results

| Model/objective | Mean Dice | SD | Runs |
|---|---:|---:|---:|
| swinunetr/dice | 0.1412 | 0.0000 | 1 |
| swinunetr/dice_bands | 0.1386 | 0.0000 | 1 |
| swinunetr/dice_sdf | 0.1378 | 0.0000 | 1 |
| swinunetr/dice_bands_sdf | 0.1381 | 0.0000 | 1 |
| swinunetr/dice_bands_edge | 0.1375 | 0.0000 | 1 |
| swinunetr/dice_bands_edge_sdf | 0.1374 | 0.0000 | 1 |

Selected validation scores, not independent held-out test performance.
Fold/seed standard deviations are descriptive; runs are not independent datasets.
Warm-start training, calibration and all attempted updates are charged to each run.
