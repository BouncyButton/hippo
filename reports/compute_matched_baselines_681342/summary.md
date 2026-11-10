# Compute-matched baseline validation results

Selected validation macro hard Dice, nine runs per model (three folds × three seeds).

| Model | Mean Dice | Sample SD | Interrupted runs |
| --- | ---: | ---: | ---: |
| unet | 0.8170 | 0.0040 | 0 |
| swinunetr | 0.6904 | 0.0566 | 0 |
| nnunet | 0.6938 | 0.2174 | 0 |

Dice is selected on validation data; this is not independent held-out test performance.
Standard deviations describe runs; overlapping folds and repeated seeds are not nine independent datasets.
Interrupted attempts consume allowance; inspect discarded_attempts before comparing trajectories.
