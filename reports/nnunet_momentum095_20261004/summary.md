# Compute-matched baseline validation results

Selected validation macro hard Dice, 3 runs per model (1 folds × 3 seeds).

| Model | Mean Dice | Sample SD | Interrupted runs |
| --- | ---: | ---: | ---: |
| nnunet | 0.8512 | 0.0015 | 0 |

Dice is selected on validation data; this is not independent held-out test performance.
Standard deviations describe runs; overlapping folds and repeated seeds are not independent datasets.
Interrupted attempts consume allowance; inspect discarded_attempts before comparing trajectories.
