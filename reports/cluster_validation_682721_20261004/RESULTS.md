# U-Net validation results: job 682721

Retrieved 2026-10-04. All eight runs completed (Slurm COMPLETED, exit 0:0, elapsed 25m39s). Fold 0, seed 0; 10 training patients and 52 validation patients; 64³ inputs, batch size 1. Each run's hard cap is 8,138.082968869 GFLOPs, including training-only constraint-weight calibration.

Best validation hard Dice is the macro average of anterior and posterior scores across validation patients. All eight selected checkpoints were at their final training update.

| Objective | Best Dice (%) | Delta vs Dice (percentage points) | Anterior (%) | Posterior (%) | Updates | Epoch equivalents |
|---|---:|---:|---:|---:|---:|---:|
| Dice | 42.14 | +0.00 | 67.10 | 17.18 | 849 | 84.9 |
| Dice + bands | 54.74 | +12.60 | 68.39 | 41.10 | 840 | 84.0 |
| Dice + original edge | 41.33 | -0.82 | 66.88 | 15.77 | 840 | 84.0 |
| Dice + bands + original edge | 52.80 | +10.66 | 68.20 | 37.40 | 834 | 83.4 |
| Dice + signed distance | 61.15 | +19.01 | 66.89 | 55.41 | 845 | 84.5 |
| Dice + bands + signed distance | 65.27 | +23.13 | 68.58 | 61.96 | 839 | 83.9 |
| Dice + original edge + signed distance | 57.25 | +15.11 | 66.56 | 47.95 | 839 | 83.9 |
| Dice + bands + original edge + signed distance | 64.97 | +22.83 | 68.17 | 61.78 | 833 | 83.3 |

All configuration, selected validation, budget ledger, and calibration hashes match their completion records. Completion scores match selected validation summaries, ledger totals match completion totals, and all totals are within the cap. These results are one-fold, one-seed validation observations; they do not establish repeatability across seeds or folds.

Source run root: `/mnt/beegfsstudents/home/3160552/compute_matched_constraints20_682721`.
