Final validation Dice comparison: job 691005

Optimization cap per run: 31,976.45066709 GFLOPs; calibration excluded and reported separately. Inputs 64³, batch size 1, ten training cases per fold, 52 validation cases. Four models, fourteen objectives, folds 0/1/2, seeds 0/1/2. 448 new runs plus 56 retained fold 0/seed 0 pilot conditions.

All reported scores are final checkpoint Dice, not best checkpoint Dice. Gains are percentage-point differences against the paired model/fold/seed baseline. Original depth-weighted cross-entropy and original edge consistency are used. Timing policies were frozen before replication; the SDF timing differs by model/objective according to the selected pilot policy. nnU-Net retains its native Dice+CE objective and deep supervision.

All nine fold/seed combinations, including the selection pilot

Cells: mean final foreground Dice % (mean paired gain in percentage points). Bold indicates the highest mean for each model within this cohort.

| Objective | nnU-Net | SwinUNETR | UNETR | 3D U-Net width 16 |
|---|---:|---:|---:|---:|
| Baseline | 63.96 | 51.79 | 38.16 | 65.08 |
| + bands | 65.22 (+1.26) | 54.70 (+2.91) | 40.56 (+2.40) | 74.78 (+9.70) |
| + signed distance | 62.36 (-1.59) | 52.20 (+0.41) | 38.52 (+0.35) | 65.02 (-0.06) |
| + bands + signed distance | 65.12 (+1.16) | 55.73 (+3.93) | 41.41 (+3.25) | 74.72 (+9.64) |
| + bands + original edge | 64.95 (+1.00) | 54.88 (+3.08) | 40.69 (+2.53) | 75.12 (+10.04) |
| + bands + signed distance + original edge | 64.95 (+1.00) | 55.91 (+4.12) | 41.55 (+3.39) | 75.15 (+10.06) |
| + depth CE | 63.44 (-0.52) | 59.64 (+7.85) | 40.54 (+2.38) | 70.83 (+5.75) |
| + bands + depth CE | 65.47 (+1.51) | 62.88 (+11.09) | 42.28 (+4.12) | 75.52 (+10.44) |
| + original edge + depth CE | 63.25 (-0.71) | 60.01 (+8.22) | 40.60 (+2.44) | 70.70 (+5.62) |
| + signed distance + depth CE | 63.12 (-0.83) | 59.92 (+8.12) | 41.09 (+2.93) | 70.25 (+5.16) |
| + bands + original edge + depth CE | 65.42 (+1.46) | 63.11 (+11.32) | 42.31 (+4.15) | **76.10 (+11.02)** |
| + bands + signed distance + depth CE | **65.65 (+1.69)** | 63.33 (+11.53) | 43.19 (+5.03) | 75.53 (+10.45) |
| + original edge + signed distance + depth CE | 63.27 (-0.68) | 60.32 (+8.53) | 41.26 (+3.09) | 71.08 (+5.99) |
| + bands + original edge + signed distance + depth CE | 65.38 (+1.43) | **63.66 (+11.87)** | **43.35 (+5.19)** | 75.86 (+10.78) |

Baseline and highest observed mean: run-to-run standard deviation and paired improvement counts.

| Model | Baseline mean ± SD | Highest mean ± SD | Paired gain mean ± SD | Improved combinations |
|---|---:|---:|---:|---:|
| nnU-Net | 63.96 ± 10.53 | 65.65 ± 10.24 | +1.69 ± 1.18 | 9/9 |
| SwinUNETR | 51.79 ± 10.63 | 63.66 ± 9.75 | +11.87 ± 10.53 | 9/9 |
| UNETR | 38.16 ± 16.65 | 43.35 ± 21.68 | +5.19 ± 5.20 | 7/9 |
| 3D U-Net width 16 | 65.08 ± 6.59 | 76.10 ± 2.76 | +11.02 ± 4.07 | 9/9 |

Eight new replication combinations, excluding fold 0/seed 0

Cells: mean final foreground Dice % (mean paired gain in percentage points). Bold indicates the highest mean for each model within this cohort.

| Objective | nnU-Net | SwinUNETR | UNETR | 3D U-Net width 16 |
|---|---:|---:|---:|---:|
| Baseline | 63.57 | 52.94 | 37.54 | 64.47 |
| + bands | 64.57 (+1.00) | 56.14 (+3.20) | 39.83 (+2.29) | 74.63 (+10.17) |
| + signed distance | 62.24 (-1.33) | 53.28 (+0.34) | 37.71 (+0.17) | 64.47 (+0.01) |
| + bands + signed distance | 64.45 (+0.89) | 57.07 (+4.13) | 40.51 (+2.98) | 74.55 (+10.08) |
| + bands + original edge | 64.44 (+0.87) | 56.34 (+3.40) | 39.94 (+2.41) | 74.87 (+10.40) |
| + bands + signed distance + original edge | 64.42 (+0.85) | 57.30 (+4.36) | 40.71 (+3.17) | 74.88 (+10.41) |
| + depth CE | 63.19 (-0.37) | 59.66 (+6.72) | 40.04 (+2.51) | 70.62 (+6.15) |
| + bands + depth CE | 64.75 (+1.18) | 62.62 (+9.68) | 41.81 (+4.27) | 75.35 (+10.88) |
| + original edge + depth CE | 62.97 (-0.60) | 60.02 (+7.08) | 40.10 (+2.56) | 70.41 (+5.94) |
| + signed distance + depth CE | 62.79 (-0.77) | 59.97 (+7.03) | 40.51 (+2.97) | 69.93 (+5.46) |
| + bands + original edge + depth CE | 64.74 (+1.17) | 62.84 (+9.90) | 41.87 (+4.33) | **75.83 (+11.37)** |
| + bands + signed distance + depth CE | **64.95 (+1.38)** | 63.15 (+10.21) | 42.73 (+5.20) | 75.36 (+10.89) |
| + original edge + signed distance + depth CE | 62.81 (-0.76) | 60.37 (+7.42) | 40.67 (+3.13) | 70.78 (+6.31) |
| + bands + original edge + signed distance + depth CE | 64.77 (+1.20) | **63.45 (+10.51)** | **42.88 (+5.34)** | 75.61 (+11.14) |

Baseline and highest observed mean: run-to-run standard deviation and paired improvement counts.

| Model | Baseline mean ± SD | Highest mean ± SD | Paired gain mean ± SD | Improved combinations |
|---|---:|---:|---:|---:|
| nnU-Net | 63.57 ± 11.19 | 64.95 ± 10.72 | +1.38 ± 0.77 | 8/8 |
| SwinUNETR | 52.94 ± 10.75 | 63.45 ± 10.40 | +10.51 ± 10.39 | 8/8 |
| UNETR | 37.54 ± 17.69 | 42.88 ± 23.13 | +5.34 ± 5.53 | 6/8 |
| 3D U-Net width 16 | 64.47 ± 6.76 | 75.83 ± 2.82 | +11.37 ± 4.20 | 8/8 |

These are exploratory validation results. The highest observed objective is selected descriptively from this comparison; it is not a prespecified confirmatory winner. The selection pilot is not independent evidence. Validation sets overlap across folds, and seeds share data; the displayed standard deviations describe variation across runs and are not confidence intervals or significance tests.

Source scores and per-fold/seed aggregates: final_summary_20261010.json and final_aggregates_20261010.json. Full verified metadata backup and audit receipt accompany this report.
