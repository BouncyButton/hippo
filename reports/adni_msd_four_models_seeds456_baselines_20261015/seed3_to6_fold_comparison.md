# Seed 3–6 baseline comparisons across folds

Final-budget validation hard Dice (%). Each cell is one completed run; Mean is the equal-weight mean of folds 0–2. Same original optimization-FLOP cap (31,976,450,667,090) per run, no augmentation, and the frozen cohort for each fold. Seed 3 came from job 702897; seeds 4–6 from job 703110. All 96 archives passed source/hash verification before cleanup.

## ADNI

| Seed | Fold | nnU-Net | SwinUNETR | UNETR | U-Net |
|---:|:---:|---:|---:|---:|---:|
| 3 | 0 | 76.12 | 65.33 | 46.33 | 7.85 |
| 3 | 1 | 78.40 | 69.66 | 48.55 | 8.25 |
| 3 | 2 | 76.01 | 69.49 | 49.79 | 10.34 |
| **3** | **Mean** | **76.84** | **68.16** | **48.23** | **8.81** |
| 4 | 0 | 75.01 | 64.02 | 51.92 | 65.97 |
| 4 | 1 | 79.09 | 69.97 | 55.25 | 70.92 |
| 4 | 2 | 76.47 | 68.66 | 55.80 | 69.29 |
| **4** | **Mean** | **76.86** | **67.55** | **54.32** | **68.73** |
| 5 | 0 | 75.81 | 62.99 | 63.02 | 71.47 |
| 5 | 1 | 79.22 | 68.93 | 68.55 | 74.79 |
| 5 | 2 | 77.52 | 67.10 | 66.24 | 75.92 |
| **5** | **Mean** | **77.52** | **66.34** | **65.93** | **74.06** |
| 6 | 0 | 73.46 | 66.44 | 53.37 | 72.65 |
| 6 | 1 | 76.81 | 70.92 | 56.80 | 76.82 |
| 6 | 2 | 75.41 | 71.55 | 54.93 | 75.62 |
| **6** | **Mean** | **75.22** | **69.64** | **55.03** | **75.03** |

## MSD

| Seed | Fold | nnU-Net | SwinUNETR | UNETR | U-Net |
|---:|:---:|---:|---:|---:|---:|
| 3 | 0 | 66.42 | 69.12 | 35.08 | 73.39 |
| 3 | 1 | 73.34 | 68.58 | 34.02 | 70.20 |
| 3 | 2 | 69.49 | 68.29 | 34.87 | 73.53 |
| **3** | **Mean** | **69.75** | **68.66** | **34.65** | **72.37** |
| 4 | 0 | 36.27 | 69.33 | 25.45 | 72.63 |
| 4 | 1 | 51.41 | 66.56 | 22.90 | 70.21 |
| 4 | 2 | 37.22 | 68.93 | 23.25 | 72.14 |
| **4** | **Mean** | **41.64** | **68.27** | **23.86** | **71.66** |
| 5 | 0 | 74.12 | 41.57 | 55.34 | 80.66 |
| 5 | 1 | 76.77 | 38.97 | 54.69 | 78.36 |
| 5 | 2 | 73.09 | 40.19 | 55.44 | 77.25 |
| **5** | **Mean** | **74.66** | **40.24** | **55.16** | **78.76** |
| 6 | 0 | 35.46 | 67.09 | 50.39 | 67.24 |
| 6 | 1 | 60.17 | 64.25 | 50.98 | 69.32 |
| 6 | 2 | 36.42 | 63.97 | 48.83 | 70.85 |
| **6** | **Mean** | **44.02** | **65.10** | **50.07** | **69.14** |

Seed 0–2 archived results are incomplete across the requested eight model/dataset groups. The available comparisons are in `reports/adni_seed_audit_20261015` and `reports/unetr_seed1_readonly_audit_20261015`; do not treat them as a complete seed grid.
