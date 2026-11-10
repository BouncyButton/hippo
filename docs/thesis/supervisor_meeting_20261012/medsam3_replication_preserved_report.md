# Three-class MedSAM3 replication: audited validation results

Exploratory replication on folds 0, 1, and 3 requested after viewing binary fold outcomes and the multiclass fold-0 seed-46 pilot. These selected folds are not an unbiased full cross-validation estimate. All nine new comparisons are retained regardless of outcome; the seed-46 pilot is excluded from aggregate results. Patient identities are unavailable.

Primary Dice is the mean of anterior and posterior Dice, excluding background. Values first average volumes, then the three seeds within each fold. FP/FN below are disjoint foreground/background errors; A/P swaps are reported separately. Error units: mean voxels per volume.

| Fold | Arm | A/P Dice (%) | Gain (pp) | Seed SD (pp) | Foreground FP | Foreground FN | A→P | P→A | Total swaps | Positive seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 73.463 | +0.000 | 5.382 | 552.86 | 950.01 | 47.78 | 46.17 | 93.95 | 0/3 |
| 0 | bands | 78.356 | +4.892 | 1.443 | 507.98 | 682.09 | 71.75 | 43.56 | 115.31 | 3/3 |
| 0 | bands_edge | 79.599 | +6.136 | 0.809 | 586.41 | 608.83 | 33.07 | 56.03 | 89.09 | 3/3 |
| 1 | baseline | 73.887 | +0.000 | 2.634 | 399.67 | 945.88 | 18.38 | 102.85 | 121.23 | 0/3 |
| 1 | bands | 78.889 | +5.002 | 1.163 | 476.13 | 644.29 | 34.78 | 87.96 | 122.74 | 3/3 |
| 1 | bands_edge | 79.262 | +5.375 | 1.222 | 585.08 | 589.03 | 33.58 | 73.75 | 107.33 | 3/3 |
| 3 | baseline | 77.432 | +0.000 | 4.402 | 317.64 | 805.48 | 42.30 | 75.54 | 117.83 | 0/3 |
| 3 | bands | 79.034 | +1.603 | 2.478 | 563.31 | 553.70 | 62.77 | 68.02 | 130.79 | 2/3 |
| 3 | bands_edge | 81.504 | +4.072 | 0.539 | 519.54 | 466.57 | 53.25 | 61.90 | 115.15 | 3/3 |

## Class-specific validation errors

Per-class FP and FN include class swaps; do not add these to the disjoint foreground-error table.

| Fold | Arm | Class | Dice (%) | FP/volume | FN/volume |
|---|---|---|---:|---:|---:|
| 0 | baseline | background | 98.753 | 950.01 | 552.86 |
| 0 | baseline | anterior | 78.225 | 112.78 | 548.30 |
| 0 | baseline | posterior | 68.702 | 534.03 | 495.66 |
| 0 | bands | background | 99.014 | 682.09 | 507.98 |
| 0 | bands | anterior | 81.120 | 160.31 | 442.58 |
| 0 | bands | posterior | 75.592 | 462.98 | 354.83 |
| 0 | bands_edge | background | 99.010 | 608.83 | 586.41 |
| 0 | bands_edge | anterior | 83.129 | 182.06 | 373.69 |
| 0 | bands_edge | posterior | 76.069 | 493.45 | 324.24 |
| 1 | baseline | background | 98.884 | 945.88 | 399.67 |
| 1 | baseline | anterior | 76.779 | 288.01 | 467.41 |
| 1 | baseline | posterior | 70.995 | 232.89 | 599.71 |
| 1 | bands | background | 99.069 | 644.29 | 476.13 |
| 1 | bands | anterior | 81.019 | 296.16 | 341.35 |
| 1 | bands | posterior | 76.760 | 302.72 | 425.69 |
| 1 | bands_edge | background | 99.024 | 589.03 | 585.08 |
| 1 | bands_edge | anterior | 81.500 | 308.40 | 322.90 |
| 1 | bands_edge | posterior | 77.024 | 384.01 | 373.46 |
| 3 | baseline | background | 99.039 | 805.48 | 317.64 |
| 3 | baseline | anterior | 78.280 | 229.99 | 457.14 |
| 3 | baseline | posterior | 76.584 | 205.48 | 466.17 |
| 3 | bands | background | 99.039 | 553.70 | 563.31 |
| 3 | bands | anterior | 78.912 | 387.71 | 341.37 |
| 3 | bands | posterior | 79.157 | 306.39 | 343.11 |
| 3 | bands_edge | background | 99.151 | 466.57 | 519.54 |
| 3 | bands_edge | anterior | 83.332 | 286.22 | 277.57 |
| 3 | bands_edge | posterior | 79.675 | 348.47 | 304.15 |

## Individual seeds

| Fold | Seed | Arm | A/P Dice (%) | FP/volume | FN/volume | Swaps/volume |
|---|---|---|---:|---:|---:|---:|
| 0 | 17 | baseline | 67.277 | 953.93 | 1076.54 | 81.24 |
| 0 | 17 | bands | 80.017 | 331.91 | 753.30 | 83.43 |
| 0 | 17 | bands_edge | 80.088 | 462.48 | 666.74 | 85.52 |
| 0 | 83 | baseline | 77.072 | 312.41 | 920.00 | 74.20 |
| 0 | 83 | bands | 77.642 | 535.24 | 657.63 | 137.26 |
| 0 | 83 | bands_edge | 80.045 | 544.48 | 601.67 | 88.85 |
| 0 | 191 | baseline | 76.041 | 392.22 | 853.48 | 126.41 |
| 0 | 191 | bands | 77.409 | 656.78 | 635.35 | 125.24 |
| 0 | 191 | bands_edge | 78.665 | 752.28 | 558.09 | 92.91 |
| 1 | 17 | baseline | 73.963 | 204.02 | 1020.50 | 138.27 |
| 1 | 17 | bands | 79.729 | 330.31 | 695.25 | 119.52 |
| 1 | 17 | bands_edge | 80.430 | 400.13 | 612.46 | 121.56 |
| 1 | 83 | baseline | 76.483 | 372.81 | 905.67 | 77.77 |
| 1 | 83 | bands | 77.561 | 538.75 | 652.50 | 138.87 |
| 1 | 83 | bands_edge | 77.993 | 709.04 | 582.73 | 108.29 |
| 1 | 191 | baseline | 71.216 | 622.19 | 911.48 | 147.65 |
| 1 | 191 | bands | 79.377 | 559.35 | 585.13 | 109.85 |
| 1 | 191 | bands_edge | 79.364 | 646.06 | 571.90 | 92.15 |
| 3 | 17 | baseline | 80.601 | 304.34 | 716.55 | 82.25 |
| 3 | 17 | bands | 81.532 | 464.09 | 506.73 | 112.89 |
| 3 | 17 | bands_edge | 82.089 | 535.59 | 447.86 | 101.23 |
| 3 | 83 | baseline | 72.405 | 345.61 | 936.52 | 176.11 |
| 3 | 83 | bands | 76.577 | 581.48 | 643.91 | 151.02 |
| 3 | 83 | bands_edge | 81.028 | 440.91 | 499.43 | 141.75 |
| 3 | 191 | baseline | 79.290 | 302.98 | 763.36 | 95.14 |
| 3 | 191 | bands | 78.995 | 644.36 | 510.45 | 128.45 |
| 3 | 191 | bands_edge | 81.393 | 582.11 | 452.41 | 102.48 |

All saved predictions, raw confusion counts, input/output hashes, adapters, matched training traces and exact cross-seed calibration were checked before aggregation. The exploratory seed-46 pilot is excluded. Repeated seeds reuse each fold’s volumes and do not create independent patients.
