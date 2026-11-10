Final mean foreground validation Dice (%), gains in percentage points versus each model baseline. Exploratory fold 0 / seed 0; ten training and 52 validation volumes. Optimization ceiling 31,976.45066709 GFLOPs; calibration separately reported and excluded. All signed-distance objectives use the new approximately 56%-of-allowance delay. Depth means original depth-weighted cross-entropy; edge means original edge consistency. Retained nnU-Net constraint rows use early calibration. Bold marks each model’s best final result.

| Objective | nnU-Net | SwinUNETR | UNETR | 3D U-Net, width 16 |
|---|---:|---:|---:|---:|
| Baseline | 67.07 | 42.60 | 43.15 | 70.01 |
| + bands | 70.38 (+3.31) | 43.15 (+0.55) | 46.39 (+3.24) | 75.95 (+5.94) |
| + signed distance | 63.39 (-3.68) | 42.89 (+0.29) | 44.06 (+0.91) | 69.41 (-0.60) |
| + bands + signed distance | 70.44 (+3.37) | 44.33 (+1.73) | **47.70 (+4.56)** | 75.83 (+5.82) |
| + bands + original edge | 69.08 (+2.01) | 43.17 (+0.57) | 46.65 (+3.50) | 77.18 (+7.17) |
| + bands + signed distance + original edge | 69.23 (+2.16) | 44.33 (+1.73) | 47.61 (+4.46) | 77.30 (+7.29) |
| + depth-weighted CE | 65.37 (-1.70) | 59.49 (+16.89) | 44.53 (+1.39) | 72.53 (+2.52) |
| + bands + depth-weighted CE | 71.23 (+4.16) | 64.96 (+22.36) | 46.00 (+2.85) | 76.87 (+6.86) |
| + original edge + depth-weighted CE | 65.47 (-1.60) | 59.90 (+17.31) | 44.64 (+1.49) | 73.06 (+3.05) |
| + signed distance + depth-weighted CE | 65.75 (-1.32) | 59.47 (+16.87) | 45.74 (+2.59) | 72.78 (+2.77) |
| + bands + original edge + depth-weighted CE | 70.83 (+3.77) | 65.27 (+22.67) | 45.86 (+2.71) | **78.26 (+8.25)** |
| + bands + signed distance + depth-weighted CE | **71.25 (+4.18)** | 64.75 (+22.15) | 46.87 (+3.72) | 76.89 (+6.89) |
| + original edge + signed distance + depth-weighted CE | 67.01 (-0.06) | 59.96 (+17.36) | 45.94 (+2.80) | 73.45 (+3.44) |
| + bands + signed distance + original edge + depth-weighted CE | 70.28 (+3.21) | **65.30 (+22.70)** | 47.13 (+3.98) | 77.90 (+7.89) |
