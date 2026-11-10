Replication status: ready_not_submitted; 2/448 new runs complete; 56 pilot conditions retained.

Final foreground validation Dice. Optimization cap: 31,976.45066709 GFLOPs.
Calibration is reported separately. SDF timing was frozen from fold 0 / seed 0 pilot results.
Each completed run retains its configuration, calibration, case metrics and verified compute ledger.
Model checkpoints are verified before discard and cannot be used for later inference.

| Model | Objective | Fold | Seed | Dice % | Gain pp | Updates | Epochs |
|---|---|---:|---:|---:|---:|---:|---:|
| nnunet | dice | 0 | 0 | 67.07 | +0.00 | 90 | 9.0 |
| nnunet | dice_bands | 0 | 0 | 70.38 | +3.31 | 89 | 8.9 |
| nnunet | dice_sdf | 0 | 0 | 63.39 | -3.68 | 89 | 8.9 |
| nnunet | dice_bands_sdf | 0 | 0 | 70.44 | +3.37 | 89 | 8.9 |
| nnunet | dice_bands_edge | 0 | 0 | 69.08 | +2.01 | 89 | 8.9 |
| nnunet | dice_bands_edge_sdf | 0 | 0 | 69.23 | +2.16 | 89 | 8.9 |
| nnunet | dice_depth | 0 | 0 | 65.37 | -1.70 | 89 | 8.9 |
| nnunet | dice_bands_depth | 0 | 0 | 71.23 | +4.16 | 89 | 8.9 |
| nnunet | dice_edge_depth | 0 | 0 | 65.47 | -1.60 | 89 | 8.9 |
| nnunet | dice_sdf_depth | 0 | 0 | 65.75 | -1.32 | 89 | 8.9 |
| nnunet | dice_bands_edge_depth | 0 | 0 | 70.83 | +3.77 | 89 | 8.9 |
| nnunet | dice_bands_sdf_depth | 0 | 0 | 71.25 | +4.18 | 89 | 8.9 |
| nnunet | dice_edge_sdf_depth | 0 | 0 | 67.01 | -0.06 | 89 | 8.9 |
| nnunet | dice_bands_edge_sdf_depth | 0 | 0 | 70.28 | +3.21 | 89 | 8.9 |
| swinunetr | dice | 0 | 0 | 42.60 | +0.00 | 206 | 20.6 |
| swinunetr | dice_bands | 0 | 0 | 43.15 | +0.55 | 206 | 20.6 |
| swinunetr | dice_sdf | 0 | 0 | 43.58 | +0.98 | 206 | 20.6 |
| swinunetr | dice_bands_sdf | 0 | 0 | 44.96 | +2.36 | 206 | 20.6 |
| swinunetr | dice_bands_edge | 0 | 0 | 43.17 | +0.57 | 206 | 20.6 |
| swinunetr | dice_bands_edge_sdf | 0 | 0 | 44.81 | +2.21 | 206 | 20.6 |
| swinunetr | dice_depth | 0 | 0 | 59.49 | +16.89 | 206 | 20.6 |
| swinunetr | dice_bands_depth | 0 | 0 | 64.96 | +22.36 | 206 | 20.6 |
| swinunetr | dice_edge_depth | 0 | 0 | 59.90 | +17.31 | 206 | 20.6 |
| swinunetr | dice_sdf_depth | 0 | 0 | 59.47 | +16.87 | 206 | 20.6 |
| swinunetr | dice_bands_edge_depth | 0 | 0 | 65.27 | +22.67 | 206 | 20.6 |
| swinunetr | dice_bands_sdf_depth | 0 | 0 | 64.75 | +22.15 | 206 | 20.6 |
| swinunetr | dice_edge_sdf_depth | 0 | 0 | 59.96 | +17.36 | 206 | 20.6 |
| swinunetr | dice_bands_edge_sdf_depth | 0 | 0 | 65.30 | +22.70 | 206 | 20.6 |
| unetr_vit_b16 | dice | 0 | 0 | 43.15 | +0.00 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands | 0 | 0 | 46.39 | +3.24 | 245 | 24.5 |
| unetr_vit_b16 | dice_sdf | 0 | 0 | 44.99 | +1.85 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_sdf | 0 | 0 | 48.61 | +5.46 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_edge | 0 | 0 | 46.65 | +3.50 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_edge_sdf | 0 | 0 | 48.33 | +5.18 | 245 | 24.5 |
| unetr_vit_b16 | dice_depth | 0 | 0 | 44.53 | +1.39 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_depth | 0 | 0 | 46.00 | +2.85 | 245 | 24.5 |
| unetr_vit_b16 | dice_edge_depth | 0 | 0 | 44.64 | +1.49 | 245 | 24.5 |
| unetr_vit_b16 | dice_sdf_depth | 0 | 0 | 45.74 | +2.59 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_edge_depth | 0 | 0 | 45.86 | +2.71 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_sdf_depth | 0 | 0 | 46.87 | +3.72 | 245 | 24.5 |
| unetr_vit_b16 | dice_edge_sdf_depth | 0 | 0 | 45.94 | +2.80 | 245 | 24.5 |
| unetr_vit_b16 | dice_bands_edge_sdf_depth | 0 | 0 | 47.13 | +3.98 | 245 | 24.5 |
| basic_unet16 | dice | 0 | 0 | 70.01 | +0.00 | 563 | 56.3 |
| basic_unet16 | dice_bands | 0 | 0 | 75.95 | +5.94 | 563 | 56.3 |
| basic_unet16 | dice_sdf | 0 | 0 | 69.41 | -0.60 | 563 | 56.3 |
| basic_unet16 | dice_bands_sdf | 0 | 0 | 76.13 | +6.12 | 562 | 56.2 |
| basic_unet16 | dice_bands_edge | 0 | 0 | 77.18 | +7.17 | 562 | 56.2 |
| basic_unet16 | dice_bands_edge_sdf | 0 | 0 | 77.30 | +7.29 | 562 | 56.2 |
| basic_unet16 | dice_depth | 0 | 0 | 72.53 | +2.52 | 563 | 56.3 |
| basic_unet16 | dice_bands_depth | 0 | 0 | 76.87 | +6.86 | 562 | 56.2 |
| basic_unet16 | dice_edge_depth | 0 | 0 | 73.06 | +3.05 | 562 | 56.2 |
| basic_unet16 | dice_sdf_depth | 0 | 0 | 72.78 | +2.77 | 563 | 56.3 |
| basic_unet16 | dice_bands_edge_depth | 0 | 0 | 78.26 | +8.25 | 562 | 56.2 |
| basic_unet16 | dice_bands_sdf_depth | 0 | 0 | 76.89 | +6.89 | 562 | 56.2 |
| basic_unet16 | dice_edge_sdf_depth | 0 | 0 | 73.45 | +3.44 | 562 | 56.2 |
| basic_unet16 | dice_bands_edge_sdf_depth | 0 | 0 | 77.90 | +7.89 | 562 | 56.2 |
| nnunet | dice | 0 | 1 | 68.59 | +0.00 | 90 | 9.0 |
| nnunet | dice_bands | 0 | 1 | 69.07 | +0.48 | 89 | 8.9 |
