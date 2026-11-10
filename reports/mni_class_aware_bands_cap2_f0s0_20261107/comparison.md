# MNI class-aware outer-band pilot

Fold 0, seed 0; same 2× optimization cap and baseline recipes. Final-budget scores, percentages.

| Model | Baseline val | Original bands val | Class-aware val | Δ vs baseline | Δ vs bands | Class-aware train |
|---|---:|---:|---:|---:|---:|---:|
| swinunetr | 76.07 | 76.53 | 76.20 | +0.13 | -0.33 | 82.50 |
| unetr_vit_b16 | 69.41 | 69.48 | 70.66 | +1.25 | +1.18 | 78.68 |

## Error counts and subfields

| Model / condition | FN | FP | Internal swaps | CA1–3 Dice | Subiculum Dice | CA4–DG Dice | Union Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| swinunetr / dice | 6324 | 7487 | 4532 | 77.17 | 74.53 | 76.53 | 85.74 |
| swinunetr / dice_bands | 6697 | 6510 | 4440 | 77.58 | 75.39 | 76.62 | 86.19 |
| swinunetr / class_aware_bands | 6544 | 6836 | 4504 | 77.45 | 75.25 | 75.92 | 86.08 |
| unetr_vit_b16 / dice | 11423 | 7527 | 5115 | 66.80 | 69.74 | 71.69 | 79.29 |
| unetr_vit_b16 / dice_bands | 11028 | 7031 | 5238 | 68.23 | 69.40 | 70.82 | 80.29 |
| unetr_vit_b16 / class_aware_bands | 10276 | 7356 | 4974 | 69.57 | 71.13 | 71.29 | 80.97 |

Categorical targets on existing outer-contour bands only; no added internal-interface support or class reweighting. Two fresh runs and four verified controls. Checkpoints retired only after verified durable archival.

![Training and validation curves](learning_curves.png)
