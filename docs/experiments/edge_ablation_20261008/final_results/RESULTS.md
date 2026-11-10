# Ordinary SwinUNETR edge ablation: 5%, fold 0, seed 0

Completed: 20/20. Dice+bands is the fresh matched baseline.

| Objective | A/P Dice (%) | Δ Dice (pp) | Extra | Missed | Swaps | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| bands | 76.7272 | +0.0000 | 64806 | 16923 | 5203 | 75 |
| legacy_edge | 77.3031 | +0.5759 | 60565 | 17727 | 5178 | 69 |
| same_only | 76.3016 | -0.4256 | 66762 | 16779 | 5443 | 73 |
| split_equal | 77.3174 | +0.5902 | 61170 | 17197 | 5223 | 68 |
| probability_1mm_pairs_outer | 77.2426 | +0.5154 | 61057 | 17441 | 5281 | 62 |
| probability_1mm_pairs_outer_ap | 76.7717 | +0.0444 | 64693 | 16992 | 5119 | 70 |
| probability_1mm_anchors_outer | 77.3928 | +0.6655 | 59605 | 17692 | 5388 | 73 |
| probability_1mm_anchors_outer_ap | 76.6966 | -0.0306 | 65972 | 16282 | 5191 | 60 |
| probability_1and3mm_pairs_outer | 77.6467 | +0.9194 | 60037 | 16898 | 5186 | 65 |
| probability_1and3mm_pairs_outer_ap | 77.0208 | +0.2936 | 62626 | 17321 | 5201 | 69 |
| probability_1and3mm_anchors_outer | 77.6272 | +0.8999 | 59525 | 17156 | 5255 | 68 |
| probability_1and3mm_anchors_outer_ap | 76.7664 | +0.0392 | 64344 | 16957 | 5265 | 75 |
| score_1mm_pairs_outer | 76.7619 | +0.0347 | 64552 | 16644 | 5376 | 75 |
| score_1mm_pairs_outer_ap | 76.7532 | +0.0260 | 65139 | 16379 | 5341 | 75 |
| score_1mm_anchors_outer | 76.9102 | +0.1830 | 64398 | 16549 | 5193 | 75 |
| score_1mm_anchors_outer_ap | 76.5525 | -0.1747 | 67121 | 15832 | 5320 | 68 |
| score_1and3mm_pairs_outer | 76.7344 | +0.0071 | 63575 | 17532 | 5279 | 75 |
| score_1and3mm_pairs_outer_ap | 76.5043 | -0.2230 | 66902 | 16058 | 5377 | 65 |
| score_1and3mm_anchors_outer | 76.7941 | +0.0669 | 66697 | 15177 | 5327 | 60 |
| score_1and3mm_anchors_outer_ap | 76.5508 | -0.1764 | 65085 | 17031 | 5387 | 69 |
