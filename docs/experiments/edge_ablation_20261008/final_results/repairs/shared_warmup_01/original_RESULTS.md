# Ordinary SwinUNETR edge ablation: 5%, fold 0, seed 0

Completed: 8/20. Dice+bands is the fresh matched baseline.

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
