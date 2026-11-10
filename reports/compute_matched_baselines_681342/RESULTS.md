# Completed compute-matched baseline results

Job **681342** completed successfully after **2 hours 28 minutes 30 seconds**, finishing at **21:52:25 on 3 October 2026 (Europe/Rome)**. All 27 runs completed, with no discarded training attempts. The cluster aggregator verified source/data bindings, checkpoint/metric hashes, selection histories and per-run compute accounting.

All scores below are selected validation macro hard Dice, averaging anterior and posterior Dice over 52 cases per fold. There are nine runs per model: three folds × three seeds. SD is descriptive sample SD across runs, not a confidence interval.

| Model | Mean Dice ± SD | Updates/run | Patient-pass epochs | Estimated training GFLOPs/run |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 0.8170 ± 0.0040 | 12,134 | 1213.4 | 116,255.591 |
| SwinUNETR | 0.6904 ± 0.0566 | 749 | 74.9 | 116,104.090 |
| nnU-Net-derived (matched-input) | 0.6938 ± 0.2174 | 327 | 32.7 | 116,181.104 |

## Means within each fold

| Model | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: |
| U-Net | 0.8206 | 0.8166 | 0.8138 |
| SwinUNETR | 0.6986 | 0.6785 | 0.6942 |
| nnU-Net-derived (matched-input) | 0.6905 | 0.6966 | 0.6944 |

## Means for each seed across the three folds

| Model | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| U-Net | 0.8201 | 0.8142 | 0.8168 |
| SwinUNETR | 0.7048 | 0.7466 | 0.6200 |
| nnU-Net-derived (matched-input) | 0.8414 | 0.8358 | 0.4043 |

U-Net achieved the highest aggregate mean and smallest variation in this protocol. nnU-Net seed 2 had anterior hard Dice of **0.0000 in all three folds**, with macro Dice 0.3862, 0.4153 and 0.4113. Its six seed-0/seed-1 runs scored 0.8304–0.8436. This seed pattern is preserved in the primary nine-run aggregate; no run was excluded. It indicates a reproducibility issue within this budget and recipe, but its cause has not been established.

These are selected validation results, not independent held-out test scores. The folds overlap in training data. The recipes differ in loss and optimizer, and nnU-Net uses an explicitly saved matched-input topology rather than its unrestricted automatic pipeline. The results therefore describe the specified compute/data-limited baseline recipes; they do not isolate architecture alone or establish a constraint benefit.

Full per-run and per-class scores: [summary.csv](summary.csv). Raw verified aggregate: [summary.json](summary.json).
