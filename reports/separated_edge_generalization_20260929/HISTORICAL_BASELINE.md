# Supplementary historical Dice-only baseline

This context uses only the existing hash-verified coherence audit; it is not part of the requested two-arm tables. Same fold-0 case IDs (10 train, 52 validation), data/split hashes, runtime, no augmentation, 75-epoch cap, optimizer, learning-rate schedule and early-stopping settings were verified. Original configuration hashes match saved checkpoint bindings. Model construction and preprocessing functions are AST-identical; validation hard-Dice calculation is unchanged (the only evaluator difference is an inactive added constraint flag). Band/face geometry counts also match case by case. Baseline selected epochs are **73, 74, 71** for seeds 0, 1, 2. This remains a historical comparison at independently selected epochs, not an exact contemporaneous baseline rerun or a matched temporal comparison.

Values are percentages and gaps are train minus validation in pp.

| Metric | Seed | Train | Validation | Gap |
| --- | --- | --- | --- | --- |
| Inner pair correctness | 0 | 99.985637 | 86.681236 | 13.304402 |
| Inner pair correctness | 1 | 99.923868 | 85.760140 | 14.163728 |
| Inner pair correctness | 2 | 100.000000 | 95.266658 | 4.733342 |
| Inner pair correctness | Mean | 99.969835 | 89.236011 | 10.733824 |
| Inner equality satisfaction | 0 | 99.985637 | 93.746614 | 6.239023 |
| Inner equality satisfaction | 1 | 99.923868 | 92.705546 | 7.218321 |
| Inner equality satisfaction | 2 | 100.000000 | 97.461852 | 2.538148 |
| Inner equality satisfaction | Mean | 99.969835 | 94.638004 | 5.331831 |
| Outer pair correctness | 0 | 38.342057 | 50.452531 | -12.110474 |
| Outer pair correctness | 1 | 55.196748 | 59.553309 | -4.356561 |
| Outer pair correctness | 2 | 12.555419 | 30.494454 | -17.939035 |
| Outer pair correctness | Mean | 35.364742 | 46.833431 | -11.468690 |
| Outer equality satisfaction | 0 | 70.762498 | 82.631570 | -11.869072 |
| Outer equality satisfaction | 1 | 73.426517 | 83.362568 | -9.936051 |
| Outer equality satisfaction | 2 | 78.290178 | 83.086509 | -4.796331 |
| Outer equality satisfaction | Mean | 74.159731 | 83.026882 | -8.867151 |
| Crossing correct_transition | 0 | 16.189370 | 18.010074 | -1.820704 |
| Crossing correct_transition | 1 | 31.022460 | 24.605736 | 6.416723 |
| Crossing correct_transition | 2 | 2.232159 | 11.847124 | -9.614965 |
| Crossing correct_transition | Mean | 16.481330 | 18.154312 | -1.672982 |
| Macro Dice | 0 | 80.550744 | 74.136267 | 6.414477 |
| Macro Dice | 1 | 85.813609 | 77.285962 | 8.527647 |
| Macro Dice | 2 | 70.394980 | 68.016059 | 2.378922 |
| Macro Dice | Mean | 78.919778 | 73.146096 | 5.773682 |
The baseline already has an inner-correctness gap of **10.733824 pp** and an inner-equality gap of **5.331831 pp**, alongside a macro-Dice gap of **5.773682 pp**. This directly argues against interpreting the entire inner gap as unique to explicit constraints. The constrained arms enlarge the inner gap while improving validation Dice and boundary accuracy; the observations are consistent with a changed false-positive/false-negative balance and localized generalization difficulty. They do not isolate memorization caused by the constraint formulation. Baseline gaps vary considerably across seeds, especially seed 2. No new baseline inference or training was performed.
