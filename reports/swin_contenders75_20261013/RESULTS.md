# Final SwinUNETR contender comparison

Job **698711 completed successfully**: all 15 runs. Same 5% training subset, fold 0, seed 0; maximum 75 epochs with the established early stopping. All original-constraint rows include Dice.

The winning constraint reproduced **81.501572%**. It leads the best tested alternative by **4.198 percentage points**. This compares these fixed recipes on the development fold, not every possible tuning.

| Constraint | Best Dice | Best epoch | Epochs trained | FN | FP | A/P swaps |
|---|---:|---:|---:|---:|---:|---:|
| Winning constraint | 81.502% | 41 | 49 | 25,901 | 30,413 | 4,609 |
| Bands + edge | 77.303% | 69 | 75 | 17,727 | 60,565 | 5,178 |
| Bands + edge + depth CE | 77.248% | 58 | 66 | 16,365 | 62,861 | 5,182 |
| Bands + edge + signed distance + depth CE | 77.204% | 58 | 66 | 16,324 | 63,301 | 5,124 |
| Bands + edge + signed distance | 76.962% | 70 | 75 | 17,984 | 61,845 | 5,240 |
| Original bands | 76.727% | 75 | 75 | 16,923 | 64,806 | 5,203 |
| Bands + signed distance + depth CE | 76.628% | 75 | 75 | 16,799 | 65,396 | 5,227 |
| Bands + depth CE | 76.617% | 58 | 66 | 15,390 | 67,678 | 5,179 |
| Bands + signed distance | 76.416% | 66 | 67 | 15,900 | 67,719 | 5,327 |
| Edge + depth CE | 75.720% | 75 | 75 | 16,343 | 70,500 | 5,408 |
| Edge + signed distance + depth CE | 75.528% | 75 | 75 | 16,267 | 71,918 | 5,342 |
| Depth CE | 75.035% | 75 | 75 | 15,132 | 76,248 | 5,446 |
| Signed distance + depth CE | 74.886% | 75 | 75 | 15,523 | 76,237 | 5,486 |
| Dice baseline | 74.304% | 73 | 75 | 15,758 | 79,384 | 5,577 |
| Signed distance | 74.181% | 73 | 75 | 15,853 | 79,656 | 5,674 |

FN means true hippocampus predicted as background; FP means background predicted as hippocampus. A/P swaps mean detected hippocampal tissue assigned to the wrong class. Counts are summed over the 52 validation volumes at each run’s Dice-selected checkpoint.

All recorded artifacts and retirement receipts were verified. No experiment checkpoints, prediction volumes, or temporary reference masks remain. Aggregate results, curves, calibrated configurations and hashes are archived locally in `final_audit.json`; paired shared-tissue swap diagnostics are included.
