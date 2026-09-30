# PCGrad: completed anatomical-generalization assessment

**The tested symmetric PCGrad configuration follows a different trajectory, but within this budget it does not generalize hippocampal anatomy better.** It improves inner-pair correctness and recall while substantially overpredicting foreground, losing correct boundary transitions, and producing more disconnected components. This pattern occurs in all three seeds.

The six runs completed successfully (job 676122; 50 minutes 30 seconds). No optimizer updates were skipped. The comparison uses the same five weighted task gradients, summed normally in the control or projected with symmetric PCGrad. Data, initialization, original separated-rule coefficients, schedule, and stopping policy are matched. The fresh sum control is the primary comparator; the earlier separated experiment is a secondary reference.

## Selected checkpoints

Hard macro Dice averages anterior/posterior Dice within each case, then cases. These are post-training audits of selected checkpoints; they are not the logged training Dice loss or necessarily the final training epoch.

| Seed | Arm | Selected epoch | Stopped epoch | Train Dice % | Validation Dice % | Gap (pp) |
|---|---|---:|---:|---:|---:|---:|
| 0 | sum | 64 | 67 | 85.3798 | 77.4542 | +7.9255 |
| 0 | pcgrad | 61 | 69 | 56.8363 | 52.6761 | +4.1602 |
| 1 | sum | 63 | 71 | 89.0764 | 79.3703 | +9.7061 |
| 1 | pcgrad | 58 | 66 | 63.3519 | 66.3309 | -2.9790 |
| 2 | sum | 73 | 75 | 83.0186 | 76.2885 | +6.7301 |
| 2 | pcgrad | 62 | 70 | 46.7751 | 42.2383 | +4.5367 |
| Mean | sum | — | — | 85.8249 | 77.7043 | +8.1206 |
| Mean | pcgrad | — | — | 55.6544 | 53.7484 | +1.9060 |

## Validation anatomy and boundary measurements

Case means within seed, then the three seeds. Percentages are used for correctness/disagreement/Dice; counts are mean counts per case. Inner/outer pair correctness requires both endpoints to be correct. Crossing correctness is the correctly oriented ground-truth transition. Agreement alone can be coherently wrong.

| Metric | Sum control | PCGrad | PCGrad minus control | Seeds improved |
|---|---:|---:|---:|---:|
| Macro Dice % | 77.7043 | 53.7484 | -23.9559 | 0/3 |
| Foreground-union Dice % | 80.3444 | 50.1333 | -30.2111 | 0/3 |
| False-positive voxels | 1196.7436 | 7428.1026 | +6231.3590 | 0/3 |
| False-negative voxels | 295.1538 | 121.0769 | -174.0769 | 3/3 |
| Inner pair correctness % | 85.8432 | 94.7246 | +8.8814 | 3/3 |
| Outer pair correctness % | 59.6642 | 15.0768 | -44.5874 | 0/3 |
| Correct boundary transitions % | 24.7193 | 4.5312 | -20.1881 | 0/3 |
| Inner pair disagreement % | 7.1716 | 1.8368 | -5.3348 | 3/3 |
| Outer pair disagreement % | 17.4215 | 6.3829 | -11.0386 | 3/3 |
| Connected components (6-neighbor) | 3.8846 | 37.8526 | +33.9679 | 0/3 |
| Foreground voxels outside largest component | 8.0641 | 170.3846 | +162.3205 | 0/3 |

At the common epoch-60 endpoint, mean validation Dice is 77.2336% control versus 51.8730% PCGrad. The decline is therefore not just a consequence of different selected epochs.

## Does this support better anatomical learning or a need for more updates?

The higher inner correctness and fewer false negatives are real validation improvements. They occur alongside a much larger foreground region and sharply worse outer/boundary correctness; these observations are consistent with increased foreground coverage rather than improved anatomical specificity. Lower neighboring disagreement does not imply better global connectivity: disconnected-component and outside-largest-component counts worsen in every seed. These measurements do not establish an internal anatomical representation.

PCGrad also performs poorly on the training cases. Its smaller macro-Dice train–validation gap is therefore not evidence of better generalization. The ordinary sum controls fit and generalize substantially better with the same data/update budget. Limited data or insufficient updates may interact with PCGrad, but this experiment cannot isolate those explanations.

All PCGrad runs used patience-based stopping under the 75-epoch cap, selecting epochs 61/58/62 and stopping at 69/66/70. The final-ten-epoch validation Dice slopes are descriptive only:

| Seed | Final window | Validation Dice slope (pp/epoch) |
|---|---|---:|
| 0 | 60–69 | -0.2931 |
| 1 | 57–66 | -0.7123 |
| 2 | 61–70 | -0.0214 |

Training soft Dice continues improving over these windows, but it is the minibatch soft score including background, measured during updates; it is not the audited hard macro Dice. Validation hard Dice declines over each final window. This provides no observed late validation recovery, while leaving longer-budget behavior unknown. No longer run or new experiment has been launched.

The ideal foreground probabilities p=y jointly satisfy all three edge rules. Gradient competition here is an optimization issue, not a logical inconsistency of the desired constraints. Symmetric PCGrad can redirect the supervised gradient even when the conflicting constraint has a small positive scalar weight; preserving the pre-projection loss coefficients does not bound the effect of projection.

These conclusions apply to this symmetric five-task PCGrad implementation, reused validation fold and limited budget; they do not establish that every PCGrad variant or longer schedule would fail. Case/seed repetitions are not independent patient replications. Full original result summaries, per-update projection records, per-case audits and checkpoint hashes are retained.

![Validation learning curves](/Users/filippofocaccia/Desktop/hippo/reports/pcgrad_edge_20260929/validation_learning_curves.png)
