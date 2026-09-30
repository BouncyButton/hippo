# 5%-data result: bands-only versus bands + edge

Job **667831 completed successfully** in 19m36s (exit 0). All three seeds completed training and both split audits. Exact historical fold-0 subset: ten training cases and 52 validation cases, no augmentation. Existing controls reused.

**Bands + edge improved validation Dice and boundary errors in every seed and reached the historical bands scores in fewer epochs.**

| Seed | Bands validation Dice (%) | Bands + edge (%) | Gain (percentage points) | Edge selected / stop epoch |
|---|---:|---:|---:|---|
| 0 | 76.855 | 77.525 | +0.670 | 73 / 75 |
| 1 | 79.142 | 79.539 | +0.397 | 63 / 71 |
| 2 | 75.246 | 76.261 | +1.015 | 74 / 75 |
| Mean | 77.081 | 77.775 | +0.694 | |

Mean training Dice gain is +1.427 percentage points; validation gain is +0.694. Unlike the completed full-data experiment, improvement transfers to validation here. This is a conditional result for this subset and schedule, not proof of an intrinsic data-scarcity interaction.

## Learning speed

| Seed | Bands best-score epoch | Edge first reaches that score | Edge three-epoch crossing starts / confirmed |
|---|---:|---:|---|
| 0 | 68 | 48 | 58 / 60 |
| 1 | 74 | 49 | 49 / 51 |
| 2 | 74 | 50 | 50 / 52 |

First crossings occur 20, 25 and 24 epochs earlier (200, 250, 240 fewer optimizer updates); aggregate threshold-attainment updates fall from 2160 to 1470, about 31.9%. These are threshold-attainment savings, not total training-time savings. Three-consecutive-epoch confirmation is reached at 60, 51 and 52. All edge curves also have higher mean Dice over their common observed horizons.

Useful fixed-threshold checks: seed 0 first reaches 76% at epoch 42 versus bands 48 (three-epoch confirmation 44 versus 57); seed 1 first reaches 78% at 30 versus 37 (confirmation 37 versus 45); seed 2 reaches 76% at 65, a level bands never reaches in its 75 epochs.

The new GPU is 3g.40gb and historical controls used 4g.40gb. No hardware-controlled wall-clock speedup is established. The new seeds took 263.7, 250.4 and 276.5 seconds for training including I/O, with about four seconds per seed for edge calibration using retained calibration checkpoints; these figures exclude the original cost of training those source checkpoints.

## Boundary errors

Counts below are within the two-voxel bands, excluding AP swaps. Positive FN change means more missed foreground; negative FP change means fewer extra foreground voxels.

| Seed | Bands FN+FP | Edge FN+FP | Net fewer errors | Change in inner FN | Change in outer FP |
|---|---:|---:|---:|---:|---:|
| 0 | 73787 | 70587 | 3200 | +1494 | -4694 |
| 1 | 64141 | 62303 | 1838 | +337 | -2175 |
| 2 | 80925 | 75330 | 5595 | +1727 | -7322 |

Mean boundary-error count falls from 72951 to 69406.7 (4.86%). The improvement primarily reduces false positives, at the cost of more missed foreground. Mean case inner-FN rate rises 0.959 percentage points, outer-FP rate falls 2.700 points, and balanced boundary error improves 0.870 points. Boundary AP swaps change by -103, +54 and +33, so that component is not consistently improved.

Averaging the seed-paired effects within each validation case first, Dice improves in 51/52 cases, balanced boundary error improves in 51/52, and boundary error counts improve in 52/52. Conditional paired-case bootstrap 95% interval for the mean Dice gain is +0.597 to +0.790 percentage points. Per-seed Dice difference intervals are also positive. These intervals condition on this subset and these validation-selected models; they omit seed/subset and checkpoint-selection uncertainty. The repeated seeds do not create 156 independent validation cases.

## Interpretation

This is a promising improvement over bands alone under the requested 5%, 75-epoch training policy: both earlier attainment and higher validation scores, replicated across the three initializations. It does not establish a superior asymptotic solution: learning-speed effects could shrink under longer schedules, as occurred in earlier low-data comparisons. Early stopping remained active; seeds 0/2 reached the cap, seed 1 stopped at 71. Minimum epoch 60 and patience 8 matched the historical controls.

The explicit FN/FP trade-off matters: this is not uniformly better preservation of every boundary property. Validation is the same previously used development fold, not an independent test cohort. Historical/new inference differs slightly under AMP, so accuracy and error tables use common audit inference; crossing thresholds use the saved training histories.

![Learning curves](learning_curves.png)

Aggregate evidence, learning curves, provenance and calibration weights: RESULTS.json. Case-level audit records and checkpoints remain on the cluster.
