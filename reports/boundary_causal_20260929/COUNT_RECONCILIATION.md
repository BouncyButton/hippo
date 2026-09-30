# Reconciliation of the 4k and 33–40k counts

The original archived 52-case CSV reconstructs **35,909 foreground/background voxel errors** (19,512 FP + 16,397 FN) and **3,872 anterior/posterior swaps**, giving **39,781 distinct wrongly labeled voxels**. The categories are mutually exclusive and sum correctly in every case. Source checkpoint: `msd_fold0_none_20260806_110334_616958/checkpoint_best.pt`. Original inference was CUDA; this analysis only sums saved case metrics, without recomputing predictions in a different runtime.

| Measurement | Cohort total | Mean/case | Median/case | Cases with errors |
|---|---:|---:|---:|---:|
| Foreground FP | 19,512 | 375.23 | 348 | 52/52 |
| Foreground FN | 16,397 | 315.33 | 299.5 | 52/52 |
| Union FP+FN | 35,909 | 690.56 | 646 | 52/52 |
| A/P swaps | 3,872 | 74.46 | 67 | 49/52 |
| All wrong labels | 39,781 | 765.02 | 715.5 | 52/52 |

Boundary mistakes are broadly distributed: the five highest-error cases contribute15.66% of union errors. A/P swaps are more concentrated: the five highest contribute35.18%. Thus “fairly distributed” is more accurate for outer errors than for swaps, although both affect most cases. There are174,350 GT foreground voxels total (3,352.88/case). The FP+FN count has both inside and outside support and should not be called a foreground miss rate.

The historical equivariance model has33,788 FP+FN and3,531 swaps. Later augmented models have about31–32k shell FP+FN. These are different checkpoints and definitions; they do not form one invariant error count.

## Definitions that must stay separate
- Unique union errors: one count per voxel, `(pred>0)!=(GT>0)`; FP and FN disjoint.
- A/P swaps: GT and prediction both foreground, but different A/P class. No foreground miss is a swap.
- Two-step six-neighbor shell: restrict union errors to inner/outer morphological bands. For the later augmented seed0 Dice audit,31,321 shell errors versus31,474 all-grid errors;153 are outside the shells.
- Historical augmentation “boundary”: endpoints of any GT three-class label transition; includes A/P interface. Its33,263 errors for augmented seed2 must not be called outer-shell FP+FN.
- Edge crossing: an exact six-neighbor **foreground/background** GT face. One wrong voxel can invalidate several faces. It is never the anterior/posterior interface.
- Crossing correctness requires the correct transition at the exact GT face and orientation. A synthetic20-cubed mask dilated by one six-neighbor layer has86.96% union Dice,100% surface Dice at one voxel, but0% exact crossing correctness. This illustrates metric strictness, not a result on MRI cases.
- Same-side equality can be satisfied by two wrong labels; correctness must be reported alongside it.
- Bands fuzzy truth is confidence derived from BCE; not voxel or pair accuracy.

Raw sums and input SHA256s are in COUNT_RECONCILIATION.json; the synthetic illustration is in METRIC_STRICTNESS.json.
