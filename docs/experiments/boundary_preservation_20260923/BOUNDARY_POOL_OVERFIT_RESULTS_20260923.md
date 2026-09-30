# Boundary-local pooling: better discrimination, preservation requirement unmet

Job **666849** completed all **24 fits** in **4m55s**, with exit code zero.
Boundary pooling lowered fitting loss and improved correction-score AUROC in
every one of the 12 matched fold/seed comparisons. **None met the predefined
preservation criteria. E remains the retained model.**

## Matched comparison and scope

The [frozen protocol](BOUNDARY_POOL_OVERFIT_PROTOCOL_20260923.md) compared whole-ray
mean pooling with separate Gaussian-weighted pools around the frozen predicted
lower and upper edges (sigma one voxel). Both arms used the same 13 input
features, width 24, 8,185 trainable parameters, initial weights, interval loss,
optimizer and 1,024 case updates. Initial rendered outputs exactly reproduced E.
Only pooling changed. The backbone, geometry and original E checkpoints stayed
frozen. These are renderer-correction heads, not newly trained anatomical
presence classifiers or boundary-position heads.

Each fold used the same four fitting cases as the preceding experiment:
**seven distinct cases in total**, with 16 case/fold exposures. There were three
head-fitting seeds per fold. Repeated cases and seeds are not independent patient
samples. No inner or outer validation cases were evaluated, and no checkpoint
was selected after seeing results. These are deliberately small fitting tests,
not evidence of generalization or clinical usefulness.

The freshly fitted mean control reproduced the earlier renderer-state control
exactly for every recorded checkpoint's loss and mask metrics (maximum absolute
difference zero). Pooling changes parameter Jacobians and optimization
conditioning as well as feature aggregation; the experiment does not isolate
information retention from these other effects.

## Final-step results

Values average cases within each four-case subset and then the 12 fold/seed fits.
AUROC ranks feasible, initially missed true thin rays against hard-empty rays
using learned corrections. It does not measure calibrated presence or edge error.

| Metric | Unmodified E | Mean-pooling correction | Boundary-pooling correction |
|---|---:|---:|---:|
| Interval fitting loss | 0.185186 | 0.118788 | **0.103305** |
| Correction AUROC | 0.5000 (zero residual) | 0.6093 | **0.7907** |
| Thin-ray true-overlap recall | 66.4372% | 70.1443% | **70.3763%** |
| Thin-voxel recall | 64.2963% | 68.4159% | **69.0171%** |
| FP rays/case | **7.0625** | 8.6042 | 8.4375 |
| FP voxels/case | **239.2500** | 247.0417 | 246.5625 |
| Newly deleted true voxels/case, relative to E | **0** | **0** | 0.0417 |
| Union Dice | 0.939250 | 0.939539 | 0.939613 |
| Fits meeting all preservation criteria | — | **0/12** | **0/12** |

Boundary pooling reduced the mean objective by **13.0344%** relative to mean
pooling. Per-pair reductions ranged from **7.2215% to 24.0467%**. All 12 pairs
improved AUROC. However, thin-overlap recall improved over the mean control in
only **7/12** pairs, with an average increase of **0.2321 percentage points**.
The average reduction in FP burden relative to the mean control was just
0.1667 rays and 0.4792 voxels/case.

Relative to E, every final boundary head increased both FP burdens. The mean
increase was **1.3750 FP rays and 7.3125 FP voxels/case**. Two heads (fold 2,
seeds 0 and 1) each newly deleted one true voxel across their four fitting cases,
or 0.25 voxels/case for each head. These are two model/case deletion events;
the aggregate report does not establish two distinct affected voxels or patients.
The small count still violates the predefined preservation condition. The
aggregate does not specify whether those deleted voxels belong to thin rays.

No recorded intermediate checkpoint at 128 or 512 steps met the criteria either.
No alternative checkpoint is promoted. ASSD was not computed in this diagnostic;
the fitting Dice cannot be compared with the earlier 208-case validation Dice.

## Heterogeneity

| Fold | Mean AUROC | Boundary AUROC | Mean thin overlap | Boundary thin overlap | Mean FP voxels | Boundary FP voxels |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.7421 | 0.8578 | 67.2889% | 67.7465% | 253.6667 | 251.9167 |
| 2 | 0.6067 | 0.8677 | 68.2789% | 68.3312% | 236.0000 | 235.9167 |
| 3 | 0.4911 | 0.6858 | 78.0780% | 78.8133% | 281.9167 | 280.5000 |
| 4 | 0.5971 | 0.7514 | 66.9312% | 66.6144% | 216.5833 | 217.9167 |

Fold 4 illustrates the distinction between ranking and decisions: AUROC rose
while thin-overlap recall fell and FP voxels increased relative to the matched
control. This fitting-subset result does not establish the cause of historical
fold-4 validation failures.

## What changed in the loss and gradients

| Weighted loss contribution | Mean pooling | Boundary pooling |
|---|---:|---:|
| Initially missed, feasible thin | 0.078165 | 0.066721 |
| Initially missed, infeasible thin | 0.026694 | 0.022654 |
| Initially overlapping thin | 0.003586 | 0.004104 |
| Other true rays | 0.006461 | 0.006397 |
| Hard-empty rays | 0.003480 | 0.003206 |
| Other empty rays | 0.000403 | 0.000223 |

Most of the objective reduction is on the intended feasible-missed-thin group,
with additional improvement on infeasible thin rays. Loss on already-overlapping
thin rays increased. Thus this is useful task-related fitting, but the objective
does not guarantee preservation of discrete voxel decisions.

Among still-missed feasible thin rays, mean shortfall to the safe rescue lower
bound fell from **1.3938 to 1.2735 residual logits**. This compares the remaining
receiver sets under each fitted head; it is not a paired same-ray gap estimate.
The remaining gaps help explain why stronger ranking need not produce rescue:
the correction must cross each ray's own threshold, while respecting its own
upper protection bound. AUC alone says neither requirement is met.

The mean rescue-versus-hard-empty gradient cosine changed from **−0.999649** to
**−0.999008**. Rescue versus infeasible missed thin remained **−0.999903** in the
boundary arm. This intervention did not remove the near-opposition of these
group-averaged parameter gradients.

For both arms, the normalized total SGD diagnostic raised mean correction on
remaining feasible thin rays in **4/12** heads and lowered it in **8/12**. In
every head, hard-empty corrections moved in the same direction. No total
direction increased mean thin rescue correction without increasing mean
hard-empty correction. These are local first-order group-mean responses, not
AdamW updates or proof that no selective parameter direction exists.

All 24 gradient-response diagnostics passed the numerical reliability checks.
Maximum reconstruction error relative to the sum of group-gradient norms was
6.63e-6; maximum relative error to the total gradient norm was 0.0003353.

## Decision and next discriminating question

**Retain E.** Boundary-local pooling is a supported candidate for improving the
correction representation on these fitting cases, but it has not solved
selective preservation. The stronger hypothesis that this pooling change alone
would meet the predefined preservation criteria was not supported.

The next question is whether the improved scores encode a usable safe-action
region, or whether they merely rank examples better while still requiring
overlapping, incompatible action magnitudes from this learned mapping. Before
another training run, inspect signed distances of corrections to each ray's
label-derived admissible interval, stratified by feasible missed thin,
already-overlapping thin and hard-empty rays. In particular, attribute newly
added FP voxels and true deletions to lower/upper interval violations and compare
the same rays between arms. Any use of label-derived intervals is a fitting-only,
non-deployable diagnostic.

If errors are chiefly a shared calibration offset, a predeclared low-dimensional
calibration control could test that on fitting data with a separate inner
selection rule. If required corrections remain inseparable, calibration cannot
solve the representation/action problem. Neither explanation is established by
the present AUC result. Do not launch a larger training or hyperparameter screen
on the strength of that AUC alone.

## Verification and artifacts

Four implementation tests passed before cluster execution: normalized label-free
boundary pools including border cases; initial equality, parameter counts and
mean-control parity; a synthetic locality test; and two-arm fitting/save/audit
execution with original weights unchanged. Launcher shell syntax passed.
Source, dataset and checkpoint hashes were verified. All final weights stayed
on the cluster. Only non-identifying aggregates were downloaded under the user's
explicit authorization for this investigation.

- [Compact summary, including all 12 pairs](BOUNDARY_POOL_OVERFIT_SUMMARY_666849.json)
- [Full aggregate diagnostics](BOUNDARY_POOL_OVERFIT_AGGREGATES_666849.json)
- Implementation: `experiments/presence_decisive_20260923/boundary_pool_overfit.py`
- Tests: `experiments/presence_decisive_20260923/test_boundary_pool_overfit.py`
- Remote results: `/home/3160552/boundary_pool_overfit_20260923_01a0cd/results_666849`
