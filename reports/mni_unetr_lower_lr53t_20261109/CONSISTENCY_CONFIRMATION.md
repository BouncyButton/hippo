# UNETR constraint consistency confirmation

The shared LR 0.0003 + delayed original-bands recipe improved final-budget validation macro hard Dice in all nine fixed seed/fold comparisons. Mean paired gain: **+0.7603 percentage points**; minimum: **+0.0345 points**. Mean baseline/constrained Dice: **65.6602% / 66.4205%**.

The user explicitly selected consistent positive matched gains over the highest absolute Dice. The reduced-LR recipe has lower absolute performance than stronger earlier baselines; this tradeoff was accepted before promotion.

| Seed | Fold | Matched baseline Dice (%) | Bands Dice (%) | Gain (pp) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 69.7541 | 70.1601 | +0.4060 |
| 0 | 1 | 70.5849 | 70.9324 | +0.3475 |
| 0 | 2 | 72.7753 | 72.9716 | +0.1963 |
| 5 | 0 | 68.0717 | 68.2353 | +0.1636 |
| 5 | 1 | 69.4306 | 69.4651 | +0.0345 |
| 5 | 2 | 70.8580 | 71.4035 | +0.5456 |
| 19 | 0 | 55.8180 | 57.6268 | +1.8088 |
| 19 | 1 | 55.8790 | 57.5882 | +1.7092 |
| 19 | 2 | 57.7699 | 59.4015 | +1.6316 |

Recipe: AdamW peak LR 0.0003; 24-update warmup followed by cosine decay to 1% of peak; no augmentation or dropout; original foreground squared Dice + CE (1:1); 409 successful updates. Original bands calibrated using all 20 training crops at update 102, with 5% target gradient ratio, 0.75 p95 cap, and a 33-update ramp. Optimization cap: 53,305,109,592,450 FLOPs. Calibration is charged separately under the established accounting convention.

Verification: the nine pairs share the same constraint protocol and matched-baseline protocol. Fold splits, fixed seeds, training-only calibration, preconstraint training prefixes, final updates, and archive hashes were checked. All 36 fresh runs in the reduced-LR study were archived and verified before cleanup; cumulative released checkpoint storage was 62.159 GiB. All attempted candidates remain in the study results.

Scope: these are observed development comparisons, not an untouched test or a guarantee on new data or individual subjects. Bands is the only combination confirmed positive in all nine pairs. The full combination already had a negative seed-5/fold-0 result; that result is retained in the full matrix.

Automatic continuation: matrix mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109 was submitted as job 748838, with 504 logical cells (4 models × 14 conditions × 3 seeds × 3 folds), 32 exact archived reuses, and 472 new/resumed runs. UNETR base recipe/timing extends to every combination; nnUNet retains LR 0.02, and the other established recipes and cap are unchanged. Runtime startup verification is recorded in the matrix report.
