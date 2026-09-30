# Edge-consistency experiment

Diagnostic job **667752 completed and passed**. On 32 training cases, the
combined logit update reduced inner-FN + outer-FP by 83 (step 0.25) and 256
(step 1.0) relative to the base update. Both error types decreased. This is
not a validation result or a parameter-update experiment.

Training + paired boundary audit job **667767 COMPLETED**, exit 0, in 11m01s
on gnode02 (`gpu:3g.40gb:1`). Selected epoch 26, early stopped at epoch 32.

Initial common validation inference on 52 cases:

| Model | Macro Dice | Boundary FN+FP | Inner FN | Outer FP |
|---|---:|---:|---:|---:|
| Dice | 0.88766014 | 31323 | 15500 | 15823 |
| Ordinary bands | 0.88769427 | 31711 | 13861 | 17850 |
| Bands + edge | 0.88757130 | 31608 | 15100 | 16508 |

No clear validation improvement. Edge-minus-Dice mean case Dice difference
-0.00008885 (paired 95% bootstrap interval -0.00294611 to +0.00278037);
edge-minus-bands -0.00012297 (-0.00246282 to +0.00225696). Boundary-count
difference intervals also cross zero. Tiny inference count differences across
audits are possible with CUDA AMP; compare models within the same audit.

Full train/validation follow-up **667813 completed** in 3m52s, exit 0.
See GENERALIZATION_REPORT.md: edge improves training Dice and boundary errors,
but there is no clear overall validation improvement. This follow-up's common
inference has 31321/31713/31608 validation boundary errors for Dice/bands/edge;
the 1–2 voxel difference from the first audit is consistent with AMP variation.
Remote root: `/mnt/beegfsstudents/home/3160552/edge_consistency_training_20260924_01`.
One new Dice+ordinary-bands+edge run. Existing early-stopped Dice and ordinary
bands are reused. Edge lambda 0.03137547415201203, bands lambda unchanged at
0.01114736174580409. Patience 8, min epochs 25, min delta 0.0005, max epochs 50.

Pre-submission quota: **5.09 GiB free**, above the 2 GiB actual-storage reserve.
No cleanup needed. Ten tests passed; shell syntax and whitespace checks passed.
Frozen training archive SHA256:
`1ebdd67e812068c5b39c960ec3cd74bba5cf2b5dc801bdcf2439d0a3a0bda5e9`.
Bound diagnostic summary SHA256:
`7bfb52043d30592c7e632e4f69eb6158a3eec8afeb5bb1c77e85a257eefcce42`.

See PROTOCOL.md for the predeclared diagnostic gate and training/audit plan.
Validation remains a previously used development fold; a positive outcome
would need confirmation across seeds/folds before a general claim.
