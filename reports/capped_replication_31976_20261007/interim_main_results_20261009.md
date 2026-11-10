Interim final validation Dice: replication job 691005

Snapshot: 166/448 new conditions completed. Equal optimization ceiling 31,976.45066709 GFLOPs; calibration reported separately.

Means over three new matched fold/seed combinations: (0,1), (0,2), (1,0). Fold 0/seed 0 selection pilot excluded. Best objective selected provisionally from completed comparisons; two U-Net objectives are not yet available for all three combinations.

| Model | Baseline Dice % | Best completed objective | Mean final Dice % | Paired gain pp | Wins |
|---|---:|---|---:|---:|---:|
| nnU-Net | 61.85 | Dice + bands + SDF + depth CE | 62.94 | +1.09 | 3/3 |
| SwinUNETR | 51.67 | Dice + bands + original edge + SDF + depth CE | 64.16 | +12.49 | 3/3 |
| UNETR | 39.35 | Dice + bands + original edge + SDF + depth CE | 44.47 | +5.12 | 2/3 |
| 3D U-Net width 16 | 67.19 | Dice + bands + original edge + depth CE | 76.64 | +9.45 | 3/3 |

Bands-only mean paired gains: nnU-Net +0.85 pp (3/3), SwinUNETR +3.22 pp (3/3), UNETR +2.29 pp (2/3), 3D U-Net +7.85 pp (3/3).

Strong baseline variability: nnU-Net fold 0/seed 2=41.34%; UNETR fold 0/seed 1=17.22%. These are preliminary observations, not significance claims.

Complete per-objective aggregates and individual scores: interim_main_results_20261009.json. Source snapshot: interim_summary_20261009.json.
