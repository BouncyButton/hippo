# GeoFWI3D: final three-fold / three-seed results

Job **694083** completed successfully, exit 0:0, on 11 October 2026 at **13:07:45 Europe/Rome**, starting at 11:25:18 and taking **1 hour 42 minutes 27 seconds**. All 48 new runs completed; together with six verified fold-0/seed-0 pilot references, this gives 54 matched cells.

All objectives include foreground Dice + cross entropy. Every cell uses 19 labelled training volumes, 96 complete native 96³ validation volumes and a 500-update cap. Model seeds are 0, 1 and 2 on each of folds 0, 1 and 2. The original fold is unchanged; validation IDs are disjoint across the three folds.

| Added objective | Mean validation Dice ± sample SD | Paired gain over baseline | Improved cells / 9 |
|---|---:|---:|---:|
| None (baseline) | 71.53% ± 3.25 pp | +0.00 pp | 0/9 |
| Bands | 73.05% ± 3.55 pp | +1.52 pp | 8/9 |
| Signed distance | 72.09% ± 2.79 pp | +0.56 pp | 7/9 |
| Bands + signed distance | 73.43% ± 2.97 pp | +1.90 pp | 9/9 |
| Bands + edge | 73.11% ± 2.96 pp | +1.59 pp | 8/9 |
| Bands + signed distance + edge | 73.54% ± 2.89 pp | +2.01 pp | 9/9 |

The combined bands + signed-distance + edge objective has the highest overall mean. The frozen settings were not retuned for these repetitions.

## Mean validation Dice by fold, across three seeds

| Added objective | Fold 0 | Fold 1 | Fold 2 |
|---|---:|---:|---:|
| None (baseline) | 73.62% ± 2.91 pp | 70.30% ± 3.60 pp | 70.67% ± 3.31 pp |
| Bands | 75.36% ± 2.38 pp | 72.07% ± 1.65 pp | 71.70% ± 5.46 pp |
| Signed distance | 73.69% ± 2.82 pp | 71.08% ± 2.56 pp | 71.50% ± 3.29 pp |
| Bands + signed distance | 75.23% ± 2.98 pp | 72.08% ± 2.46 pp | 72.98% ± 3.54 pp |
| Bands + edge | 74.97% ± 2.52 pp | 72.09% ± 1.57 pp | 72.29% ± 4.30 pp |
| Bands + signed distance + edge | 75.58% ± 2.59 pp | 72.46% ± 1.86 pp | 72.58% ± 3.73 pp |

Scores are selected validation foreground Dice, not test performance. SD describes variation over fold/seed cells, not a confidence interval: training pools overlap across folds. The 500-update budget was informed by the original pilot validation curve. Overall Dice does not by itself establish a boundary-specific metric improvement.

## Verification and retention

Every exported record passed artifact-hash verification, matched its frozen training and validation IDs, covered all 96 cases exactly once, and reproduced Dice from voxel counts. Aggregate means, SDs and paired gains were independently recomputed locally. In total, 5,184 per-volume rows were verified. Full new checkpoint hashes were checked on the compute node after selected-checkpoint Dice reproduction; those temporary checkpoints were then discarded as planned. Original pilot checkpoints remain on the cluster. The final allocation-end quota check recorded 9.09 GiB free, above the 1.25 GiB reserve.

Files:

- [Every fold/seed/objective score](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_replication_20261011/results/per_run_dice.csv)
- [All per-volume Dice and counts](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_replication_20261011/results/all_validation_volumes.csv)
- [Full aggregate statistics](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_replication_20261011/results/summary.json)
- [Final verification](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_replication_20261011/results/final_verification.json)
