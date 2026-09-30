# Fold 3 completed results

Job **675308** completed successfully (exit 0) in **03:57:04** on 2026-09-29. All nine models finished 200 updates and evaluation on the same 44 held-out volumes. All three per-seed audits passed, with result hashes verified against the audit. Mean Dice was independently recomputed from each seed's 44 per-volume values, agreeing within 1e-12. The model checkpoints and images remain on the cluster.

| Seed | Baseline Dice | Bands Dice | Bands+edge Dice | Bands+edge minus baseline (pp) |
|---|---:|---:|---:|---:|
| 17 | 0.85241 | 0.85580 | 0.85635 | 0.394 |
| 83 | 0.84429 | 0.84966 | 0.85035 | 0.606 |
| 191 | 0.84114 | 0.83531 | 0.85280 | 1.166 |
| Mean | 0.84595 | 0.84692 | 0.85317 | 0.722 |

Bands alone gain 0.098 percentage points on average and improve two of three seeds. Bands+edge improve all three seeds and outperform bands by 0.625 points on average.

## Combined evidence, retaining fold 2

Across all four folds and three seeds, the equally weighted run means are baseline 0.83329, bands 0.83323, and bands+edge 0.84322. Bands+edge versus baseline: +0.992 percentage points, positive in 9/12 runs and 3/4 fold averages. The exploratory hierarchical 95% interval is [-0.272, +2.332] points; the two-sided fold sign-flip p-value is 0.25. These data support a promising trend, not an established general improvement. Bands alone are essentially flat overall (-0.006 points).

## Integrity and interpretation

Fold 3 reused exactly identical training-only calibration reports across all seeds; both coefficient consistency checks pass with zero deviation. All nine models used the same specified budget, and every per-seed audit confirms matched initializations/schedules/learning rates, active constraints and an unchanged frozen backbone.

The combined report remains **reported_with_calibration_deviation**, with the strict study gate false because previously documented coefficient differences across seeds in folds 0–2 exceed the original strict tolerance (maximum relative deviation about 0.1054%). This is not a new failure in fold 3; the historical deviation has not been waived or erased. The saved metadata include all consistency checks and aggregate statistics.

Fold 3 was added after viewing earlier results and must be described as exploratory; fold 2 is retained. The three seeds share each fold's cases and are not independent patient cohorts. Patient identifiers are unavailable, so volume-level separation does not establish patient-independent generalization. No jobs were cancelled and no artifacts deleted. The seed-17 monitor remains paused.

See FINAL_METADATA.json for retrieved, reserialized result metadata, remote result hashes, audits and the aggregate report.

