# GeoFWI3D 5% loss comparison

Job **683761** completed successfully (Slurm exit 0:0), 5 October 2026, 10:50:32–11:00:24 Europe/Rome; elapsed **9 minutes 52 seconds**. Four arms ran sequentially in one sbatch allocation.

All six compared arms use the same 19 training volumes, 96 complete native 96³ validation volumes, seed 0 and 500 optimizer-update cap. Every new arm selected its best checkpoint at update 500 and stopped at the cap. The base loss is foreground Dice + cross entropy.

| Added objective | Best validation Dice | Gain over baseline | Change vs bands alone |
|---|---:|---:|---:|
| None (control job 683068) | 73.74% | — | -3.11 pp |
| Bands (control job 683092) | 76.84% | +3.11 pp | — |
| Signed distance | 75.30% | +1.56 pp | -1.55 pp |
| Bands + signed distance | 76.71% | +2.97 pp | -0.13 pp |
| Bands + edge | 76.05% | +2.31 pp | -0.79 pp |
| Bands + signed distance + edge | 76.50% | +2.76 pp | -0.34 pp |

Bands alone remains highest: 76.84%, versus 76.71% for bands + signed distance. All four new variants improve on the 73.74% baseline. These are exploratory results for one seed under a validation-informed budget; there is no test evaluation or evidence yet of consistent gains across seeds.

Scores are mean foreground hard Dice over the same 96 validation volumes, selected by validation Dice. Matching training/validation IDs, downloaded JSON/CSV/config artifact hashes, summary/completion agreement and means recomputed from per-volume CSVs were verified. Checkpoints remain on the cluster and were not downloaded or independently rehashed in this follow-up.

Per-volume results:

- [Signed distance](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/5pct_sdf/validation_per_image.csv)
- [Bands + signed distance](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/5pct_bands_sdf/validation_per_image.csv)
- [Bands + edge](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/5pct_bands_edge/validation_per_image.csv)
- [Bands + signed distance + edge](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/5pct_bands_sdf_edge/validation_per_image.csv)
- [Paired comparison of all six arms](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/paired_validation_dice.csv)
- [Verification record](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_four_losses_20261005/results/verification.json)
