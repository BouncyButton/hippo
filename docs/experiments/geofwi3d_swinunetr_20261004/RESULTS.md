# GeoFWI3D SwinUNETR baseline results

Slurm job **682953** completed with exit code `0:0` in **47 minutes 28 seconds**, from 20:13:00 to 21:00:28 on 2026-10-04 (Europe/Rome). All three runs completed; the campaign has no failure record.

Scores are selected validation mean foreground hard Dice, computed per complete native 96³ volume and then averaged over the same 96 salt-containing validation volumes. No test set was evaluated. Model and split seed are 0. These are baseline results; the constraint arm has not been trained.

| Training fraction | Training volumes | Best mean Dice | Per-volume SD | Best update | Stopped update | Stop reason |
|---|---:|---:|---:|---:|---:|---|
| 5% | 19 | 0.8327 | 0.1164 | 2,400 | 3,400 | early_stopping |
| 20% | 77 | 0.9288 | 0.0650 | 3,500 | 4,500 | early_stopping |
| 100% | 383 | 0.9384 | 0.0339 | 4,100 | 5,000 | update_cap |

Best-checkpoint per-volume Dice files:

- [5% Dice CSV](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_swinunetr_20261004/results/5pct_baseline/validation_per_image.csv)
- [20% Dice CSV](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_swinunetr_20261004/results/20pct_baseline/validation_per_image.csv)
- [100% Dice CSV](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_swinunetr_20261004/results/100pct_baseline/validation_per_image.csv)

Slurm accounting reported `COMPLETED`; stderr was empty. Every artifact listed in each completion record, including best/latest checkpoints, passed a remote SHA-256 check. Retrieved result files passed local SHA-256 checks. All three CSVs contain exactly the same 96 validation IDs and reproduce their reported means. Checkpoints remain on the cluster; the small result records, training histories and Dice CSVs have been retrieved locally.

The 5% baseline is 10.57 percentage points below the 100% baseline in mean Dice. This provides a measured low-data baseline for a later paired constraint comparison; it does not establish a constraint benefit.
