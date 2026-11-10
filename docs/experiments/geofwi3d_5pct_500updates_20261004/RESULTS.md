# GeoFWI3D 5% 500-update pilot results

Job **683068** completed successfully in **3 minutes 39 seconds**, with Slurm exit code `0:0` and empty stderr. The trainer invocation took 119.26 seconds; training and checkpoint-selection validation accounted for 109.98 seconds.

| Training volumes | Validation volumes | Best validation Dice | Selected update | Stopped update |
|---:|---:|---:|---:|---:|
| 19 | 96 | **0.7374 (73.74%)** | 500 | 500 |

The success metric is unweighted mean foreground hard Dice over complete native 96³ validation volumes. The result is selected validation performance; no test set was evaluated. All five validation scores reproduced the first 500 updates of the original seed-0 run exactly. The model trained from scratch.

| Update | Mean validation Dice |
|---:|---:|
| 100 | 0.4997 |
| 200 | 0.6111 |
| 300 | 0.6482 |
| 400 | 0.7085 |
| 500 | 0.7374 |

[Per-volume Dice CSV](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_500updates_20261004/results/5pct_baseline/validation_per_image.csv)

All artifacts listed in the completion record, including best/latest checkpoints, passed remote SHA-256 verification. Retrieved small artifacts passed local SHA-256 verification. The Dice CSV contains all 96 shared validation IDs and reproduces the reported mean. Checkpoints remain in the new campaign on the cluster.

This is an exploratory, validation-informed budget choice: 500 updates were selected after observing the earlier baseline learning curve. The original longer-budget 5% result remains **0.8327**, with its best checkpoint at update 2400. The 73.74% score is specific to the shorter training budget and must be labelled that way. A later constraint comparison should use the same 500-update cap and original learning-rate horizon.
