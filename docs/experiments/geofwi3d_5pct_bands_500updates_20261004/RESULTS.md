# GeoFWI3D 5% bands: paired pilot results

Job **683092** completed successfully in **3 minutes 45 seconds**, with Slurm exit code `0:0` and empty stderr.

| Method | Best mean validation Dice | Best update | Stopped update | Training and selection validation |
|---|---:|---:|---:|---:|
| Baseline (job 683068) | 0.7374 (73.74%) | 500 | 500 | 110.0 s |
| Bands (job 683092) | 0.7684 (76.84%) | 500 | 500 | 111.7 s |

The bands arm improved selected validation mean Dice by **3.11 percentage points**. Dice increased on 80 of the 96 validation volumes and decreased on 16. Both arms used the same 19 training cases, the same 96 complete native 96³ validation volumes, seed 0, model initialization policy, sample order, augmentation, optimizer and five validation opportunities within 500 updates. Frozen source is identical, and the protocol differs only in the selected loss method.

This is a full-region salt segmentation Dice comparison, not a boundary-only metric. Bands add the predefined two-voxel inner/outer BCE objective at weight 0.1 with a 500-update ramp. The auxiliary loss is computed from selected training labels only.

[Bands per-volume Dice](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_bands_500updates_20261004/results/5pct_bands/validation_per_image.csv)

[Baseline per-volume Dice](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_500updates_20261004/results/5pct_baseline/validation_per_image.csv)

[Paired per-volume Dice changes](/Users/filippofocaccia/Desktop/hippo/reports/geofwi3d_5pct_bands_500updates_20261004/results/paired_validation_dice.csv)

Every completion artifact, including best/latest checkpoints, passed remote SHA-256 verification. Retrieved result files passed local SHA-256 verification, and all 96 validation IDs match the control. The CSV mean reproduces the completion record. Checkpoints remain on the cluster.

This is a single-seed exploratory selected-validation result. The 500-update budget was chosen using the earlier baseline learning curve; no test set was evaluated and the longer-budget 5% baseline remains 0.8327. Band parameters were fixed before this arm was trained. The update budgets are equal, but bands incur additional computation; the table reports observed single-run timing rather than claiming identical GPU cost.
