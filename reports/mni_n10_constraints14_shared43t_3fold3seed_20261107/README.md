# MNI shared 43T: 14 conditions × 3 folds × 3 seeds

Four models; 10 training subjects (20 crops) and five validation subjects (10 crops) per fold. Fixed historical folds 0–2, model seeds 0–2, both hippocampi together. Training subset is fixed across seeds.

504 logical cells: 496 fresh runs, eight verified fold-0/seed-0 baseline and original-bands archives reused. The historical 14 conditions include the baseline.

Optimization cap: 43,006,868,505,381 FLOPs per cell; calibration and diagnostics accounted separately. Final-budget Dice is primary. No score-dependent stopping or seed replacement. These are development-validation comparisons.

| Model | Peak LR | Gradient target | LR horizon | Baseline updates |
|---|---:|---:|---|---:|
| nnunet | 0.01 | 60% | 200; hold first 100 | 121 |
| swinunetr | 0.0003 | 10% | matched to each arm cap | 277 |
| unetr_vit_b16 | 0.0006 | 5% | matched to each arm cap | 330 |
| basic_unet16 | 0.00016 | 10% | matched to each arm cap | 757 |

Targets apply to sequential calibration of each active term, in bands → edge → SDF → depth order, with a 75% p95 guard. Preserve the original bands prefix/ramp; delay SDF to the update nearest 5/9 of the shared cap.

Conditions:
- dice
- dice_bands
- dice_sdf
- dice_bands_sdf
- dice_bands_edge
- dice_bands_edge_sdf
- dice_depth
- dice_bands_depth
- dice_edge_depth
- dice_sdf_depth
- dice_bands_edge_depth
- dice_bands_sdf_depth
- dice_edge_sdf_depth
- dice_bands_edge_sdf_depth

Two GPU workers. At most one running Slurm allocation and one dependent queued continuation; stop when the finite matrix completes. Each run is archived outside BeeGFS, SHA256 verified, then its checkpoints and bulky generated artifacts are retired. No prediction caches are retained.

Full source, profiles, splits, configurations, curves, error diagnostics and cleanup receipts are preserved under this report. The finite archive collector must retain connectivity to the cluster.
