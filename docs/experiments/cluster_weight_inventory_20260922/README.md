# Cluster model-weight inventory — 2026-09-22

Read-only snapshot of `/mnt/beegfsstudents/home/3160552` at 13:29 CEST, updated after the requested July-checkpoint deletion verified at 13:42 CEST. The complete inventory is [weights.csv](weights.csv): one row per existing weight file, with its exact cluster path, file size, timestamp, role and available run metadata.

- **327 checkpoint or weight files**, totaling 29.42 GB.
- **81 `model.pt` exports** from SwinUNETR runs; many are calibration sources or the same training run represented by a best and exported copy.
- **78 explicitly named best checkpoints** (`checkpoint_best.pt`, `model_best.pt`, or nnU-Net `checkpoint_best.pth`).
- **42 named latest/final checkpoints**. Intermediate epoch snapshots and auxiliary teacher/head weights are also listed.
- **91 small auxiliary teacher or classifier-head weights** under `hippopotamus_runs`, plus 5 full SwinUNETR weights there.
- **Two nnU-Net weights** (`checkpoint_best.pth` and `checkpoint_final.pth`).
- No training checkpoint for the new sagittal staircase constraint was present in this scan.

These counts are **files, not independent trained models**. In the newer SwinUNETR runner, an early-stopped run's `MSD_fold*/model.pt` is an export of the selected best epoch; `checkpoint_latest.pt` holds the stopping epoch and optimizer state. When early stopping is disabled, `model.pt` is the final epoch. The July 2026 baseline `model.pt` was a 50-epoch final model and was deleted at the user's request on 2026-09-22. The CSV records the policy and epoch when `final_metrics.json` is available. For the 2026-09-21 `hippopotamus_runs/swin_early_stopping_665420` experiment, `model_best.pt` explicitly names the selected weights.

The 2026-09-19 quota cleanup [report](../../../experiments/quota_cleanup_20260919/CLEANUP_REPORT.md) records removal of 52 older `checkpoint_latest.pt` files. This inventory reports **what exists now**, not every weight that ever existed. It skips environments, caches, raw dataset folders, and one Python installation `.pth` file that is not a model.

## Separate early-stopping Swin experiment

`hippopotamus_runs/swin_early_stopping_665420/models` has fold-0 and fold-1 `model_best.pt` (selected epochs 21 and 22) plus `model.pt` at the stopping epochs (29 and 30). The cluster training summary is recorded in the CSV fields for those four files. This is separate from the September 16 early-stopped baseline used in the main comparison table.

## Files by cluster experiment root

| Experiment root | Weight files |
| --- | ---: |
| `ap_interface_weighted_full_20260920_02` | 10 |
| `augmentation_replication_20260916_01` | 2 |
| `bands_augmented_20260916_04` | 4 |
| `baseline_replication_20260916_01` | 2 |
| `hippo` | 11 |
| `hippopotamus_runs` | 96 |
| `low_data75_seed1_pair_20260917_01` | 4 |
| `low_data75_seed2_pair_20260917_01` | 4 |
| `low_data_aug_5pct_200ep_fold0_20260919_01` | 24 |
| `low_data_aug_bands_5pct_200ep_fold0_20260919_01` | 21 |
| `low_data_bands75_seed0_20260917_01` | 2 |
| `low_data_bands_seed0_20260917_01` | 2 |
| `low_data_baseline75_seed0_20260917_01` | 2 |
| `low_data_noaug_12p5pct_fold0_seed0_20260918_01` | 5 |
| `low_data_noaug_25pct_fold0_seed0_20260918_01` | 5 |
| `low_data_noaug_5pct_200ep_es_fold0_seed0_20260918_01` | 5 |
| `low_data_noaug_5pct_200ep_es_fold0_seeds12_20260918_01` | 10 |
| `low_data_noaug_5pct_400ep_es_fold0_seed2_20260918_01` | 5 |
| `low_data_noaug_fold1_20260917_01` | 15 |
| `low_data_noaug_fold2_20260917_01` | 15 |
| `low_data_noaug_fold3_20260917_01` | 15 |
| `low_data_noaug_seed0_20260917_01` | 7 |
| `low_data_noaug_seed1_20260917_01` | 7 |
| `low_data_noaug_seed2_20260917_01` | 7 |
| `low_data_seed1_pair_20260917_01` | 2 |
| `low_data_seed2_pair_20260917_01` | 3 |
| `matched_control_20260902` | 2 |
| `nnunet_recovery_20260920_01` | 2 |
| `regularization_fast_20260916_01` | 2 |
| `regularization_round1_20260916_01` | 2 |
| `seed1_replicate_20260903` | 4 |
| `seed2_four_arm_600ep_20260919_01` | 28 |
| `translation_followup_matched_20260906_02` | 2 |

## Full-data fold-0 comparison checkpoints

| Experiment | Selected epoch | Stop epoch | Validation hard Dice | Cluster checkpoint |
| --- | ---: | ---: | ---: | --- |
| Unaugmented Dice, seed 0, early stopped | 16 | 25 | 0.878599 | `regularization_fast_20260916_01/runs/baseline_seed0/checkpoint_best.pt` |
| Unaugmented Dice, seed 1, early stopped | 21 | 29 | 0.877637 | `baseline_replication_20260916_01/runs/baseline_seed1/checkpoint_best.pt` |
| Mild augmentation, seed 0, early stopped | 24 | 32 | 0.887663 | `regularization_round1_20260916_01/runs/augmentation_seed0/checkpoint_best.pt` |
| Mild augmentation, seed 1, early stopped | 21 | 29 | 0.887135 | `augmentation_replication_20260916_01/runs/augmentation_seed1/checkpoint_best.pt` |
| Mild augmentation + bands, seed 0, early stopped | 21 | 29 | 0.887693 | `bands_augmented_20260916_04/runs/bands_augmented_seed0/checkpoint_best.pt` |
| Augmented Dice control, seed 0, early stopped | 34 | 50 | 0.887977 | `ap_interface_weighted_full_20260920_02/runs/augmented_dice/checkpoint_best.pt` |
| Augmented weighted A/P CE, seed 0, early stopped | 44 | 56 | 0.888804 | `ap_interface_weighted_full_20260920_02/runs/augmented_weighted_ap_ce/checkpoint_best.pt` |
| Augmented uniform A/P CE, seed 0, early stopped | 52 | 56 | 0.889641 | `ap_interface_weighted_full_20260920_02/runs/augmented_uniform_ap_ce/checkpoint_best.pt` |

Validation Dice values describe different experiment protocols and should not be ranked across families without checking their matched controls.
