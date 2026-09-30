# Cluster storage inventory, 22 September 2026

Scope: `/home/3160552` on BeeGFS. Sizes below are pre-cleanup logical bytes from `du`, rounded to GiB. BeeGFS quota is separately metered; during the initial audit it reported about 90 of 93.13 GiB used. The approved cleanup was completed after the pilot finished; see [`../cluster_cleanup_20260922/RESULTS.md`](../cluster_cleanup_20260922/RESULTS.md) for results preservation, exact removed paths, and the final quota reading. The candidate table below remains a historical inventory.

## Keep for the requested research scope

- All `low_data*` directories, including the 5%, 12.5%, 25%, and 75 epoch variants, all folds and seeds, and their calibration sources.
- `seed2_four_arm_600ep_20260919_01` (3.89 GiB): contains both Dice baselines and full data bands arms, with and without augmentation.
- `bands_augmented_20260916_04` (0.50 GiB), `baseline_replication_20260916_01` (0.26 GiB), `seed1_replicate_20260903` (0.50 GiB), `matched_control_20260902` (0.46 GiB), and `nnunet_recovery_20260920_01` (0.37 GiB).
- `hippopotamus_runs/swin_early_stopping_665420` (0.26 GiB) and both `hippopotamus_runs/nnunet_Dataset101_MSD_fold0_es50_*` runs (0.18 GiB together).
- `hippo` (1.96 GiB), especially `hippo/models` (1.80 GiB) with historical baseline and bands checkpoints, and the active `hippocampus` conda environment (5.92 GiB). `hippo/datasets` is only 0.05 GiB.
- `hippopotamus_runs/step_head_pilot_20260922_665917`: the A–D arm directories, selected checkpoints, and probabilities were deleted on 2026-09-22 at the user's request. Only small run-level records remain.

## Outside that scope: largest archive candidates

| Path under `/home/3160552` | Logical size | What occupies the space | Caution |
|---|---:|---|---|
| `hippopotamus_runs/ap_cut_head_665709` | 2.03 GiB | Fold 0 and 1 training inference, about 1.02 GiB each | Separate A/P cut experiment; archive the entire run if no longer needed. |
| `ap_interface_weighted_full_20260920_02/runs/{augmented_weighted_ap_ce,augmented_uniform_ap_ce}` | 0.86 GiB combined | Two A/P interface training runs | The sibling `augmented_dice` baseline is 0.61 GiB; keep that sibling if retaining all baselines. |
| `ap_swap_mechanism_20260919_01` | 1.02 GiB | 1.01 GiB cache of paired baseline/bands artifacts | Archive only after checking the protected low data source runs still contain the needed originals. |
| `hippopotamus_runs/cst_early_stopped_reanalysis_665422` | 0.52 GiB | Mostly two fold inference folders | CST study, outside requested protected families. |
| `hippopotamus_runs/cst_fold1_replication_665384` | 0.33 GiB | CST replication | Outside requested protected families. |
| `hippopotamus_runs/swin_overfit_audit_665389` | 2.03 GiB | Two training set prediction folders, about 1.02 GiB each | Baseline *audit* data; preserve if those exact saved predictions matter, otherwise archive them while retaining baseline checkpoints and summary. |
| `python_build` and `python3.9` | 0.30 + 0.21 GiB | Python 3.9 source/build and custom installation | Current pilot and baseline scripts use `.conda/envs/hippocampus`; verify no old launcher points here before archiving. |
| `.conda/pkgs` | 0.22 GiB | Conda download/package cache | Cache only; keep `.conda/envs/hippocampus`. |

The A/P cut, two A/P interface treatment arms, two CST runs, Python 3.9 build/install, and conda package cache were removed after results and metadata were saved locally. Their exact pre-removal total was 4.543 GiB of logical files. The AP swap cache and baseline overfit audit add about 3.1 GiB but still need the stated source-checkpoint review before any removal.

Smaller out-of-scope studies: `ap_interface_weighted_full_20260920_01` (<0.001 GiB); `onecut_protocol_20260902` (<0.001 GiB); the remaining `hippopotamus_runs/cst_*` directories (each under 0.01 GiB, except those tabulated above); and `hippopotamus_runs/ap_cut_voxels_665712` (<0.001 GiB).

Other negligible items outside the requested research scope are the two `hippo_*_eval_20260728.py` scripts (39 KB together), `python_build.sh` (2 KB), `transfer.log` (<1 KB), `project` (7 KB), `wandb` (9 KB), `.cache` (66 KB), `.codex-backups` (90 KB), `.nv` (472 KB), and `.config` (776 KB). Removing these would make no practical difference to the quota. Keep `.ssh` and account shell files for access.

## Mixed or ambiguous folders: inspect before archiving

| Path under `/home/3160552` | Logical size | Reason for caution |
|---|---:|---|
| `augmentation_replication_20260916_01` | 0.26 GiB | Contains an augmentation baseline replicate and audit. |
| `regularization_round1_20260916_01` | 0.26 GiB | Contains an augmentation arm, potentially part of baseline comparison. |
| `regularization_fast_20260916_01` | 0.25 GiB | Contains `baseline_seed0`. |
| `translation_followup_matched_20260906_02` | 0.25 GiB | Contains `augmentation_seed0`, potentially a baseline control. |
| `hippo/models/swin_unetr_new_constraints/msd_fold0_none_calibration_pilot_seed0_20260806_220206` | 0.43 GiB | Calibration source may be needed to reconstruct full data bands weights. |
| `.codex` | 0.27 GiB | Codex runtime, unrelated to experiments but useful for the current workflow. |

The full home scan accounted for about 44.6 GiB of logical files. BeeGFS reported much higher quota usage, so use `beegfs quota list-usage --uids current --gids current` to verify the actual quota change after any archive or cleanup. Do not infer available quota directly from `du`.
