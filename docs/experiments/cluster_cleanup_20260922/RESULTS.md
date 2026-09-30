# Results preserved before cluster cleanup — 2026-09-22

The cluster runs below are complete research outputs outside the protected low-data experiments and full-data baseline/bands runs. This note records their principal results before the specified output directories are removed to relieve the BeeGFS quota. Values are from each run's selected checkpoint and its 52 patient-disjoint development cases unless stated otherwise. These folds have been used during method development, so they are not untouched test results.

The original JSON, CSV, and Markdown metadata from all four cleaned run groups is preserved in [`metadata.tar.gz`](metadata.tar.gz) (82 files, SHA-256 `82384d1c2deec8a914857add2acd486390a0f0047e09df62789ebc3c1c62b3c6`). Paths inside the archive are relative to `/home/3160552`. Source and method details are also in [`semantic_constraints/ap_cut_head/EXPERIMENT_20260922.md`](../../../semantic_constraints/ap_cut_head/EXPERIMENT_20260922.md) and [`semantic_constraints/cst_teacher/EXPERIMENT_20260921.md`](../../../semantic_constraints/cst_teacher/EXPERIMENT_20260921.md).

## A/P cut head — job 665709

Frozen early-best Swin checkpoints supplied training and held-out probabilities. Heads predicted the coronal A/P cut from MRI plus Swin probabilities or Swin probabilities alone. An inner 166/42 split selected the head; the outer fold had 52 cases. Neither head changed segmentation labels.

| Fold | Swin cut MAE, slices | MRI + Swin head MAE | Swin-only head MAE | Better / worse cases |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.885 | 0.846 | 0.846 | 2 / 0 |
| 1 | 1.058 | 1.038 | 1.038 | 1 / 0 |

The modest gain does not establish an MRI-derived patient-specific cue: the two heads made identical hard cuts in fold 0 and neither had a net fold-1 advantage. The separate frozen-decoder cut-band probe (job 665712, documented in the linked experiment note) was worse than the Swin cut in both folds: 0.923 versus 0.885 in fold 0, and 1.154 versus 1.058 in fold 1. Neither frozen head supports imposing the proposed cut-ordering constraint.

Cleanup target: `/home/3160552/hippopotamus_runs/ap_cut_head_665709`. The archived metadata includes both fold reports and training-inference indexes and summaries. It does not include the large prediction arrays or checkpoints.

## Full-data A/P interface loss — run `ap_interface_weighted_full_20260920_02`

Three arms were compared on the same 52-case development fold. The Dice arm and its outputs remain on the cluster as the baseline. The two A/P interface treatment arms have their small configs, epoch curves, completion manifests, validation details, and final metrics in the metadata archive.

| Arm | Selected epoch | Hard Dice | Soft Dice | Ground-truth boundary ECE |
| --- | ---: | ---: | ---: | ---: |
| Augmented Dice baseline | 34 | 0.887977 | 0.884955 | 0.175587 |
| Dice + uniform A/P CE | 52 | 0.889641 | 0.887790 | 0.178839 |
| Dice + weighted A/P CE | 44 | 0.888804 | 0.886517 | 0.177172 |

Uniform A/P CE improved hard Dice by 0.001664 over the matched Dice arm; weighted A/P CE improved it by 0.000827. Neither improved boundary calibration ECE. These are small development-fold differences, not evidence of a robust independent gain.

Cleanup targets: `/home/3160552/ap_interface_weighted_full_20260920_02/runs/augmented_uniform_ap_ce` and `/home/3160552/ap_interface_weighted_full_20260920_02/runs/augmented_weighted_ap_ce`. The run root and `runs/augmented_dice` remain.

## CST fold-1 replication — job 665384

The dense 32-slice Smooth-L1 teacher replicated profile accuracy on fold 1 (profile MAE 0.0460 ± 0.0038), but raw profile disagreement did not transfer as a standalone case-error cue (correlation -0.091). The combined CST/uncertainty slice-risk features captured 61.0% of measured error in the top 20% of slices, versus 54.0% for uncertainty alone using the original epoch-50 Swin checkpoint. The candidate use is selective quality control; forcing predictions toward the teacher profile failed the repair gate in earlier work.

Cleanup target: `/home/3160552/hippopotamus_runs/cst_fold1_replication_665384`. The archive contains the profile-ablation summary, teacher evaluation summaries, inference index, and risk reports; larger image/prediction arrays and checkpoints are omitted.

## CST early-best reanalysis — job 665422

Frozen CST teachers were reevaluated against the matched early-best Swin checkpoints, which reduced train–development overfit. The anterior-volume violation remained associated with error across seeds. Raw profile disagreement remained inconsistent between folds. The combined features retained their slice-level error-ranking gain:

| Fold | Swin checkpoint | Slice-risk Pearson r, uncertainty → combined | Top-20% error capture, uncertainty → combined |
| --- | --- | ---: | ---: |
| 0 | Early best, epoch 21 | 0.339 → 0.534 | 42.6% → 57.6% |
| 1 | Early best, epoch 22 | 0.532 → 0.633 | 51.9% → 61.8% |

The main inference is that combined CST features help rank slices for review even after checkpoint selection. These results do not establish a safe segmentation correction direction. The linked CST experiment note contains the full epoch-50 comparison and cautions about prior use of both folds.

Cleanup target: `/home/3160552/hippopotamus_runs/cst_early_stopped_reanalysis_665422`. The archive includes the checkpoint comparison, fold risk and teacher evaluations, and inference indexes; it omits large image/prediction arrays and checkpoints.

## Cleanup record

The user approved removal of these A/P cut, A/P interface treatment, and CST outputs after saving their main results here. All five listed experiment paths were removed. Their pre-removal `du -sb` total was **4,011,826,493 logical bytes (3.736 GiB)**.

The separately identified unused `python_build`, `python3.9`, and `.conda/pkgs` directories were also removed after checking that no research launcher referenced the old Python 3.9 installation. They totaled **865,856,464 logical bytes (0.806 GiB)**. The combined removed logical size was **4,877,682,957 bytes (4.543 GiB)**. This is a file-size inventory, not a direct BeeGFS quota delta.

Afterward, BeeGFS reported **80.82 GiB used of 93.13 GiB** (about 12.31 GiB available). Quota accounting is refreshed asynchronously and other current-pilot checkpoint housekeeping occurred during this interval, so the quota change cannot be attributed exactly to these deletions. The protected A/P interface Dice baseline and `.conda/envs/hippocampus/bin/python` were checked afterward. The active step-head pilot, all `low_data*` directories, full-data bands and baseline experiments, and their needed checkpoints remain.
