# Sagittal contour head pilot

> **Withdrawn, 2026-09-22.** The user stopped this experiment and the fold-1 follow-up. All A–D arm directories, checkpoints, and probabilities from this pilot were deleted, along with its local results archive. The figures below are retained only as a record of why the experiment was stopped. The hard head mask was applied at inference for a secondary evaluation; the requested direct training-time enforcement of segmentation by head predictions was not implemented. The head losses influenced the shared decoder indirectly, but did not constrain segmentation logits to follow the head contour.

Exploratory MSD fold-0 DEV cohort (52 previously studied cases). Checkpoints selected only on 42 inner-validation patients from the 208 outer-training cases.

| Arm | Selected epoch | Inner-val Dice | DEV union Dice | DEV posterior Dice | DEV anterior Dice | HD95 (voxels) | Auxiliary weight | Runtime (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21 | 0.8656 | 0.8927 | 0.8765 | 0.8623 | 1.180 | 0 | 0.09 |
| B | 20 | 0.8636 | 0.8915 | 0.8746 | 0.8588 | 1.204 | 0.0903012 | 0.10 |
| C | 18 | 0.8635 | 0.8920 | 0.8761 | 0.8588 | 1.166 | 0.00212514 | 0.09 |
| D | 21 | 0.8642 | 0.8929 | 0.8785 | 0.8635 | 1.169 | 0.00165616 | 0.10 |

Paired patient bootstrap intervals are percentile 95% intervals with 2,000 resamples. A positive Dice change favors the first arm; a negative edge MAE change favors the first arm.

- B vs A union Dice: -0.0012 (95% interval -0.0029 to +0.0004; n=52).
- C vs A union Dice: -0.0007 (95% interval -0.0025 to +0.0012; n=52).
- D vs A union Dice: +0.0002 (95% interval -0.0010 to +0.0015; n=52).
- D vs C union Dice: +0.0009 (95% interval -0.0007 to +0.0026; n=52).
- C head minus its segmentation lower edge MAE: +0.011 voxels (95% interval +0.003 to +0.020; n=52).
- C head minus its segmentation upper edge MAE: +0.010 voxels (95% interval -0.002 to +0.022; n=52).
- C disagreement: 8781 columns; head closer 3483, segmentation closer 4482, tied 816; correction fixed 4499 and broke 6388 voxels.
- D head minus its segmentation lower edge MAE: +0.001 voxels (95% interval -0.005 to +0.006; n=52).
- D head minus its segmentation upper edge MAE: +0.001 voxels (95% interval -0.007 to +0.009; n=52).
- D disagreement: 7639 columns; head closer 3158, segmentation closer 3638, tied 843; correction fixed 4105 and broke 5640 voxels.

The head correction sets foreground exactly where predicted presence is at least 0.5 and rounded ordered expected z edges enclose the voxel; added voxels take the larger predicted foreground class probability. No reference occupancy is used at inference.

The D transition loss compares signed edge displacement for every ordered y pair inside each connected annotated run, with exp(-(distance-1)/4) weighting, SmoothL1 error, and a soft upward hinge allowing annotated upward excursions. The term is multiplied by 32; total auxiliary weight is calibrated from TRAIN decoder-feature gradients to a 0.1 auxiliary/Dice RMS ratio.

Fold 0 was exploratory DEV, not independent confirmation. Small run-level `case_metrics.csv`, `summary.json`, and `config.json` remain for audit. Per-arm `done.json`, checkpoints, and probabilities were deleted.

## Run and artifact record

Slurm job `665924` completed with exit code `0:0` in `00:18:38`. It resumed the original job `665917` in the unchanged run root `/home/3160552/hippopotamus_runs/step_head_pilot_20260922_665917` after BeeGFS quota pressure interrupted that job. All four arms evaluated 52 distinct outer-DEV patients; `case_metrics.csv` has 208 rows. The selected checkpoints and their verified SHA-256 hashes are:

| Arm | Checkpoint relative to run root | SHA-256 |
|---|---|---|
| A | `A/best_epoch_021.pt` | `acb17e6226b83e6b79ca1ec59fcb06b8cb0766edca344b55439d2d6895eaa68d` |
| B | `B/best_epoch_020.pt` | `ef2ad3862e33e6ae7892304fb992f0b45ec0e5164060d860c62a0c47cd20e7e5` |
| C | `C/best_epoch_018.pt` | `f98ec5f036e6b83ecc0788ea96b9be93ba901e40afb2306833dbc5e59e9902ca` |
| D | `D/best_epoch_021.pt` | `e97c9d5ab172276275f58561e31ab493e956e058993595fb79ba09c0419e489b` |

The selected checkpoints, 52 per-arm probability files, and local results archive were deleted at the user's request. Small cluster run-level logs and summaries remain. The frozen pilot source hash was `dd97bbf585d2a3b5217a7f2c17c36a4630f71e89452d7301f0b8b0a192feecd8`.

The head's hard contour correction broke more voxels than it fixed in both C and D. The paired Dice intervals include zero, so this pilot does not justify promoting the head or transition loss as a full-data improvement. The D head's edge MAE matched its segmentation edges within the reported intervals; it did not provide a reliably better patient-specific contour cue.

## Follow-up diagnosis: how good was the head?

The head learned a meaningful contour. For D, across the 52 DEV cases, its lower and upper edge predictions were within one voxel of the reference on **95.35%** and **92.03%** of columns where both head and reference detected foreground. The segmentation achieved 95.37% and 92.02% on its corresponding columns. D's lower/upper edge MAE was 0.454/0.586 voxels, versus segmentation 0.454/0.585. These conditional edge metrics omit columns missed or falsely declared occupied.

For D, the head detected 27,561 reference-occupied columns, missed 1,458, and declared 2,430 empty columns occupied (presence precision 0.919, recall 0.950). Its segmentation detected 27,584, missed 1,435, and falsely occupied 1,839 (precision 0.937, recall 0.951). Thus the head's main measurable presence disadvantage was extra false-positive columns. On the 7,639 columns where the hard head and segmentation differed, the head contour was closer to reference in 3,158, segmentation in 3,638, and they tied in 843.

A read-only decomposition using the saved probabilities and the same transformed DEV labels tested whether the two-edge shape representation was itself inadequate. Giving an oracle the reference presence and reference lower/upper edges and filling between them yielded **0.9994 mean union Dice**; only 192 extra voxels across 132 of 29,019 occupied columns came from internal gaps. The representation can express these masks almost exactly. D's predicted head edges with *reference presence* yielded 0.9088 Dice; this is an oracle diagnostic, not an inference-time result.

Simple deployable combinations remained below D's segmentation Dice of 0.8929: segmentation presence with head edges scored 0.8915, and head presence with segmentation edges where available scored 0.8893. D's standalone head mask scored 0.8881. In a post-hoc DEV-only presence-threshold sweep, changing the threshold from 0.5 to 0.6 raised the reconstructed head mask only to 0.8885. This threshold was examined on DEV and is **not** an independently selected model. The evidence points to a good but largely redundant contour estimate, with small edge differences and a measurable presence penalty, rather than a representational failure.
