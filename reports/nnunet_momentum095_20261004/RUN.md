# nnU-Net momentum diagnostic: job 682319

**Retired:** the three run directories and checkpoints were deleted at the user’s request after preserving their records. See [retirement record](RETIREMENT.md). Results below are historical.

**Final status:** all three seeds completed successfully in 7:46 on 2026-10-04 at 12:42:58 Europe/Rome. Mean selected validation Dice is 0.851169 ± 0.001470 (sample SD). See [final results](STATUS.md).

Submitted 2026-10-04 at 10:35:11 UTC (12:35:11 Europe/Rome).
Verified RUNNING on gnode02 at 1:31 elapsed. The allocated A100 MIG GPU preflight passed and printed momentum 0.95, no warmup, three reruns, and 327 updates per run; logs had no errors.
Exactly one Slurm job executes fold 0 seeds 0/1/2 sequentially.

The sole training change relative to original campaign 681342 is SGD momentum
0.99 → 0.95. All runs start fresh using their original seed recipes, with the
original polynomial schedule and no warmup. The failed warmup recipe is not
combined with this intervention. No automatic reinitialization or extra
recovery updates are allowed.

Each run has the same ten training patients, 52 validation patients, batch one,
64³ inputs, architecture, foreground Dice+CE/deep supervision, BF16 precision,
Nesterov SGD, initial LR 0.01, weight decay 3e-5, clipping at 12, and 20 fixed
validation checkpoints. All three seeds will be reported, including any failures.

327 updates equal 32 complete patient-pass epochs plus seven updates (32.7
pass equivalents). Declared cost is 116181.104090427 GFLOPs per run, below
116259 GFLOPs. Three runs add 348543.312271281 GFLOPs of research training
compute; previous spent compute is retained separately in the experiment record.

Cluster outputs: `/mnt/beegfsstudents/home/3160552/nnunet_momentum095_682319`.
Cluster package: `/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_momentum095`.
Logs: `slurm-hippo-nnunet-m095-682319.out` and `.err` in the package directory.
The source snapshot and hashes remain in the output root. The final summary
requires all three seeds and describes one fold with three seeds.

Checks: ten local regression tests passed in 47.791 seconds with pinned
profiling dependencies; three cluster optimizer/loss preflight tests passed;
archive/staged-file and dataset hashes passed; bash syntax passed. The launcher
also verifies versions, BF16 GPU support, and staged source hashes before training.

Based on the prior serial campaign, allow approximately 5–10 minutes from job
start, with uncertainty from source/data loading and cluster load. The allocated
wall-time maximum is one hour. Running status does not establish that the
hypothesis fixes segmentation; inspect anterior and posterior Dice after completion.

This is a post hoc fold-0 diagnostic, not a full three-fold baseline. Do not
select the better recipe independently for each seed. Source protocol and
submission hashes are in [submission.json](submission.json). See
[validation details](VALIDATION.md).
