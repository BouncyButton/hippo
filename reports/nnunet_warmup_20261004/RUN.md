# nnU-Net warmup rerun: job 682232

**Final status: CANCELLED and rerun directories deleted.** The user stopped the
job on 2026-10-04 at 09:58:11 UTC after four completed runs. After the checkpoint
audit, 994052730 bytes of rerun artifacts were deleted; metrics, logs, and audit
evidence were preserved. There is no completed nine-run aggregate. See the
[failure report](../nnunet_warmup_failure_682232/REPORT.md).

The submission details below are historical.

Submitted on 2026-10-04 at 09:47:20 UTC (11:47:20 Europe/Rome).
Initially verified running on gnode02; exactly one Slurm job was submitted.

- Nine serial fresh runs: folds 0/1/2 × seeds 0/1/2.
- Same original 10 training patients and 52 validation patients per fold.
- Same batch one, 64³ architecture, loss, SGD, precision, and validation selection.
- 17 warmup updates, then 310 polynomial-decay updates; **327 total updates**.
- Estimated cost 116181.104090427 GFLOPs per run, within the **116259-GFLOP cap**.
- New runs use one revised recipe consistently. No per-seed choice between recipes.
- Original nine runs are an additional historical compute expenditure; the revision is post hoc.

Cluster outputs: `/mnt/beegfsstudents/home/3160552/nnunet_warmup_682232/`.
Cluster package: `/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_warmup/`.
The package contains `slurm-hippo-nnunet-warmup-682232.out` and `.err`.
After all nine completions, the job verifies every artifact/budget and writes
`summary.json`, `summary.csv`, and `summary.md` in the output directory.

The original `/mnt/beegfsstudents/home/3160552/compute_matched_baselines_681342/nnunet`
directory was deleted after preserving and verifying all records. 2253176653
bytes were removed. U-Net, SwinUNETR, original summaries, and frozen source remain.

Preserved local records: `old_nnunet_records.tar.gz` and `retirement_manifest.json`
in this directory. The archive contains 68 records, including all original
metrics, validation records, configurations, budget ledgers, completions, original
protocol, and campaign summaries. Its SHA-256 is
`1dfd6093b240beffc45d0b2a2f05cf06e4574d19ab02b355755fe9d317a1d52f`.

Validation before submission: 9 local regression tests passed (43.845 seconds),
3 cluster preflight checks passed, and `bash -n` passed. Original nnU-Net timed
training sections totalled 365.84 seconds; allow roughly 10–15 minutes from job
start for this campaign, including startup and data preparation. This is an
estimate, not a Slurm guarantee; the requested maximum wall time is four hours.

Warmup is an experimental intervention; successful segmentation is not yet
established by submitting the job. Read its completed validation summaries before
claiming the anterior failure is resolved.
