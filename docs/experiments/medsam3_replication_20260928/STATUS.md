# Final status: all 27 runs completed; strict final gate failed

All nine fold/seed comparisons finished and passed their individual audits. Job 673784 ended after 10:58:24 with Slurm FAILED status only because the cross-seed calibration-consistency gate failed during final aggregation. No GPU job remains active for this study and no additional training was submitted.

Observed mean Dice: baseline **82.91%**, bands **82.87%**, bands+edge **83.99%**. The combined improvement averages **+1.08 percentage points**, but the exploratory 95% interval includes zero and fold 2's mean effect is negative. This is mixed evidence, not proof of a robust generalization benefit.

Read `RESULTS.md` and `aggregate_observed.json` for all nine comparisons, statistical limitations, and the recorded calibration deviation. The original consistency threshold was not relaxed. `FINAL_INTEGRITY.json` verifies all saved adapters/predictions and 320 frozen input files after completion. Raw model and prediction artifacts remain on the cluster; local reports contain the result/audit/training metadata.

No automatic monitoring is enabled.

---

## Historical submission and manual-check instructions

# MedSAM3 replication job 673784

Resumed 28 September 2026 at 15:32 Europe/Rome. Original job 673549 stopped at an overly strict floating-point audit after finishing the first comparison; the corrected full audit passed and those results are retained. See `AUDIT_PRECISION_FIX.md`. One Slurm allocation runs 3 folds × 3 new training seeds × 3 arms = 27 fine-tuning runs. Each fold uses five training volumes. The arms are baseline, supervised bands loss, and supervised bands + edge loss. Evaluation uses 145 distinct held-out volumes across the three folds.

Remaining runtime: approximately 11 hours; allocation limit: 20 hours. No monitoring automation is configured, as requested. The base checkpoint and integrity manifest were delivered successfully to node-local memory. The worker passed its fresh quota check and is resuming after verified `fold0_seed17`; no further assistant intervention is required for normal execution.

## Manual checks

Cluster root: `/mnt/beegfsstudents/home/3160552/medsam3_replication_20260928`.

Queue and current comparison:

```sh
ssh bocconi-cluster 'squeue -j 673784; cat /mnt/beegfsstudents/home/3160552/medsam3_replication_20260928/job_673784_status.json'
```

When complete, the status is `complete`, the nine comparisons appear in `completed_runs.json`, and each completed fold has `fold_completed_0.json`, `fold_completed_1.json`, or `fold_completed_2.json`.

Final report: `analysis/RESULTS.md` under the cluster root. Detailed statistics: `analysis/aggregate.json`. Predictions, training records, final adapters and independent audits: `results/foldN_seedS/`. Logs: `logs/673784.out` and `logs/673784.err`.

If the job disappears from the queue, check the status file and accounting:

```sh
ssh bocconi-cluster 'sacct -j 673784 --format=JobID,State,ExitCode,Elapsed,MaxRSS'
```

All 27 local checks passed. The original 26 checks passed on the cluster, followed by all 10 updated replication tests after the storage-budget correction. Source hashes and submission details are saved in `SOURCE_MANIFEST.json` and `SUBMISSION.json`.

Quota before resume: **6.06 GiB free**; corrected remaining-work minimum **5.654 GiB** including mirrored copies and 1 GiB reserve. Cleanup reclaimed approximately **2.61 GiB of charged quota**, with all nine removed cluster checkpoints preserved locally and hash-verified. See `CLEANUP.md` and the archive receipts for exact restoration paths.

This tests repeatability at volume level. Patient mapping is unavailable, so it cannot establish patient-independent generalization. The analysis reports paired effects, exploratory uncertainty and consistency across folds/seeds; three folds are insufficient for strong significance claims.

Verified first comparison (fold 0, seed 17; 46 validation volumes): baseline Dice **0.83545**, bands **0.84911**, bands+edge **0.86545**. One of nine comparisons is complete; the overall replication conclusion is pending.
