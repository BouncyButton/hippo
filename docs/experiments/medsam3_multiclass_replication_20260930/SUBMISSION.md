# Combined MedSAM3 study submission

Job **677580** runs folds **0, 1, 3**, each with seeds **17, 83, 191**, sequentially in one GPU allocation. Every comparison includes baseline, bands and bands+edge with background/anterior/posterior classes: 27 models. The job audits saved training/validation predictions after every comparison and aggregates validation Dice and disjoint FP/FN/A/P swaps after all nine complete.

Slurm account QoS: `stud`, maximum 24 hours per job, one running job and two submitted jobs per user. This job requests 24 hours. Separate fold jobs 677573 and 677574 were cancelled at the user's request; their partial outputs are excluded. The old submission controller was stopped.

BeeGFS preflight after binary cleanup passed with **7.81 GiB available**, against **7.499 GiB required**. Complete old binary results and cleanup manifests are preserved in the adjacent `medsam3_binary_retirement_20260930` record.

Remote study root: `/mnt/beegfsstudents/home/3160552/medsam3_multiclass_replication_20260930`. Check `combined_status.json`, `fold*_status.json`, `logs/677580.out`, `logs/677580.err`, and `BROADCAST_STATUS.json`. Final reports will be `analysis/RESULTS.md` and `analysis/aggregate.json`. A 30-minute heartbeat checks for completion or actionable failures and stays quiet during normal progress.
