# GeoFWI3D: matched three-seed comparisons on three folds

The requested repetition includes all six objectives: Dice + cross entropy alone, and that same base plus bands, signed distance, bands + signed distance, bands + edge, or bands + signed distance + edge. There is **one sbatch allocation**, with all runs executed sequentially on one GPU.

The six completed fold-0/seed-0 pilot runs are reused after checking their completion records, checkpoint and result hashes, case identities and shared protocol settings. New training comprises 48 runs:

- Fold 0: seeds 1 and 2, six objectives each (12 runs).
- Folds 1 and 2: seeds 0, 1 and 2, six objectives each (36 runs).

This yields 54 fold/seed/objective cells including the original six. Model seed controls initialization, sample order and augmentation. Within each fold, labelled training IDs and validation IDs are fixed across seeds and objectives, so this varies training randomness while keeping the labelled subset fixed.

## Folds and data

Archive 0's existing 479 salt-containing volumes, normalized seismic images and corrected physical salt masks are reused without copying or modifying the data. The original fold-0 manifest, training pool order, 19-volume subset and 96-volume validation set are preserved exactly.

To build genuine disjoint validation folds while keeping the original split, its 383-volume training complement is shuffled deterministically within the two fault-presence strata (NumPy seed 0) and partitioned into four groups. Together with original fold 0, this defines five nonoverlapping validation partitions of sizes 96, 96, 96, 96 and 95, covering all 479 cases. Only folds 0, 1 and 2 are trained here. Each has 96 validation volumes and a 383-volume eligible training pool. No simulation seed crosses a training/validation split.

New fold training pools are shuffled within strata using `SeedSequence([0, fold, 32091])`, then proportionally interleaved using the original subset algorithm. The first 19 volumes constitute approximately 5% of each eligible training pool. This split seed is separate from model seeds 0, 1 and 2. Dataset views consist of small manifests and links to existing image/mask directories; directory names retain their original storage locations, while manifests define fold membership.

## Training and analysis

The actual trainer and binary loss module are copied byte-for-byte from the frozen four-loss campaign of 5 October 2026. Model architecture, native 96³ grids, normalization, Dice + CE objective, AdamW, augmentation, auxiliary coefficients/ramps and evaluation are unchanged. Each new run trains from scratch with 500 optimizer updates, validation every 100 updates, the original 5,000-update LR horizon and 100-update warmup. Early stopping cannot trigger before this short cap; the exact maximum mean validation Dice selects the checkpoint.

The job exports each run's history, selected validation record, per-volume Dice CSV, run configuration, completion record and retention verification. After each arm it refreshes a partial comparison. At completion, `results/summary.json`, `per_run_dice.csv` and `RESULTS.md` contain per-fold three-seed mean/sample SD, mean/sample SD over nine fold/seed cells, and gains paired against baseline using the same fold and seed.

These are selected validation results without test evaluation. The 500-update budget was chosen from the original pilot validation curve; there is no new tuning to obtain a particular Dice value. Cross-fold training pools overlap, and standard deviations describe variation rather than confidence intervals over nine independent datasets. Overall Dice does not establish a separate boundary-specific metric improvement.

## Storage and queue safeguards

Initial BeeGFS user quota was **90.62/93.13 GiB**, leaving **2.51 GiB**. Standard full checkpoint retention for 48 runs would require about 11.9 GiB logical storage, roughly twice that under buddy mirroring, so that policy is not used. Existing data, checkpoints and completed experiments are preserved.

Each new arm uses a private directory on node-local scratch for full `best.pt` and `latest.pt` checkpoints. The unchanged trainer reloads the best checkpoint and reproduces its selected Dice. All completion-record artifact hashes, validation identities and per-volume Dice counts are then checked. JSON/CSV artifacts are copied to BeeGFS; the completion marker is copied last. Only after successful export verification is that arm's private temporary directory removed. **New model checkpoints are not retained**, and this is stated in each arm's retention record. Completed results remain resumable/skippable, while an interrupted unexported arm must be retrained if its scratch directory is lost.

Estimated retained records are below 50 MiB logical / 100 MiB mirrored quota. Submission and allocation startup require at least **1.25 GiB free user quota**, accounting for applicable group limits and finite inode limits. Each arm rechecks that reserve. The allocated node must provide at least **4 GiB scratch space on a filesystem distinct from BeeGFS**, checked before training and each subsequent arm. No dataset copy is made on scratch. Quota service data can lag by 30 seconds; the reserve gives room beyond the projected outputs.

Submission is idempotent under a lock and checks the whole account queue immediately before calling sbatch. It refuses if any job is already waiting or more than one is active. If one job is active and no job is queued, it adds an `afterany` dependency to prevent concurrent allocations. Resources are the prior 40 GiB GPU profile, four CPU threads, 32 GiB host RAM and a four-hour wall-time limit. Based on approximately two minutes per completed pilot arm, expected runtime is roughly 1.5–2 hours after allocation, with allowance for checks and I/O. No recurring monitor is created.

## Preparation checks

Local checks cover exact preservation of the pilot split, deterministic disjoint stratified validation partitions, quota parsing with finite user/group and inode limits, rejection of corrupted/missing per-case results, and aggregation requiring all 54 matched cells. Bash syntax is checked separately. Cluster verification hashes every shared dataset file, all frozen runtime/input files and the retained original pilot checkpoint/result artifacts before submission.

All four local checks passed in 1.13 seconds and all four cluster checks passed in 2.76 seconds; Bash syntax passed. After retrieval, all **41 frozen input hashes** matched, and the trainer/data/loss files matched the prior frozen campaign byte-for-byte. Fold 0 matched the original manifest exactly; the three validation sets were independently checked for disjoint IDs.

## Submission

Submitted **Slurm job 694083** on 11 October 2026, named `geofwi3d-3fold-3seed`, from an empty account queue, with no dependency required. Slurm reported it running on `gnode04`, starting at cluster-reported 11:25:18 Europe/Rome. The final pre-submission quota check was **83.29/93.13 GiB used**, leaving **9.84 GiB free**, above the 1.25 GiB reserve; no pre-existing files were deleted by this campaign.

Cluster campaign: `/mnt/beegfsstudents/home/3160552/geofwi3d_3fold_3seed_5pct_500updates_20261011_01`. Local frozen inputs, pilot references, submission and quota receipts are under `reports/geofwi3d_replication_20261011/cluster/`; local verification is `reports/geofwi3d_replication_20261011/verification.json`.

Allocation startup passed: node-local `/tmp` had **409.59 GiB free**, with a different filesystem device from BeeGFS; user quota remained **9.84 GiB free**. The first active arm was fold 0 / seed 1 / baseline, with no failure record. The error log contained only the BeeGFS quota service's standard 30-second usage-refresh notice.

## Completion

Job 694083 completed with exit 0:0 at **13:07:45 Europe/Rome on 11 October 2026**, after **1 hour 42 minutes 27 seconds**. All 48 new runs and all 54 matched cells are complete. Local collection independently verified artifact hashes, fold/seed identities, all 5,184 per-volume Dice rows and aggregate statistics. Final free BeeGFS user quota was **9.09 GiB**. See [RESULTS.md](RESULTS.md) for the complete three-fold / three-seed comparison and links to all records.
