# Training launcher verification

The complete comparison is specified by
[`protocol.json`](../../experiments/compute_matched_baselines_20261003/protocol.json)
and submitted through
[`run_compute_matched_baselines.sbatch`](../../scripts/run_compute_matched_baselines.sbatch).

Six automated checks passed locally (59.025 seconds) and in the existing cluster
environment (24.787 seconds), using temporary synthetic data and CPU execution:

1. Literal 116,259 GFLOP caps, all 27 model/fold/seed identities, and LR/validation
   schedules that fit their allowed updates.
2. Ten-case subset provenance, disjoint validation sets, and consistency with
   the original dataset folds.
3. Actual 64³ forward, backward, optimizer, validation and selected-checkpoint
   publication for U-Net, SwinUNETR and the nnU-Net-derived baseline.
4. Interruption before checkpoint commit: lost work remains charged on resume,
   reducing retained updates rather than exceeding the allowance.
5. Interruption after checkpoint commit: selected weights and metric files are
   recovered without consuming another training update.
6. Completed-run idempotence, all-run aggregate verification, and rejection of
   changed checkpoint/validation artifacts.

`bash -n scripts/run_compute_matched_baselines.sbatch` passed locally and on the
cluster. The original dataset SHA-256 was verified on the cluster. The `stud`
partition advertises the requested `gpu:4g.40gb` resource, and its time limit
accommodates the 23-hour-50-minute request. `sacctmgr show qos` reports four submitted
jobs and two concurrently running jobs for `stud`. A 27-task array was rejected
by the scheduler dry run. Following the user's explicit requirement of one
running job and one queued job, the final launcher uses **one job** that runs
all 27 model/fold/seed combinations sequentially, with no array directive.

The isolated source bundle is staged at:

```text
/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261003/compute_matched_baselines
```

All 207 transferred source/protocol files were checked against their SHA-256
manifest. The CPU-tested training-source archive SHA-256 before the single-job
launcher adjustment was
`086b502a75d98958252e48747a28c111ad493dcf9f402f04f4c970a8c1f3a738`.
The cluster test log is `cluster_cpu_checks.log` inside that directory. Tests
use the cluster's existing PyTorch 2.12.1+cu130, MONAI 1.6.0 and
dynamic-network-architectures 0.4.4 environment, and exact parameter counts
match the profiled configurations.

These checks establish code execution and artifact accounting. They do not
establish baseline accuracy, convergence, CUDA execution, or exact GPU hardware
instruction counts. No full baseline training run was performed by these CPU
checks. Final single-job scheduler validation and submission are recorded
separately.

The final single-job script passed both `bash -n` and `sbatch --test-only`.
It was submitted once as **job 681342** on 2026-10-03 at 17:23:54 UTC. The local
[`submission.json`](../../experiments/compute_matched_baselines_20261003/submission.json)
records the immutable script/protocol hashes, exact allowances and result path.
The final deployed archive hash is
`2dc1b980f387254740691da17d7e9152e15a4cab2e9f4808a246c2d7e6e9faeb`.

After submission, job 681342 was observed running on `gnode01` with an
`NVIDIA A100 80GB PCIe MIG 4g.40gb` allocation. U-Net fold 0 seed 0 completed its
first two GPU training/validation milestones (607 and 1,214 updates), and its
ledger subsequently recorded 1,561 charged attempts with no failure manifest.
This verifies initial CUDA training, validation and budget bookkeeping for
U-Net; the complete 27-run campaign remains in progress.
