# Degree bands A/B submission

**Boundary audit complete: job 667725, 2m03s, exit 0.** Neither A nor B achieved a net reduction in boundary foreground/background mistakes versus the early-stopped Dice baseline: A +201, B +497 pooled errors in the original bands. See [BOUNDARY_AUDIT_REPORT.md](BOUNDARY_AUDIT_REPORT.md). The initial audit 667718 stopped at an exact-count check; follow-up quantified only 1–2 voxel inference differences. Previous status entries below are historical.

**Training completed:** job 667630 completed successfully in 27m34s. A stopped at epoch 30 (selected epoch 22); B stopped at epoch 38 (selected epoch 30). Both selected checkpoints and the built-in four-model audit are complete.

**Focused boundary audit submitted as job 667718**, using selected checkpoints from A, B, early-stopped Dice and ordinary bands; no retraining. See `BOUNDARY_AUDIT_PROTOCOL.md` and `BOUNDARY_AUDIT_SUBMISSION.json`. Earlier status entries below are historical.

**Current job: 667630 — RUNNING on gnode02.** Submitted 24 September 2026 at 12:31 CEST with A100 MIG `3g.40gb`, four CPU cores, 32 GB host RAM and a four-hour limit. Job 667506 was cancelled before starting. The A/B losses and training settings are unchanged; the explicit calibration-source GPU-profile transfer is recorded. Fresh quota preflight passed with approximately 6.85 GiB free.

Current remote root: `/mnt/beegfsstudents/home/3160552/degree_bands_ab_20260924_02_3g`. Logs: `job-667630.out`, `job-667630.err`, `calibrate_A.log`, `calibrate_B.log`, `train_A.log`, `train_B.log`. See `JOB_667630_STATUS.json` and the GPU update in `PROTOCOL.md`. Earlier submission records below are historical.

**Resubmitted unchanged as job 667506** on 24 September 2026 at 11:33 CEST, at the user's request. Frozen source, command, A/B settings and early stopping are identical. Fresh preflight passed with approximately 6.85 GiB quota free.

Last observed 2026-09-24T09:33:38.932554+00:00: **PENDING (Resources)**; neither arm has started. Scheduler logs for this submission will be `job-667506.out` and `job-667506.err`. See `RESUBMISSION_02.json` and `JOB_667506_STATUS.json`.

The earlier submission and cancellation below are retained as history.

**Cancelled at the user's request before starting.** Slurm confirmed job 667480 as `CANCELLED`, with elapsed time `00:00:00`. Source and experiment files are retained for modifications. The submission details below are historical.

Slurm job **667480**, submitted 24 September 2026 at
10:45 CEST. Last observed 2026-09-24T08:46:09.028255+00:00:

```
JOBID        STATE         TIME               NODELIST(REASON)
            667480      PENDING         0:00                    (Resources)
```

Remote experiment: `/mnt/beegfsstudents/home/3160552/degree_bands_ab_20260924_01`.
One allocation runs A (inner normalization), then B (surface normalization),
both alpha=2. Existing Dice and Dice+bands controls are evaluated, not trained.
Early stopping: patience 8, minimum 25 epochs, min-delta 0.0005, maximum 50.

Validation before submission:
- 126 local tests passed across degree weighting, existing band behavior,
  calibration, early stopping, augmentation, training options and runner arguments.
- Shell syntax and `git diff --check` passed.
- Cluster CPU checks passed for explicit neighbour counts, both loss formulas,
  gradients and exact zero-alpha parity on PyTorch 2.12.1+cu130.
- Frozen source, historical checkpoint bindings, data/split hashes and quota
  preflight passed. Cluster pytest is not installed; the remote numerical smoke
  checks ran directly with Python and Torch without modifying its environment.

Storage before submission: **86.28 / 93.13 GiB**, approximately **6.85 GiB free**.
Removed 12 redundant latest checkpoints from yesterday's completed study:
2,358,664,372 logical bytes, recovering about 4.39 GiB of mirrored quota.
All selected checkpoints, reports, metrics and source were retained.
See `CLEANUP.json` for the deletion manifest and hashes.

Frozen source archive SHA256:
`39a260c57c03b39a5304c25fcd4c8420c25cc87d1b35336f0a85498b4e795aaf`.

After completion, inspect `audit/summary.json`, `audit/comparison.json`,
`completion.json` and each run's `completion_manifest.json` in the remote root.
Training logs are `train_A.log` and `train_B.log`; scheduler logs are
`job-667480.out` and `job-667480.err`. No performance result is available yet.

The comparison is exploratory: one seed on an already-used validation fold.
Full formulas, calibration policy and outcome measures are in `PROTOCOL.md`.
