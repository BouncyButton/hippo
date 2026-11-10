# UNETR seed-3 baseline and bands checks

Six supplementary experiments: folds 0, 1, 2 × baseline / baseline+bands; seed 3 throughout.
Original seed-1 results and the ongoing Colab sweep are preserved.

- Two isolated A100 MIG 4g.40gb workers inside one Slurm allocation, 4 CPUs and 40 GiB host RAM per worker.
- Unchanged frozen Colab scientific source and saved 10-training/52-validation-volume splits.
- No augmentation; MONAI Dice including background, AdamW at initial learning rate 0.0001, BF16, batch size 1.
- 31,976,450,667,090 optimization FLOPs per run; both arms complete 245 updates, fixed 477-update learning-rate horizon.
- Bands calibrated on training data at update 50: 10% median auxiliary-gradient target and 50% p95 ceiling, original 50-update ramp. Calibration compute remains separately accounted.
- Final-budget checkpoint is primary. Save training curves, training/validation Dice, FN/FP, A/P swaps, calibrated weights, source, configuration and logs.
- Each worker waits for a verified local durable archive before retiring only that run's bulky generated artifacts. Keep interrupted checkpoints.
- One 24-hour allocation, graceful stop and 300-second archive timeout. No prequeued continuation or automatic resubmission; interrupted work is preserved for explicit resume.
- Estimated 8 GiB logical peak for both workers (16 GiB mirrored), 4 GiB reserve, minimum personal quota headroom 20 GiB. Last verified headroom: 68.2 GiB.
- Cluster preflight passed; seed-3 initialization SHA-256: 24e0605cb7bb8fecaa4840a1bcb890fbdb0ec9b31dd8743d9c43a768260d2d41.

This is a supplementary seed-sensitivity experiment, not a replacement for seed 1. Cross-environment comparisons with Colab must identify the different GPU environment.

The standard submission controller enforces the standing one-running/one-queued limit and reuses existing active submissions. At preparation, jobs 702517 and 702580 occupy those slots. The user subsequently requested a single sbatch in response to the queue-exception question; the six runs share one two-GPU allocation behind the existing job chain.
