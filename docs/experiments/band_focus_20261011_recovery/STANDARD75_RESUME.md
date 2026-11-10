# Standard75 scope amendment

The user requested interruption of job 694515 and deletion of the longer, slower-decay experiments. Slurm confirmed cancellation and the worker lock was released.

The active scope keeps exactly 190 original arms and removes `long_dice`, `long_bands`, and `long_focus25`. The training budget stays at 75 epochs, StepLR every 20 epochs with gamma 0.5, early stopping minimum 25 epochs/patience 8/min_delta 0.0005. All runs remain 5%, fold 0, seed 0, ordinary SwinUNETR.

The original protocol identity and scientific source hashes are preserved. A separately hashed scope amendment restricts the allowed IDs; its launcher cannot select an extended-schedule run. Existing per-run identities, shared five-epoch initialization, frozen gradient calibration, model/optimizer/scheduler state and random-number states remain valid. The existing successful GPU and calibration gates cover every retained arm; there is no new loss implementation to recalibrate.

Four local checks passed: exact ordered subset with unchanged input/protocol identity, rejection of schedule changes, rejection of an unexpected exclusion set, and equality of every original scientific source hash. Python parsing and shell syntax checks passed. Evidence: [standard75_local_checks.json](standard75_local_checks.json).

The removal covers the three run directories (weights, predictions, metrics and configuration), their calibration directories and their individual logs. The active report and checkpoint-retention list exclude them. Immutable historical protocol and preflight evidence remain as provenance; the removed variants are not eligible for ranking or resubmission.

Replacement submission: **696104**, accepted by Slurm and observed RUNNING on gnode04. Old job **694515** is CANCELLED. The replacement uses the same resource limits (one 4g.40gb GPU, 4 CPUs, 24 GiB RAM, 23:50 walltime). Submission quota check found 22.46 GiB free. Receipt: [standard75_submission.json](standard75_submission.json).

Verified deletion and preservation: **13 completed standard75 runs retained**, 177 unfinished. `hard_q10_mix75` retains its epoch-5 checkpoint (the common warmup), including optimizer, scheduler and random-number states. Exactly nine paths for the three excluded runs were deleted, totaling 212,215,768 bytes. No extended-schedule run remains in the active reports or checkpoint-retention list. Receipts: [standard75_scope.json](standard75_scope.json), [standard75_deletion.json](standard75_deletion.json).

Startup confirmed: job **696104** passed source/data/dependency/GPU verification, retained valid calibration for all 190 active objectives, and resumed `hard_q10_mix75` from epoch 5. The live run configuration confirms max75/StepLR20 and the original early-stopping settings. All 13 completed run records were independently verified unchanged. Slurm stderr contained only the informational 30-second quota-cache notice. Evidence: [standard75_startup_verification.json](standard75_startup_verification.json).
