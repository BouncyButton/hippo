# Laptop-independent operation

The scientific source, protocols and plan are unchanged. The separately hashed operational overlay is in `operations/offline_v1/`.

The detached cluster service owns Slurm submission (at most one running and one pending account job), pretrained-model delivery from the persistent cluster copy, two-worker dispatch, and continuation across allocation time limits. Completed milestones can unlock the next 100 updates after integrity validation without waiting for the laptop. Selection remains the first two fold-0 seeds reaching mean validation Dice 0.60, followed by folds 1 and 2, with the original 1,000-update cap and third-seed initial trial.

The optional laptop mirror only copies and verifies result metadata; it cannot deliver models or submit jobs. When the laptop sleeps or disconnects, checkpoints and metadata remain on BeeGFS. Deletion still requires a genuine hash-verified durable archive acknowledgment and no remaining resume dependency. There are no synthetic archive acknowledgments. Estimated conservative offline retention: 16 GiB, plus the original 8 GiB quota reserve.

Handoff: old job 751670 checkpointed seed 0 at update 280 and seed 1 at update 200. Exact checkpoint hashes and the detached service command are recorded in `offline_state/handoff.json` on the cluster. The operations manifest binds the controller, test suite, mirror and Slurm script; the scientific manifest remains unchanged.

Inspect cluster `offline_service.json`, `status.json`, `parallel_status.json`, `submission.json`, and `offline_service_error.json` (if present). Setting `control.json` pause to true still stops scheduling and checkpoints the active workers. The service is independent of the laptop, but remains subject to cluster availability and quota.

Verified active job: **751779**, two distinct GPU workers. Cluster-only base delivery completed before the optional laptop mirror was started. Both seeds advanced beyond their handoff checkpoints: seed 0 reached update 285 and seed 1 reached update 205. Eight CPU checks passed both locally and on the cluster. Runtime evidence is in `operations/verification.json`. About 40.1 GiB of personal BeeGFS quota remained at verification.

Fold 0 results verified: seed 0 70.74%, seed 1 72.35%, seed 2 71.45% mean validation Dice, all at 400 updates. All three result archives were verified before completed shared checkpoints were retired. A launcher exit-code mismatch stopped job 751779 after seed 2 completed; preserved audit: `operations/recovery_751779.json`. Job 752229 was submitted and obtained an allocation to continue seeds 0 and 1 across folds 1 and 2, preserving fold-1 seed-0 update 315. Detailed metrics and cleanup totals are in `fold0_results.json`.
