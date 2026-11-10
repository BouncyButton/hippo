**Superseded:** job 682232 was cancelled and its rerun checkpoints deleted after analysis. See [final failure report](../nnunet_warmup_failure_682232/REPORT.md). The snapshot below is historical.

# Progress check: 2026-10-04 11:55 Europe/Rome

Job 682232 is running on gnode02. Two of nine runs are complete; fold 0,
seed 2 was actively advancing at charged update 131/327, with validation
recorded through update 115. The remaining six runs had not started.

| Fold | Seed | Status | Selected validation macro hard Dice |
|---|---|---|---:|
| 0 | 0 | Complete, 327 updates | 0.836279 |
| 0 | 1 | Complete, 327 updates | 0.004426 |
| 0 | 2 | Training, 131/327 charged | Pending |

The source manifest was verified, every started run uses the frozen 5% warmup
protocol, and charged arithmetic stays within the 116259-GFLOP cap. Both
completed runs have all 20 validation checkpoints and zero discarded attempts.
The error log is empty, and no run failure record was found.

**Performance concern:** fold 0, seed 1 has final anterior and posterior hard
Dice both zero; its best selected macro hard Dice is only 0.004426, at update 17.
Warmup has therefore not established a reliable solution to the class-learning
failure. Operational progress is correct, but segmentation quality is not
consistently good. All nine runs continue under the same frozen recipe.

Completed-run training/validation/checkpoint sections take roughly 40–43
seconds, but the gap before each next configuration is written is roughly 91
seconds for imports, model initialization, and data preparation. The earlier
10–15-minute total estimate underestimated this overhead. At this check,
approximately 15 minutes remained, or roughly 20–25 minutes total runtime.

The first full progress audit is saved as `progress_audit_latest.json` in this
directory. This status document includes the later update-counter check.
