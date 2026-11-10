# U-Net constraint pilot: job 682338

Retrieved on 2026-10-04. Fold 0, seed 0; 10 training patients and 52 validation patients; 64³ inputs, batch size 1. Each arm has a maximum allowance of 40,690,414,844,347 estimated FLOPs, including training-only coefficient calibration.

Slurm reports FAILED, exit 1:0, elapsed 00:16:51. Three runs completed. The Dice + bands + original edge output directory is empty, and the job's stderr is empty. The cause of the failure is unconfirmed; there is no validation score for that arm.

Follow-up investigation: the concurrent LEGO acquisition recorded multiple explicit `Disk quota exceeded` errors while extracting a 39,384,022,358-byte archive. That archive's download log was last written at 13:05:08; the edge arm completed at 13:08:03; the combined directory was created at 13:09:27; Slurm ended the job at 13:09:30 (all Italy time). Quota exhaustion is therefore the best-supported explanation for the missing startup files, but the U-Net process left no traceback to confirm it directly. Frozen source hashes match and a CPU-only `--help` startup check succeeds. Current quota is 90.31/93.13 GiB, leaving about 2.82 GiB. See `failure_investigation.json` for the preserved evidence and limitations.

| Objective | Best foreground macro hard Dice | Anterior | Posterior | Selected update | Selected patient-pass epoch | Final macro hard Dice | Completed updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.8111138831 | 0.8191305410 | 0.8030972252 | 1,912 | 191.2 | 0.8087892670 | 4,247 |
| Dice + bands | 0.8130693441 | 0.8200463836 | 0.8060923047 | 3,160 | 316.0 | 0.8096957046 | 4,213 |
| Dice + original edge | 0.8116627851 | 0.8189157110 | 0.8044098593 | 3,159 | 315.9 | 0.8087003483 | 4,212 |
| Dice + bands + original edge | Unavailable | — | — | — | — | — | No recorded updates |

The main score averages the anterior and posterior hard Dice, each averaged across the same 52 validation patients. Checkpoints were selected by the highest validation macro hard Dice among scheduled evaluations. These are validation selection scores from one fold and seed, not independent test results or evidence of a reproducible improvement.

Bands improved selected Dice by 0.0019554610 (0.196 percentage points); original edge improved it by 0.0005489020 (0.055 percentage points) relative to Dice alone.

Before retrieval, all seven artifact hashes in each completed run's completion record were checked remotely, including both checkpoints, configuration, metrics, budget ledger, calibration report, and selected validation report. Each consumed total was within its allowance. The completion records, selected per-patient validation results, and full metric curves are preserved under `unet/` here; checkpoints remain on the cluster.
