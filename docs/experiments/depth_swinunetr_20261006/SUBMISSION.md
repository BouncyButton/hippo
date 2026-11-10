# Submitted campaign

Slurm accepted job **685816** on 6 October 2026. Verified state at submission: **PENDING**, waiting for job **685757** to finish (`afterany`). The allocation requests one `4g.40gb` GPU, four CPUs, 24 GiB RAM, and 12 hours.

The allocation contains eight sequential, independent SwinUNETR runs: `dice`, `depth`, `depth_sdf3d`, and `depth_sdf_layer`, first on the fixed ten-case subset and then on all 208 fold-0 training cases. Each uses seed 0, the same 52 validation cases, and a maximum of 75 epochs with patience 8, minimum 25 epochs, and min-delta 0.0005. The baseline is Dice-only.

Local verification passed: 11 unit/integration checks, geometry validation for all 260 cases, and a real 64-cube SwinUNETR backward pass. GPU checks are queued as the allocation's first step; no training result is claimed at submission.

Sources and protocol are frozen at the remote root:

`/mnt/beegfsstudents/home/3160552/depth_swinunetr_fold0_seed0_20261006_01`

The exact accepted command and dataset/source identity are recorded in [submission.json](submission.json). The experimental definitions are in [PROTOCOL.md](PROTOCOL.md) and [protocol.json](protocol.json). Each run will write its checkpoints, selected validation metrics, predictions, and curves under `runs/<fraction>/<arm>/`; top-level `status.json` and `summary.json` describe progress and completed runs.
