# Startup failure 675249

The first allocation stopped before any new comparison started. Login-node preflight saw 2.85 GiB free against 2.7173 GiB required, but compute-node preflight reported insufficient quota. Its original version did not print the rejected quota snapshot, so the exact discrepancy cannot be established retrospectively. No scientific settings or result data were changed.

The worker had selected /tmp based on a different filesystem device ID. That alone does not prove quota isolation across multiple mounts. The retry explicitly requires /dev/shm to report filesystem type tmpfs and at least 8 GiB free, reproducing the RAM-backed storage used for the completed replication. This is a protective infrastructure change, not proof that /tmp caused the discrepancy. Compute-node quota is now logged before broadcast and at preflight, including failed checks. Two additional old model exports are archived locally to increase the quota buffer. Original source manifests, logs and submission records are retained.

## Second allocation 675289
The explicit quota snapshots show 90.02/93.13 GiB used before checkpoint broadcast and 90.52/93.13 GiB used at preflight afterward: a 0.50 GiB startup increase. The scratch filesystem was verified tmpfs, so the original /tmp choice does not explain this second discrepancy. The precise source of the quota swing is unresolved. The guard correctly aborted before any new training.

The next submission requires the original 2.7173 GiB runtime artifact/reserve allowance plus 1 GiB of extra pre-submission startup headroom. The runtime guard continues requiring the full remaining artifacts plus its 1 GiB reserve; it is not relaxed. More old checkpoints are archived locally and hash-verified before deleting cluster copies. Training source and frozen study configuration are unchanged from the second allocation.
