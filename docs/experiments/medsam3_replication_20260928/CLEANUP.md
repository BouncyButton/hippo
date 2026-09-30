# Storage cleanup before replication

Archived all three pilot adapters and six full checkpoints from older experiments to this Mac. All nine files were independently SHA256-verified locally before removing their redundant cluster copies. Lighter model/best files, metrics, logs, source and datasets remain on the cluster.

Total logical bytes reclaimed: 1,400,126,030 (1.304 GiB). BeeGFS uses two-copy buddy mirroring, so the expected charged-quota reduction is 2.608 GiB.

The replication needs 2.616 GiB of logical artifacts, charged as 5.232 GiB with mirroring, plus a 1 GiB reserve. The mandatory preflight threshold is therefore 6.232 GiB free.

Exact hashes and restoration paths are in `pilot_adapter_archive.json` and `checkpoint_archive.json`. Archiving older checkpoints is storage management; it does not classify all those experiments as unsuccessful.
