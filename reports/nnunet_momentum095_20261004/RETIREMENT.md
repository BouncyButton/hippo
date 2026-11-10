# Retirement of nnU-Net job 682319

The user requested deletion of the three completed momentum-0.95 runs ahead
of rerunning under a new training budget. The completed job was verified absent
from the queue, and all three completion/configuration/artifact bindings were
checked before preservation and deletion.

Deleted target: `/mnt/beegfsstudents/home/3160552/nnunet_momentum095_682319/nnunet`.
Deleted at UTC: 2026-10-04T10:49:58.714017+00:00.
Removed: 728,988,940 bytes (about 729 MB), including all three runs'
best/latest checkpoints. The target directory is verified absent.

Preserved 37 records, including all run metrics, validation scores, configurations,
budget ledgers, completions, campaign summaries, scheduler logs, protocol, and
frozen trainer/launcher sources. The archive and every individual record hash
were verified on the cluster and locally before deletion. Checkpoint binaries
were deliberately excluded from the preserved record archive.

The frozen source and summary remain in the job output root, which has a
`NNUNET_RETIRED.json` marker with `do_not_resume: true`. The submission package
and momentum-0.95 implementation remain available for preparing a new protocol.
No replacement job was submitted; a new FLOP budget has not yet been specified.

The historical result remains 0.851169 ± 0.001470 validation Dice over the three
fold-0 seeds. All original spent compute remains reported: 348543.312271281
GFLOPs for these three runs. Deleting artifacts does not erase experimental cost.

- [Preserved records](run_records.tar.gz)
- [Retirement manifest](retirement_manifest.json)
- [Historical results](summary.json)

Archive SHA-256: `e0145194f0b33e3a41f451c815478896ed07825d0b845597175da7be5d9b3e25`.
