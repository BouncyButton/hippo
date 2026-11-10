# Binary MedSAM3 artifact retirement

All 13 binary comparisons (39 trained models) were recorded in `BINARY_RESULTS.md` and hashed `RESULT_RECORDS.json` before deleting the generated artifacts. The preserved files were copied to BeeGFS and verified there before deletion.

Removed binary trained adapters, predictions, probability maps, plots, logs and generated analyses from the supervised pilot, replication and fold-3 extension. The cluster removal inventory contains 5,389 files totalling 2,978,554,283 logical bytes (about 2.77 GiB; mirrored storage is charged twice). Local untracked binary result folders were also removed. Pretrained weights, datasets, shared runtime, source and the three-class pilot were retained.

After cleanup, BeeGFS reported 85.31 GiB used out of 93.13 GiB: **7.82 GiB available**, exceeding the unchanged multiclass study's **7.499 GiB** submission requirement. Quota is shared with other work and may change.

Remote preservation directory: `/mnt/beegfsstudents/home/3160552/medsam3_binary_retirement_20260930`. The remote `REMOVAL_MANIFEST.json` and `REMOVAL_RECEIPT.json` record the exact removed paths and completion.
