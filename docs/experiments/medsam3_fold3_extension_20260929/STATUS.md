# Fold-3 follow-up: job 675308

Submitted 29 September 2026 at 10:29 Europe/Rome. Nine new fine-tuning models: fold 3 × seeds 17, 83, 191 × baseline, bands, bands+edge. Five new support volumes and 44 evaluation volumes. Expected runtime approximately four hours, allocation limit six hours. No scheduled monitoring is enabled.

The existing nine comparisons are retained and linked into this study. The status counter's total_runs=12 counts those nine completed comparisons plus the three new ones; it does not mean twelve new comparisons. The final report includes all four folds, retaining fold 2 and explicitly disclosing the historical calibration deviation. New-fold calibration must be exactly identical across seeds.

Quota before current launch: 3.77 GiB free against a 3.7173 GiB prelaunch requirement. This includes the 2.7173 GiB runtime budget (mirrored artifacts plus 1 GiB reserve) and an additional 1 GiB to cover the observed 0.50 GiB startup quota swing. Eleven older control/model checkpoints were hash-verified locally before their cluster copies were removed, reclaiming about 1.444 GiB of charged quota. See ARCHIVE_RECEIPT.json, BUFFER_ARCHIVE_RECEIPT.json and STARTUP_BUFFER_ARCHIVE_RECEIPT.json for restoration paths. No MedSAM3 models or results were deleted.

All 27 relevant tests passed locally and on the cluster. Frozen protocols and source hashes are in protocols/ and SOURCE_MANIFEST.json; submission evidence is in SUBMISSION.json.

Cluster root: `/mnt/beegfsstudents/home/3160552/medsam3_fold3_extension_20260929`.

```sh
ssh bocconi-cluster 'squeue -j 675308; cat /mnt/beegfsstudents/home/3160552/medsam3_fold3_extension_20260929/job_675308_status.json'
```

Outputs: new results/fold3_seed17, results/fold3_seed83, results/fold3_seed191; combined analysis/RESULTS.md and analysis/aggregate.json. Logs: logs/675308.out and logs/675308.err.

Normal final status may be reported_with_calibration_deviation because the old folds' failed calibration-consistency checks remain disclosed, even if all new-fold checks pass. The newly shared calibration is enforced exactly and cannot be waived by that reporting mode.

This is an exploratory extension chosen after viewing previous results. Fold 2's baseline Dice was comparable to other folds; the hypothesis that its volumes were unusually hard is unproven. This study tests sensitivity to another split rather than discarding the negative result.

The first allocation, 675249, failed its compute-node quota check before any new training. Its logs and source manifest are retained. The retry requires verified tmpfs scratch in /dev/shm and logs the compute-node quota snapshot; see STARTUP_675249.md and RESUBMISSION.json.

The second allocation 675289 also stopped before training when free quota changed from 3.11 to 2.61 GiB during startup. Current allocation 675308 includes additional startup headroom and has identical scientific settings/source to the second attempt. Latest submission evidence: RETRY_WITH_HEADROOM.json.

Startup confirmed: job 675308 passed compute-node preflight with 3.27 GiB free versus 2.7173 GiB required and advanced to fold3_seed17 on gnode02. The original nine comparisons were verified and skipped. See RUNTIME_PREFLIGHT.json for the quota snapshot.
