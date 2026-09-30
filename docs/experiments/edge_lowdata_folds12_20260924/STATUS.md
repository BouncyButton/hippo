# Running: bands + edge, folds 1 and 2

Job **668085**. Six new models, folds 1/2 and seeds 0/1/2. Same 5% subset protocol per fold, no augmentation, minimum 60/maximum 75 epochs, patience 8 and delta 0.0005. Uniform edge consistency without degree weighting. Existing twelve Dice/bands controls and epoch-five calibration checkpoints reused. Fold 0 preserved.

All control export hashes, calibration checkpoint/config hashes and fold-specific split hashes passed preflight. BeeGFS 89.45/93.13 GiB (3.68 GiB free), above the 2.7 GiB full-job reserve. Quota checked again before every seed. Compact best/latest checkpoints are inference-only. Automatic training and validation audits compare Dice, bands and bands+edge, boundary FN/FP, AP swaps, degree strata, surfaces and learning speed.

18 relevant working-source tests and 13 frozen-payload tests passed. Python compile and shell syntax checks passed. Payload SHA4735b87281da0bd77b094c04d7a08dd7df12da9986b614510a622a980c00f646.

User-requested deletion completed: remote degree-weighted A completed/failed run directories (607,796,312 logical bytes, approximately 1.13 GiB mirrored) and local result reports/payload. Unweighted experiments, baseline controls, datasets and calibration checkpoints retained. See CLEANUP.json; no negative result files retained in this task's degree-A result folder.

Remote root: `/mnt/beegfsstudents/home/3160552/edge_lowdata_folds12_20260924_01`.

Latest scheduler snapshot:

```
JobID|State|Elapsed|NodeList|ExitCode
668085|RUNNING|00:00:32|gnode01|0:0
668085.batch|RUNNING|00:00:32|gnode01|0:0
```

See STARTUP.json for the first live check and SUBMISSION.json for the submission receipt.

Startup verified: fold 1, seed 0 completed its training-only calibration and reached epoch 6 with finite losses. Edge coefficient 0.008676807665800899, inherited bands coefficient 0.0023759402992420972. No failure record; only existing AMP deprecation warnings. Job RUNNING on gnode01 at 1m50s.

Progress check 2026-09-24T17:02:59.548130+00:00: job still RUNNING. Four of six runs and audits complete: fold1 seeds0/1/2, fold2 seed0. Fold2 seed1 started, seed2 pending within sequential job. Partial results saved in PROGRESS.json; final comparison awaits remaining seeds.

Final status checked 2026-09-24T17:58:24.041602+00:00: JobID|State|Elapsed|ExitCode; 668085|COMPLETED|00:33:42|0:0; 668085.batch|COMPLETED|00:33:42|0:0;  All six runs and their train/validation audits completed. See SUMMARY.json.
