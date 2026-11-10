# Submitted independent-target and band sweep

Job **691444** was submitted on 9 October 2026 with dependency `afterany:691005`.
It runs the frozen 189-arm matrix in one allocation: 5% SwinUNETR, fold 0, seed 0,
10 training cases and 52 validation cases. No auxiliary branch or pretrained weights.

All seven scientific checks passed. Every objective passed CPU geometry checks on
the 10 training cases, real-model loss/gradient checks, and an actual SwinUNETR
parameter update. The three reference objectives match the predecessor exactly.
CUDA preflight and all-10-training-case calibration are gates before scientific runs.

Expected runtime: roughly 16–20 hours after the dependency clears; allocation limit
23 hours 50 minutes. Resources: one A100 MIG 4g.40gb, four CPUs, 24 GiB RAM.

Retain every prediction, result, history, configuration, calibration and source.
Keep checkpoint weights for the three controls and current top five, plus the
shared warmup and one active resumable run. Estimated additional peak storage:
1.5 GiB; quota showed 4.59 GiB free at submission. Previous campaigns are unchanged.

Frozen protocol identity: `0ea239ed557dfc2b35fac0329ae0e4e664c746731cb5a8cdcd7ea7b9e9f36111`.
Payload SHA256: `84108ab1341d35aa13afc86e9d23d83ee691c51facc87abd3f2742e7ab682166`.

See README.md for the complete matrix and protocol.json for exact weights and settings.

## Outcome and recovery

Job 691444 failed in CUDA preflight after 3 minutes 39 seconds of allocation time,
before any scientific training. The recursive tensor-transfer helper did not
traverse the new lists of sampling groups, leaving weights on CPU while logits
were on CUDA. BeeGFS was not the cause; 4.29 GiB was free when the failure was checked.

Recovery job **692240** uses a separate root, a recursive transfer fix and an added
cross-device regression test. The 189-arm matrix and every scientific setting are
unchanged. See `../band_evolution_20261010_recovery/SUBMISSION.md`.
