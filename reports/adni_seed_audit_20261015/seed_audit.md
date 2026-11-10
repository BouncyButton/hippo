# ADNI baseline seed audit

Snapshot: 15 October 2026, 20:32 Europe/Rome (18:32 UTC).

28/36 baseline results available; campaign snapshot contains 103/504 completed runs.

Primary metric: final-budget validation foreground Dice (%), not selected best-checkpoint Dice.

| Model | Fold | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| nnUNet | 0 | 72.65 | 75.18 | 75.92 |
| nnUNet | 1 | 75.61 | 77.07 | 79.16 |
| nnUNet | 2 | 74.28 | 75.86 | 76.92 |
| SwinUNETR | 0 | 67.09 | 65.10 | 28.35 |
| SwinUNETR | 1 | 71.38 | 71.27 | 28.65 |
| SwinUNETR | 2 | 69.93 | 70.40 | 31.25 |
| UNETR | 0 | 59.69 | 48.43 | 59.48 |
| UNETR | 1 | 63.74 | 53.06 | 60.64 |
| UNETR | 2 | — | — | — |
| 3D U-Net | 0 | 66.80 | 65.85 | 61.62 |
| 3D U-Net | 1 | 73.09 | — | — |
| 3D U-Net | 2 | — | — | — |

— = absent from the captured completed-results report; not a failed run.

SwinUNETR seed 2 is markedly weak in all three folds. Training Dice is also low (29.63–33.87%). Pooled validation precision is 16.66–18.78%, while recall is 96.05–98.26%, showing excessive foreground predictions.

UNETR seed 1 is lower than seeds 0 and 2 in both available folds. Its training Dice is 65.85–68.35%; fold 2 is not yet available.

All five weak-run archives were independently downloaded and fully hash/readability verified against their archive manifests. Acknowledgments, cleanup receipts, frozen protocol identities, canonical initialization hashes, frozen subject splits, absence of augmentation, full planned update counts, finite recorded losses/metrics and charge ledgers passed. SwinUNETR used 206 updates (99.835% of its cap); UNETR used 245 (99.629%). Discarded attempts were zero.

These checks did not reveal a crash, truncated run or wrong seed/split. They do not prove the cause of poor optimization. Do not exclude or replace seeds based on observed Dice; keep all predefined seeds and compare constraints against their matched model/fold/seed baselines.

No GPU computation, reruns, job changes or modifications to Drive results were performed.

[Source comparison.json](https://drive.google.com/file/d/1cY1ssoZMiuvOFYQMOLORMaBWqCcPea6k/view)

Local evidence: audit_results.json, baselines.csv, provenance.json and the five downloaded runs/.
