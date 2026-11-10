# MSD Colab results snapshot

Read from Google Drive on 15 October 2026, around 20:19–20:21 Europe/Rome. Comparison file last updated 19:51. This is the 504-run replication.

Saved report: 126/504 completed; 126 unique run IDs. All 36 matched baselines complete. No reported optimization FLOPs exceed 31,976,450,667,090; calibration is accounted separately.

| Model | Baseline Dice %, 9 runs (fold mean ± fold SD) | Bands gain, pp | Bands + edge gain, pp |
|---|---:|---:|---:|
| nnUNet | 63.49 ± 4.97 | +5.46 | +4.54 |
| SwinUNETR | 55.35 ± 1.15 | +4.58 | +4.63 |
| UNETR | 40.99 ± 0.53 | -0.30 | -0.34 |
| 3D U-Net | 71.39 ± 0.94 | +6.68 | +7.24 |

Bands gains are provisional means of paired differences for fold 0, seeds 0 and 1 ONLY; they are not differences against the nine-run baseline mean displayed above. No constraint condition yet has all nine runs. Architecture comparisons are confounded by baseline losses/optimizers. Validation folds are overlapping rather than independent test datasets.

The old session records a requested interrupt with no errors. Recovery metadata for UNETR bands+edge+depth fold 0 seed 1 agrees with its charged ledger at 36 updates; the 1.48 GB recovery archive is listed, but this check did not download/re-hash it.

A new maximum-worker session appeared at 20:19. Its preservation record confirms all 126 verified results, one resume task, five-worker ceiling, no solo profiling, no throughput backoff, no augmentation, NVMe work storage and fresh local verification. The subsequent status at 20:20:09 confirms five active workers, all ready and reporting charged training updates, using about 10.34 GiB of 80 GiB GPU memory. The interrupted UNETR run advanced from its saved 36 updates to 38. Thus the new controller has resumed training; the earlier requested stop describes the old session.

The latest completed Swin run has a verified archive acknowledgment, completed cleanup receipt and no-augmentation runtime recipe. This was a results/status read with a spot-check of receipts, not a repeat hash audit of every archived file.

Sources:
- https://drive.google.com/file/d/1ekd1XOGfbbwtQkftO2IxgENEnAEnQYfI/view
- https://drive.google.com/drive/folders/1Ntc5NHA5RZ2OOGuCIQo9h_KGxc7pa4aM
