# First nnUNet/Swin refinement screen

Job 749551 completed successfully at 17:09:08 CET on 9 November 2026 (25m34s). All 19 result archives were verified locally and all 19 cleanup receipts checked. Neither model produced an eligible replacement; no automatic confirmation runs were launched.

| Model | Candidate | Positive baseline gains | Mean gain vs baseline (pp) | Mean gain vs original bands (pp) |
|---|---|---:|---:|---:|
| nnunet | late60 | 1/3 | +0.492 | -0.078 |
| nnunet | late30 | 1/3 | +0.246 | -0.324 |
| nnunet | early30 | 2/3 | +0.314 | -0.256 |
| swinunetr | quarter10 | 2/2 | +0.472 | -0.338 |
| swinunetr | quarter05 | 2/2 | +0.290 | -0.521 |
| swinunetr | half10 | 2/2 | +0.350 | -0.461 |
| swinunetr | half05 | 2/2 | +0.220 | -0.591 |
| swinunetr | early05 | 2/2 | +0.646 | -0.164 |

The evidence argues against simply increasing the delay: nnUNet late variants still lost on both difficult pairs. Earlier half-strength bands rescued seed 5/fold 1 but not seed 0/fold 2. Swin early half strength improved CA4/DG on average but reduced macro Dice relative to its existing bands recipe. These are development-screen findings, not full nine-pair confirmation.
