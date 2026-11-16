# Full-data learning-rate comparison

Each run uses one full training fold, its original held-out cases and its prespecified final update. All three seeds and folds remain in scope, irrespective of scores. This is a development comparison: against the previous full-fold run it isolates the learning-rate change; against the old low-data baseline both the amount of data and learning rate differ.

| Model | Test |
|---|---|
| Basic U-Net and UNETR | Peak 0.001. Reuse the 54 archived MSD, ADNI and HFH runs; run the 18 MNI cells. |
| nnU-Net | Scale the archived per-update learning rates by 2 on MSD, ADNI and MNI; by 1.25 on HFH, where the original peak is already 0.1. |
| SwinUNETR | Same multipliers as nnU-Net, using SwinUNETR's own archived curve. |
| MedSAM3 | Double each adapter learning rate; peak 0.00005 → 0.0001, scheduled floor 0.000005 → 0.00001. Keep 20 warmup updates and the original decay horizon; HFH stops at update 125 on its 200-update curve. |

The plan lists 180 cells: 54 existing and 126 new. MSD/ADNI nnU-Net and SwinUNETR retain their authorized one-pass budget exception and native resolution. The 36 MedSAM3 runs keep the original two adjacent-slice slabs per update and the original holdouts. Evaluation of MSD retains both all 52 official validation crops and the same historical subset for paired comparisons. The campaign has separate output directories, result filenames and completion receipts.

Colab assignment: MSD notebook 18 MSD scratch + 3 MSD fine-tunes on fold 0; ADNI notebook 18 ADNI scratch + 9 ADNI fine-tunes + 3 MSD fine-tunes on fold 1; HFH notebook 18 HFH scratch + 3 MSD fine-tunes on fold 3. The cluster handles the remaining 45 MNI cells and 9 HFH fine-tunes. A completed result requires validation, a checked per-update trace, full crop exposure, an archive and a storage receipt. Cluster admission waits for available queue and quota headroom. A staged job is not a running job.

**Sampling opportunity, separate from the LR comparison.** An offline replay of nine MNI schedules measured 27.28 different training planes per crop on average (48 available). Twenty distinct nonoverlapping adjacent pairs per crop would present exactly 40 planes per crop with the same 800 draws and 400 optimizer updates. No Dice gain follows from this arithmetic. Compare the two samplers at a fixed LR to test its accuracy impact; do not select it on the outer validation folds.
