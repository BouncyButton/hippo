# SwinUNETR: original contenders versus the current winner

Submitted on 13 October 2026 as **Slurm job 698711**, with an empty account queue and approximately 18.44 GiB of BeeGFS quota free. It is one sequential allocation.

This fixed comparison contains **15 runs**, with no adaptive search or extra weighting variants. Every run uses 5% training data (10 volumes), fold 0, seed 0, the same 52 validation volumes, and image-only SwinUNETR trained from scratch.

The common schedule is AdamW at 0.0001, weight decay 0.00001, no augmentation, and learning-rate halving every 20 epochs. Training has a **75-epoch maximum**, with early stopping after at least 25 epochs (patience 8, minimum improvement 0.0005). The checkpoint with highest mean anterior/posterior validation Dice is evaluated. Epochs 1–5 use Dice only; constraints ramp over epochs 6–10. Initialization and epoch-five weights must match across all runs.

All rows below include the original Dice loss. **Bands in the image rows means the original bands formulation.** Focused bands and endpoint supervision occur only in the winning-constraint control.

| Image row | Added original losses |
|---|---|
| 1 | None: baseline Dice |
| 2 | Bands |
| 3 | Signed distance |
| 4 | Bands + signed distance |
| 5 | Bands + original edge |
| 6 | Bands + signed distance + original edge |
| 7 | Depth CE |
| 8 | Bands + depth CE |
| 9 | Original edge + depth CE |
| 10 | Signed distance + depth CE |
| 11 | Bands + original edge + depth CE |
| 12 | Bands + signed distance + depth CE |
| 13 | Original edge + signed distance + depth CE |
| 14 | Bands + original edge + signed distance + depth CE |
| Additional control | Current winner: focused 3/2 mm outer bands + independent 1/3 mm endpoints + depth CE |

The Dice baseline and winner run first, followed by the other image rows in order. The winner must reproduce **81.5015718459% Dice**, selected at epoch 41 and stopped at epoch 49, before the comparison proceeds. Its existing outer coefficients and depth calibration rule are preserved. Original contenders use the established training-only sequential gradient calibration (10% target per added term, 50% p95 cap), without validation-based coefficient tuning.

Each run records class/foreground Dice, FN, FP and A/P swaps. The selected checkpoint also records directional swaps, outer-boundary distances, A/P-interface distances and paired swap changes on tissue detected by both the candidate and each control. This distinguishes correcting A/P assignments from removing incorrectly classified tissue.

After evaluation, the worker verifies and archives the small configurations, curves, diagnostic results and artifact hashes, then removes the completed checkpoint and prediction archive. Two small reference prediction archives remain only until paired comparisons finish. Active recovery files are retained. At completion the reference predictions and generated geometry caches are removed. Small records remain on the cluster for recovery and are copied into this local report folder when retrieved; bulky files do not accumulate across runs.

This ranks the specified configurations on the shared development fold. It does not establish that the winning loss is best under every possible tuning or on an independent test set.

The source snapshot is `experiments/swin_contenders75_20261013/`. Submission, local checks, startup verification and retrieved results are recorded beside this file.
