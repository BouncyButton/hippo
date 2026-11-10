# MSD shared-seed exploration

Authorized seeds 10–41, one batch allocation, two isolated GPU workers, maximum 24 hours. Stop at the first shared seed with final-budget validation Dice >= 50% for each of nnU-Net, SwinUNETR, UNETR and U-Net on each of folds 0–2 (12 checks).

Each candidate starts with paired half-budget fold-0 screens for UNETR and SwinUNETR. The respective rejection gates are 28% and 39%, about five percentage points below the historical minimum halfway Dice among runs finishing at >=50%. These are heuristic filters and can miss late improvers. Passing candidates receive fresh full-budget fold-0 runs, then the remaining model/fold pairs. Any final-budget result below 50% rejects that seed. Both workers run independent experiments concurrently.

Full runs preserve the previous no-augmentation protocol, fixed data splits, losses, optimizers, learning-rate horizons and 31,976,450,667,090 optimization-FLOP cap. Screening is a separate experiment with half this cap; screening costs are recorded separately. No claim of resuming half-budget checkpoints into full-budget runs is made. At most 448 possible runs (384 full and 64 screens), with adaptive early rejection and stopping; the 24-hour limit may prevent testing all 32 seeds. There is no automatic successor.

This is exploratory selection on existing validation folds, not an unbiased held-out estimate. All completed candidates, including rejected ones, are retained as compact results. Each completed run is evaluated and hash-verified in this local durable archive before its bulky remote artifacts are removed. Interrupted artifacts remain for recovery. The collector runs locally and keeps the computer awake; loss of collector connectivity causes the batch to stop after its bounded archive wait.

The frozen source and protocols are under `frozen/`. `live.json` records the job status, decisions and any qualifying seed. `runs/` contains full metrics, curves, diagnostics, protocols and cleanup receipts.

Validation before submission: 11 scheduler, threshold, provenance and dispatch tests passed; remote preflight checks every allowed command and all four model builders without optimizer updates.
