# UNet bands timing and strength pilot

Fold 1, seeds 0 and 5. Original bands definition and Dice-only AdamW LR 0.00016 recipe unchanged. Optimization cap: 53,305,109,592,450 FLOPs. Calibration is accounted separately (382,099,497,372 FLOPs per constrained run), following the established convention.

Early onset is update 76; delayed onset is update 470 (approximately 50% of the budget). Both use a 76-update ramp. Strength is the calibrated constraint-to-supervised gradient target, 10% or 5%, not the raw loss coefficient. All constrained cells completed 938 updates; archived baselines completed 939 under the same cap.

| Recipe | Seed 5 Dice (%) | Change vs baseline (pp) | Seed 0 Dice (%) | Change vs baseline (pp) |
|---|---:|---:|---:|---:|
| Baseline | 62.62 | — | 57.71 | — |
| Original onset, 10% | 53.53 | -9.08 | 67.19 | +9.48 |
| Original onset, 5% | 58.38 | -4.24 | 65.80 | +8.09 |
| Delayed, 10% | 64.31 | +1.70 | 66.87 | +9.16 |
| Delayed, 5% | 64.32 | +1.70 | 64.46 | +6.75 |

Both delayed variants outperform the matched baseline in both development pairs. Halving strength alone reduces the seed 5 regression but does not remove it. This supports an early-training interaction as the practical failure mechanism; it does not establish the exact parameter-gradient mechanism.

For seed 5, CA4/DG Dice recovers from 17.79% with original bands to 49.63% with delayed 10% and 51.15% with delayed 5% (baseline 49.02%).

Delayed 10% is the stronger candidate for wider confirmation: essentially the same seed 5 macro Dice as delayed 5%, with 2.41 points more on seed 0. Delayed 5% has the better seed 5 CA4/DG score. Neither recipe is established across all folds and seeds. No automatic full-matrix promotion was made.

Verification: six new run archives verified before checkpoint cleanup; four exact archived controls reused. Matched timing pairs share identical calibration anchors, and 5% weights are exactly half the 10% weights. All fresh runs completed without discarded charged updates. Detailed per-class Dice, FN/FP, swap diagnostics, configurations, curves, calibration, logs and archive digests are preserved in this report directory.

Full matrix restored unchanged: job 749245 running on two GPUs; continuation 749248 queued. Both interrupted runs advanced beyond saved updates 123/376, with logged successful updates equal to charged attempts. Collector lock verified active. Resume evidence: matrix_resume_audit.json. Remaining quota at final check: 41.11 GiB.
