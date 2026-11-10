# MSD seed 5: constrained SwinUNETR and U-Net

User corrected the initial seed 4 request to seed 5 and added U-Net. Twelve new runs: two models × bands/bands+edge × folds 0/1/2. Two isolated GPU workers in one allocation. Pending SAM3 job 706255 was cancelled and its campaign paused at the user’s request.

Compare against the six already completed, hash-verified seed 5 baselines. Preserve identical initializations, frozen folds (10 training / 52 validation volumes), Dice including background, AdamW, learning-rate horizon, batch size 1, BF16 and no augmentation. The archived scientific trainer source is unchanged.

Original sequential 10% logit-gradient target and 50% p95 ceiling; calibrate on the ten training cases at update 50, then ramp over 50 updates. The optimization-only FLOP cap is 31,976,450,667,090 per run; calibration is measured and reported separately, as in the original experiment. SwinUNETR runs 206 updates for either arm; U-Net runs 563 for bands and 562 for bands+edge because of the added compute cost.

Report final-budget validation hard Dice and paired fold differences versus baseline, plus FN/FP and A/P swaps. Best-checkpoint Dice is secondary. All folds and unfavorable results remain in the comparison.

Storage estimate: up to 8 GiB logical / 16 GiB mirrored, plus 4 GiB reserve; require 20 GiB free. Evaluate and archive every completed run outside BeeGFS, verify hashes, then delete its generated checkpoints. Preserve interrupted recovery files. The collector must stay connected; after 300 seconds without an archive acknowledgment, the worker stops further launches.

The archived seed 5 baseline means are SwinUNETR 40.24% and U-Net 78.76%. Generate paired fold-level differences with `python3 -m scripts.cross_dataset_campaign.report_swin_unet_seed5_constraints --destination reports/msd_swin_unet_seed5_constraints_20261018`.

Submitted as **706344**, dependent on the running seed7–9 baseline job **706250**, at 2026-10-18T12:29:56.072445+02:00. All 12 launch commands, both archived model initializations, dataset hash, calibrated finite gradients and zero-scale baseline equivalence passed preflight. The two-worker safety suite passed six tests. Quota at submission: 28.81/93.13 GiB used, 64.32 GiB free. Verified archival and per-run cleanup are enabled.
