# SAM3 five-volume update and gentle-edge sweep

One GeoFWI3D fold-0, seed-0 Slurm allocation, capped at 24 hours, with two isolated `4g.40gb` GPU workers (the `stud` QoS limit). All runs use the original frozen five training volumes and the same 96-volume validation cohort, pretrained SAM3 initialization, native SAM3 loss, LoRA adapters, no augmentation, and a fixed 200-update learning-rate horizon. Every run is evaluated at its stopping update and archived outside BeeGFS before generated checkpoints are deleted.

The 24 runs compare:

- Native-loss control and bands at a 20% gradient target after 10, 25, 50, 100, and 200 optimizer updates.
- Bands at 10% and 40% after 50 and 100 updates.
- Bands at 20% plus gentle edge targets of 0.5%, 1%, 2%, and 5% after 50 and 100 updates; 1% and 2% edge also after 200 updates.

The gradient targets are calibrated against SAM3's native fine-tuning loss, which already contains Dice. Each auxiliary has its own 95th-percentile ceiling at five times its target. The matched native and bands-only controls at each stopping point make it possible to identify whether edge adds a small, repeatable benefit or causes broad or severe case-level losses. The primary comparison is mean validation Dice, with paired validation-volume wins/losses, lower-tail Dice changes, foreground FN/FP, training Dice, realized gradient ratios, and clipping as diagnostics. This single-fold/seed screen is exploratory.

The prior four-volume sweep favored bands alone at the 20% target (86.29% versus 85.21% native validation Dice). All tested 10–40% bands+edge settings were worse, motivating the much smaller edge targets here. The fixed learning-rate horizon makes the shorter runs stopping-point comparisons along the same initial update schedule.

Submitted as Slurm job **703589** on 16 October 2026. The allocation started with two distinct GPU workers. Submission quota was 25.29/93.13 GiB used (67.84 GiB free). The cluster preflight passed. See `submission.json`, `preflight.json`, and `live.json` for the receipts and recorded progress.

## Recovery status (18 October 2026)

Slurm recorded job 703589 as `COMPLETED` with exit 0 after 50m10s, but the campaign stopped at `waiting_for_archive_offline` when the local collector did not acknowledge the first two completed runs within five minutes. Only the 10-update native and bands20 variants ran; 22 variants remain unrun. The collector process was no longer alive at the status audit. The two completed evaluations were subsequently archived locally and hash-verified; their two generated checkpoints were then deleted from BeeGFS, releasing 446,227,434 bytes. No job is now queued or running. Current personal quota is 25.29/93.13 GiB used (67.84 GiB free).

At 10 updates, native SAM3 reached 31.575% mean validation Dice and bands20 reached 31.483% on the same 96 volumes, a -0.092 percentage-point change; bands20 improved 40 volumes and worsened 56. These two short stopping-point results do not answer whether gentler edge terms help. The verified metrics and paired case changes are in `comparison.json`.
