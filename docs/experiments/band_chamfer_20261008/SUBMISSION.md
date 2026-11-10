# Submission

Job **690825**: 64 ordinary SwinUNETR experiments, 5%, fold 0, seed 0.

Includes 45 outer-boundary/band/pair experiments and 19 A/P cut experiments. GPU smoke and all-case fixed calibration precede every scientific run. The main A/P tests are prioritized immediately after the shared controls.

Predecessor jobs 690510/690632 finished 20 accepted arms. Their metrics, per-case results, configurations, source and logs were archived and verified locally before deleting the old run directory (1,799,483,268 bytes). The epoch-5 warmup was copied into this job first.

[Experiment matrix](README.md) · [Submission receipt](submission.json) · [Checkpoint provenance](provenance.json)

Allocated GPU verification passed for all 64 objectives, including parameter updates and nonzero Chamfer-to-logit gradients. Calibration passed for all 64 objectives on all 10 training cases. The worker advanced to the fresh Dice + bands baseline.
