# Distinct cases versus optimizer budget

Run only after the 50-case seed0 summed control completes successfully. Change the unique training set to the original nested10 cases, repeated five times per shuffled50-update pass. Use the same seed, initial model, AdamW, AMP, frozen source,250-update auxiliary warmup, bands/edge coefficients, validation cohort and checkpoint criterion. Replay every actual learning rate from the50-case summed control at the corresponding update. Stop with the same patience8/min60 policy, capped at the completed50-case control's update budget. Compare fixed15/30/45/60 passes and selected checkpoints; do not claim additional epochs alone explain differences.

Training audits evaluate each of the10 unique cases once; validation audits evaluate the same52 cases. The intervention changes training diversity and repeated exposure per subject, which are inseparable at a fixed total update count. This cannot tell which of the40 added subjects matters, or establish an annotation-noise floor.

Preserve all original studies and all MedSAM3 artifacts. Estimated new weights~135MiB plus one temporary resumable checkpoint~202MiB. Require1.1GiB free at allocation. Every new meaningful run uses early stopping.
