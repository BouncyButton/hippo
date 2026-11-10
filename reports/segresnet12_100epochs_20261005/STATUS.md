Single SegResNet-12 Dice-only baseline requested by the user on 2026-10-05.

Fold 0, seed 0; ten training volumes, 52 validation volumes, 64³ inputs, batch size 1, same volume-disjoint split and mild_v1 augmentation. No auxiliary losses or weight calibration.

Exact cap: 62,166,546,655,000 FLOPs (62,166.546655 GFLOPs), with 62,166,546,655 FLOPs per update; 1,000 updates = 100 volume-pass epochs. No early stopping. Both best and final validation macro foreground hard Dice will be reported. Validation every 50 updates / five epochs.

Model: MONAI 1.6.0 SegResNet, 12 initial filters, four GroupNorm groups, deconvolution upsampling, 2,669,271 parameters. No autoencoder branch or deep supervision. Fresh synthetic FLOP profiles on two geometries reproduce the agreed cost exactly.

Optimizer: AdamW1e-3, weight decay1e-5, 50-update linear warmup, cosine decay to1e-5 across the allocated 1,000 updates. This is a single configuration, chosen before observing its validation Dice; no adjustment to force a target score.

The training adapter reuses the generic trainer’s internal unet branch for Dice+AdamW. The actual architecture is separately bound to segresnet12 in protocol.json and runtime_recipe.json, checked against its parameter count and profiled module preset, and named segresnet12 in the run directory and final summary.

Cluster synthetic smoke check passed: two updates, finite validation and idempotent completion resume, using Torch 2.12.1+cu130 and MONAI 1.6.0.

Single Slurm job **684566** submitted. First verified snapshot: RUNNING on gnode02, starting the authorized 1,000-update baseline; stderr empty. See submission.json and preflight.json. No SwinUNETR or nnU-Net runs are part of this submission.


Job **684566 completed successfully (exit0:0)** in3m13s. All1,000 updates were applied without discarded attempts, consuming exactly62,166.546655GFLOPs. Final Dice85.95294%; best85.96393% at950updates/epoch95. Final anterior Dice87.00461%, posterior84.90128%, evaluated on52volumes.

Observed checkpoints in this100-epoch trajectory: epoch15 Dice69.80633%, epoch20 Dice76.80058%, epoch25 Dice81.63434%. Thus this configuration exceeds the70–80% target at the agreed cap. These checkpoints are not endpoint results from separately trained shorter cosine schedules. No additional LR, model or constraint runs were launched.

The complete metrics curve and JSON run metadata are retained in completed_metadata/run/. Downloaded config, metrics, budget-ledger, completion and best-validation hashes match the cluster completion record. Raw images and model checkpoints were excluded from the local metadata bundle.
