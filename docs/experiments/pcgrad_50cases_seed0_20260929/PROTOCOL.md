# Seed 0 PCGrad with 50 training cases

Exactly one new model: fold 0, seed 0, PCGrad only. Start from the same random initialization seed, without fine-tuning the previous selected model. No control arm, additional seed, or fold launches automatically.

## Cohort and comparison

Retain the original ten training cases and add 40 sampled without replacement from the remaining 198 original fold-0 training candidates. Use Python random.Random(0) on the sorted candidate IDs; save the explicit final 50-case list. Preserve the original 52 validation cases and deterministic preprocessing. Verify input, frozen source, and reference checkpoint hashes before training.

Keep the original bands and edge coefficients and symmetric five-task PCGrad unchanged. Do not recalibrate weights. Preserve the model, batch size of one, AdamW settings, no augmentation, five-epoch auxiliary warmup, precision, projection RNG, and checkpoint selection criterion.

## Learning rate and early stopping

Initial learning rate: 1e-4; weight decay: 1e-5. Replace StepLR with ReduceLROnPlateau on validation hard macro Dice: maximize, absolute improvement threshold 0.0005, patience 3 (reduce after four non-improving checks), reduction factor 0.5, minimum LR 1e-6. The copied legacy step_size/gamma fields are inactive; lr_policy records the active schedule.

Maximum 75 epochs; earliest patience stop at epoch 60, patience 8, minimum delta 0.0005. This preserves the original early-stopping policy. Select the greatest validation hard macro Dice. Retain both selected inference weights and the final complete optimizer, scheduler, AMP, and RNG state for any later explicitly authorized continuation.

With batch size one, there are 50 rather than 10 updates per epoch: at most 3,750 rather than 750 updates. The five-epoch warmup spans 250 updates. This jointly tests more data, more updates, and an adaptive LR; it cannot isolate a sample-count effect. Historical ten-case runs are context, not a matched 50-case control.

## Measurements

Save every update's projection diagnostics and every epoch's losses, LR, validation Dice, and cumulative update count. Retain the same two original training cases for FP32 gradient probes. At epochs 5, 15, 30, 45, 60, 75 and the final stop, save per-case hard/soft edge, Dice and connectivity audits for both the 50 training and 52 validation cases. Audit the selected checkpoint on both sets too. Preserve case identities to distinguish the original ten from the additional forty.

## Resources and integrity

Use one stud GPU allocation (3g.40gb), four CPUs, 24 GiB RAM, and a 90-minute limit. Require 1.6 GiB free before submission and allocation. No prior experiment files or checkpoints are modified or deleted. Frozen-source checks and tests cover projection behavior, the LR policy, and exact interruption/resume across an LR reduction. No follow-up experiment is automatic.
