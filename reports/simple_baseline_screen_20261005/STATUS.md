A new common cap and a classical baseline are being evaluated.

Provisional cap: 77,506.068011 GFLOPs, anchored to 500 SwinUNETR baseline updates (50 volume-pass epochs). nnU-Net receives 218 updates (21.8 epochs). The strict requested ranges have no common cap: SwinUNETR 50–75 epochs requires 77,506.07–116,259.10 GFLOPs; nnU-Net 10–20 epochs requires 35,529.39–71,058.78 GFLOPs.

Baseline-only development screen: BasicUNet at AdamW learning rates 1e-4 and 1e-3; V-Net at the same rates only if BasicUNet does not qualify. Same ten training volumes, 52 validation volumes, fold 0, seed 0, 64³, batch 1, mild_v1 augmentation, Dice-only loss including background. Both best and final Dice of the strongest tested learning rate must lie between 70% and 80%. Search compute is accounted separately. The cap is not official until empirical evidence supports a final choice.

The existing nnU-Net bands comparison does not substantiate strong benefits in the 10–20 epoch interval; matched differences are mostly negligible or negative for the current fold and seed. SwinUNETR gains occur before 50 epochs. New-budget runs need fresh endpoint verification because changing the horizon changes cosine/poly learning-rate schedules.

CPU FLOP profiles passed full forward/loss/backward/AdamW updates on two synthetic geometries. Cluster adapter preflight passed: two synthetic updates plus validation and idempotent completion resume for both presets, Torch 2.12.1+cu130 and MONAI 1.6.0.

Submitted single sequential Slurm job **684532**. At the first snapshot it was running on gnode04 and starting BasicUNet at 1e-4. No new clinical-data Dice results were available yet. Up to four runs total, one-hour allocation; BasicUNet first, V-Net only if needed. See submission.json, preflight.json and progress_snapshot.json.


BasicUNet completed both learning rates at 344 updates / 34.4 epochs: LR 1e-4 final Dice 45.4665%, best 45.5061%; LR 1e-3 final and best Dice 85.8588%. Both have zero discarded attempts. BasicUNet is rejected at this cap because its stronger tested LR exceeds 80%. The bounded screen is proceeding to V-Net; no intermediate LR is introduced to force a lower Dice.
