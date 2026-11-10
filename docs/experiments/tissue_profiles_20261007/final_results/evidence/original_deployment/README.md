# Dice plus anatomical tissue profiles — fold 0, seed 0

Two new MSD hippocampus experiments train the entire native SwinUNETR from scratch:

| Run | Training cases | Validation cases | Objective |
|---|---:|---:|---|
| `5pct_dice_profile_s0` | 10 of 208 | 52 | Dice + tissue-profile discrepancy |
| `full_dice_profile_s0` | 208 | 52 | Dice + tissue-profile discrepancy |

The completed native Dice-only controls are reused, not retrained. Their best validation macro Dice was 74.1667% (5%) and 87.8514% (full). This is one fold and one seed, an exploratory loss ablation. There is no auxiliary image branch, depth-balanced CE, SDF, focal weighting, topology term or component-count requirement. All native network parameters are trainable.

## What the constraint measures

Use the frozen guide-derived longitudinal plane from the previous view experiment. Training frames came from out-of-fold guides. The same frame is used for prediction probabilities and ground truth; it is never recomputed from the current prediction. Only its centre and orthonormal basis are used, not the stretched UV coordinates or chart boundaries.

Partition the original MRI field of view into 2 mm signed-depth bins (separating the two sides) and 2 mm longitudinal bins. Sum probability × physical voxel volume separately for anterior and posterior tissue. Include all original field-of-view voxels, including background, so missing and extra tissue are visible to the loss; exclude artificial padding. Every source voxel contributes once.

Let `D[c,b,k]` be predicted minus annotated tissue mass for class `c`, signed-depth band `b`, longitudinal position `k`. Let `F` and `R` be its forward and reverse cumulative sums along `k`. The single auxiliary loss is:

```text
L_profile = sum(|F| + |R|) / (2 × K × max(total annotated foreground volume, 1 mm³))
L_total   = L_Dice + ramp × lambda × L_profile
```

`K` is the number of longitudinal bins spanning the original field of view. Normalizing by `K` expresses longitudinal displacement relative to that field's extent. The two cumulative directions remove arbitrary preference for either endpoint. Moving a fixed tissue mass farther along the hippocampus increases its cost. Missing tissue, excess tissue, class swaps, and movement across signed-depth bands are also penalized. No per-band weights or separate prediction/target unit-mass normalizations are applied.

This encodes anatomical location and tissue distribution, not a complete shape or a geodesic distance on an unfolded surface. Different contours can have the same profiles; Dice still provides voxel-level supervision. Whether this improves native contours is an empirical question.

## Fixed training protocol

- MONAI SwinUNETR, feature size 24, native 64³ padded MRI, no pretraining or augmentation; batch size 1; bf16 CUDA.
- Dice baseline includes background, identical to the saved controls; no cross-entropy.
- AdamW `lr=1e-4`, `weight_decay=1e-5`, StepLR every 20 epochs with factor 0.5.
- Maximum 75 epochs; early stopping after at least 25 epochs, patience 8, minimum improvement 0.0005 macro Dice. Select the highest validation macro Dice checkpoint.
- Epochs 1–5 use Dice only. After epoch 5, calibrate one fixed profile coefficient using at most 32 training cases: median profile/Dice logit-gradient RMS ratio 0.1, p95 cap 0.5. Ramp its contribution in epochs 6–10. Validation labels do not tune the coefficient.
- Save optimizer, scheduler, random generators, calibration and epoch history for exact restart. Verify and skip completed runs. Retain each best checkpoint and delete only that run's disposable latest checkpoint after successful completion.

## Evaluation and execution

Compare native-space anterior/posterior macro Dice, foreground Dice, missed tissue, extra tissue, A/P swaps, HD95 and mean surface distance. Save paired case-level differences versus the completed Dice controls and all predictions. Old GT-based depth diagnostics are used for evaluation only.

The single Slurm allocation requests an A100 MIG 4g.40gb GPU, 4 CPUs, 24 GiB host RAM and an 8-hour limit, dependent on running job 687700 ending (`afterany`). It runs 5% first, then full data. Check quota before submission and reserve at least 3 GiB free. Shared frozen atlases are not copied or modified.

Local verification covers physical mass conservation, padding exclusion, displacement sensitivity, missing/extra tissue gradients, class/side separation, reversal invariance, absence of depth-band reweighting, and exact resume before/after calibration. A native 64³ SwinUNETR forward/backward/optimizer step checks full-network gradients. CUDA bf16 checks with real training cases run inside the allocation before either experiment; failures stop the job before subsequent stages.
