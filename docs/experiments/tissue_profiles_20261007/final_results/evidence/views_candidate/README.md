# Dice plus anatomical tissue profiles — fold 0, seed 0

Twelve MSD hippocampus experiments train the entire SwinUNETR from scratch. Job 688228 was still pending when the user requested finer profiles and then both plane and unwrapped branches. The four native-only tests are retained:

| Run | Training cases | Validation cases | Objective |
|---|---:|---:|---|
| `5pct_dice_profile_s0` | 10 of 208 | 52 | Dice + tissue-profile discrepancy |
| `5pct_dice_multiscale_s0` | 10 of 208 | 52 | Dice + multiscale transverse-strip profiles |
| `full_dice_profile_s0` | 208 | 52 | Dice + tissue-profile discrepancy |
| `full_dice_multiscale_s0` | 208 | 52 | Dice + multiscale transverse-strip profiles |

Add eight runs: both losses × both data fractions × the matched plane and unwrapped input branches. Thus each cell below is a separate run:

| Input architecture | 5% + original profiles | 5% + finer profiles | Full + original profiles | Full + finer profiles |
|---|---|---|---|---|
| Native MRI only | Yes | Yes | Yes | Yes |
| Native + plane branch | Yes | Yes | Yes | Yes |
| Native + unwrapped branch | Yes | Yes | Yes | Yes |

Use the previously selected `plane_position`/`unfold_position` pair at 5%, and `plane_attention`/`unfold_attention` at full data. These are the existing frozen branch implementations and atlas inputs, with ordinary feature fusion and no entropy gates. All native, branch and fusion parameters train jointly. The loss always compares the final native-grid logits using the same anatomical coordinates: the view branch changes the input features, not the numerical loss correspondence.

The completed native Dice-only controls are reused, not retrained. Their best validation macro Dice was 74.1667% (5%) and 87.8514% (full). Reuse the four matched Dice-only branch controls from predecessor job 687700, verifying their artifacts and case splits before training. This is one fold and one seed, an exploratory loss ablation. There is no depth-balanced CE, SDF, focal weighting, topology term or component-count requirement.

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

## Added multiscale variant

Retain the transverse coordinate that the original profiles pool away. Within the same signed 2 mm depth layers, build separate longitudinal profiles for each class and transverse strip at two scales:

| Scale | Transverse strip width | Longitudinal bin width | Depth width |
|---|---:|---:|---:|
| Coarse | 4 mm | 4 mm | 2 mm |
| Fine | 2 mm | 2 mm | 2 mm |

At each scale, use the original symmetric forward/reverse cumulative formula, summed over all depth/strip/class rows. Normalize once by the total annotated foreground physical volume and the longitudinal extent, never by individual strip mass or occupancy. Average the two scale losses equally and calibrate that combined term as one auxiliary loss. This is not depth balancing or strip balancing.

Both scales use the same source voxel probabilities and guide-derived frame. Every native voxel contributes exactly once per scale, with its physical voxel volume. No rendering, UV projection, interpolation, or changing prediction-derived geometry enters the correspondence.

Before training, construct ten missing voxels on one transverse edge and ten extra voxels at the same longitudinal/depth positions on the opposite edge. The original loss must be exactly zero and the new loss positive. A second example must cancel at the coarse scale but be detected at the fine scale. Test gradient signs at the missing and extra locations, and explicitly confirm the remaining ambiguity for two different voxels within one finest cell. Dice retains voxel-level supervision; small-component preservation is not guaranteed.

## Fixed training protocol

- MONAI SwinUNETR, feature size 24, native 64³ padded MRI, no pretraining or augmentation; batch size 1; bf16 CUDA.
- Dice baseline includes background, identical to the saved controls; no cross-entropy.
- AdamW `lr=1e-4`, `weight_decay=1e-5`, StepLR every 20 epochs with factor 0.5.
- Maximum 75 epochs; early stopping after at least 25 epochs, patience 8, minimum improvement 0.0005 macro Dice. Select the highest validation macro Dice checkpoint.
- Epochs 1–5 use Dice only. After epoch 5, calibrate one fixed profile coefficient per run using at most 32 training cases: median profile/Dice logit-gradient RMS ratio 0.1, p95 cap 0.5. Ramp its contribution in epochs 6–10. Validation labels do not tune the coefficient. The multiscale coefficient applies to the equal mean of the two scales.
- Save optimizer, scheduler, random generators, calibration and epoch history for exact restart. Verify and skip completed runs. Retain each best checkpoint and delete only that run's disposable latest checkpoint after successful completion.

## Evaluation and execution

Compare native-space anterior/posterior macro Dice, foreground Dice, missed tissue, extra tissue, A/P swaps, HD95 and mean surface distance. Save paired case-level differences versus the architecture-matched completed Dice controls, finer versus original profiles within each architecture, unwrapped versus plane at each loss, and all predictions. Old GT-based depth diagnostics are used for evaluation only.

The same Slurm allocation 688228 requests an A100 MIG 4g.40gb GPU, 4 CPUs, 24 GiB host RAM and a 23-hour-50-minute limit for the expanded matrix, dependent on job 687700 ending (`afterany`). It runs six 5% tests, then six full-data tests. Each prior deployment is archived before updating the held pending job, and its dependency is retained. Check quota before release and reserve at least 4 GiB free. Shared frozen atlases are not copied or modified.

Local verification covers physical mass conservation, padding exclusion, displacement sensitivity, cancellation, missing/extra tissue gradients, class/side separation, reversal invariance, absence of depth-band reweighting, and exact resume before/after calibration. Native 64³ SwinUNETR forward/backward/optimizer checks cover both losses. Eight real-atlas CPU checks cover every branch/loss combination, including nonzero gradients through native, auxiliary encoder and fusion parameters. CUDA bf16 checks use the largest frozen view charts and verify all training cases' loss mass coverage inside the allocation before each run; failures stop the job before subsequent stages.
