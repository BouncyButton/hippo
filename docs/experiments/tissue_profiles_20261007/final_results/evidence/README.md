# Dice plus anatomical tissue profiles — fold 0, seed 0

Thirty-six MSD hippocampus experiments train the entire SwinUNETR from scratch. The first twelve experiments queued in job 688228 are retained in their original order. Preserve the next ten planned input controls, then append fourteen missing cells. Together with six completed Dice controls reused from earlier jobs, this covers 42 architecture/loss/fraction cells. The four original native-only tests are:

| Run | Training cases | Validation cases | Objective |
|---|---:|---:|---|
| `5pct_dice_profile_s0` | 10 of 208 | 52 | Dice + tissue-profile discrepancy |
| `5pct_dice_multiscale_s0` | 10 of 208 | 52 | Dice + multiscale transverse-strip profiles |
| `full_dice_profile_s0` | 208 | 52 | Dice + tissue-profile discrepancy |
| `full_dice_multiscale_s0` | 208 | 52 | Dice + multiscale transverse-strip profiles |

The original twelve-run block includes eight branch runs: both losses × both data fractions × the matched plane and unwrapped input branches. Each cell below is a separate run:

| Input architecture | 5% + original profiles | 5% + finer profiles | Full + original profiles | Full + finer profiles |
|---|---|---|---|---|
| Native MRI only | Yes | Yes | Yes | Yes |
| Native + plane branch | Yes | Yes | Yes | Yes |
| Native + unwrapped branch | Yes | Yes | Yes | Yes |

Use the previously selected `plane_position`/`unfold_position` pair at 5%, and `plane_attention`/`unfold_attention` at full data. These are the existing frozen branch implementations and atlas inputs, with ordinary feature fusion and no entropy gates. All native, branch and fusion parameters train jointly. The loss always compares the final native-grid logits using the same anatomical coordinates: the view branch changes the input features, not the numerical loss correspondence.

The completed native Dice-only controls are reused, not retrained. Their best validation macro Dice was 74.1667% (5%) and 87.8514% (full). Reuse the four matched Dice-only branch controls from predecessor job 687700, verifying their artifacts and case splits before training. This is one fold and one seed, an exploratory loss ablation. There is no depth-balanced CE, SDF, focal weighting, topology term or component-count requirement.

## Appended native-input controls

After all twelve runs above, run the following six controls, in table order, first at 5% and then at full data. All use fold 0 and seed 0:

| Native informed SwinUNETR | 5% | Full |
|---|---|---|
| Dice only | `5pct_dice_s0_informed` | `full_dice_s0_informed` |
| Dice + pooled profile | `5pct_dice_profile_s0_informed` | `full_dice_profile_s0_informed` |
| Dice + multiscale profile | `5pct_dice_multiscale_s0_informed` | `full_dice_multiscale_s0_informed` |

Feed four aligned native-grid channels into a single SwinUNETR: the original MRI, longitudinal position divided by 32 mm, signed cutting-plane depth divided by 16 mm, and ROI support. Use the same frozen guide-derived center, basis and source ROI indices as the view branches. The ROI is guide foreground plus 8 mm context, intersected with the original MRI field of view; it includes background. Zero only the extra channels outside that ROI. Keep the full original MRI unchanged. No chart grids, UVs, extra encoder, feature fusion, or annotations enter these inputs.

Preserve all seed-matched original SwinUNETR initial weights. Expand its three input convolutions from one to four channels, copy the MRI weights, and initialize the three new channel weights to zero. All weights, including those new channels, train from the first step. This starts with the same MRI-only predictor while allowing anatomical input immediately during learning. The Dice-only variant never creates a profile loss or calibrates a constraint. Both constrained variants use precisely the original loss definitions below.

## Appended MRI-only view controls

Runs 19–22 are four Dice-only controls: `5pct_dice_s0_plane_mri_context`, `5pct_dice_s0_unfold_mri_context`, `full_dice_s0_plane_mri_attention`, and `full_dice_s0_unfold_mri_attention`. Their auxiliary convolutions receive exactly one channel: the sampled MRI intensity. They receive no longitudinal, depth or support channel. Retain the original context architecture at 5% and attention architecture at full data, matching the existing Dice-only branch controls. Preserve their initial weights, slicing only the two auxiliary input kernels to the MRI channel; the full native network, encoder and fusion remain trainable.

Valid-sample masks are still required for masked global pooling, and frozen return indices/support still perform native-space mapping and the unchanged fusion. Those operations are not learned auxiliary input channels. Thus MRI-only refers to the auxiliary encoder's learned input, not removal of the guide-derived geometry or all geometric bookkeeping. The original MRI remains the native branch input.

## Fourteen final additions and complete comparison matrix

Append eight constrained MRI-only alternative-view runs at positions 23–30: for 5% then full data, plane then unwrapped, pooled then multiscale. Append six native-view MRI-only auxiliary runs at positions 31–36: for 5% then full data, Dice only, pooled, then multiscale. Do not insert them into or reorder the first 22 runs.

Each cell in this table is evaluated at both 5% and full data, fold 0, seed 0:

| Architecture and inputs | Dice only | Dice + pooled | Dice + multiscale |
|---|---|---|---|
| Ordinary native SwinUNETR, MRI | Reuse completed | Original queue | Original queue |
| Directly informed native SwinUNETR | Added | Added | Added |
| Native MRI + position-enabled plane branch | Reuse completed | Original queue | Original queue |
| Native MRI + position-enabled unwrapped branch | Reuse completed | Original queue | Original queue |
| Native MRI + MRI-only plane branch | Added | Final additions | Final additions |
| Native MRI + MRI-only unwrapped branch | Added | Final additions | Final additions |
| Native MRI + MRI-only native-view branch | Final additions | Final additions | Final additions |

The native-view auxiliary receives original MRI samples on the original voxel lattice. It covers exactly the same frozen guide-defined hippocampus-plus-8-mm ROI, including background. Preserve each existing chart's source membership and separate context-pooling group; within each chart, retain the original axis orientation and voxel positions. Crop to its native bounding box, add an empty voxel margin and pad dimensions to multiples of eight. Empty cells have zero MRI and zero support. Mapping returns features to the original voxel centers and source indices. No annotations construct the channels, ROI or chart membership.

This choice matches chart seams and context groups across controls rather than giving the native control extra cross-chart context. The union of its charts is exactly the full shared ROI, without duplicate or missing source voxels. Its MRI-only encoder, context pathway, attention (at full data), fusion and trainable parameter count match the corresponding plane/unwrapped controls exactly. Tests compare every initial state tensor, not only the total parameter count. The full SwinUNETR and auxiliary branch train together; every constraint applies to the final native-grid logits.

Necessary support masks are retained consistently for global context pooling, sparse return mapping and the unchanged fusion. The new native grids have no MRI resampling; source intensities are quantized to float16, matching the stored precision of alternative-view inputs. Alternative views use normalized trilinear splatting and return interpolation. Grid orientation, effective physical kernel size, tensor shape, padding/empty-cell layout and raster support therefore still differ. A difference between these models measures a view-processing pipeline, not a mathematically isolated coordinate rotation.

The report produces these matched comparisons for every available loss and fraction:

| Comparison | Main question |
|---|---|
| Native-view MRI auxiliary − ordinary SwinUNETR | Does another processing branch help? |
| Plane/unwrapped MRI auxiliary − native-view MRI auxiliary | Does changing the auxiliary view add value at matched capacity? |
| Position-enabled − MRI-only auxiliary, same view | Do explicit anatomical input channels add value? |
| Directly informed SwinUNETR versus every auxiliary architecture | Is separate processing and fusion useful beyond direct information? |
| Pooled/multiscale − own architecture's Dice-only control | Does anatomical supervision help? |
| Multiscale − pooled, same architecture | Does preserving transverse detail help? |

Report anterior/posterior macro and class Dice, foreground Dice, missed/extra tissue, A/P swaps, mean surface distance and HD95. Paired case-level differences preserve undefined surface metrics as missing, not zero. Every result row also reports parameter count and training runtime. New runtime is the sum of epoch training/calibration/validation times; it excludes checkpoint-write time and final detailed evaluation. Legacy controls use their saved elapsed counters, with approximate handling of process restarts. Therefore runtime comparisons are descriptive and should be benchmarked under matched execution conditions before efficiency claims.

These are preliminary single-seed findings on one split. Before claiming a reliable advantage, replicate the native-auxiliary versus ordinary baseline, best alternative view versus native-auxiliary, position-enabled versus MRI-only in that view, and the winning constraint versus its own Dice control with seeds 1 and 2. Preserve paired splits and settings and replicate both fractions. Rank candidates by paired improvements in Dice and errors together with contour metrics, not a single best validation number. Report across-seed variation, and confirm choices on held-out data before generalization claims. These replication runs are recommendations, not additional submissions in this job.

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

Replace held, never-started job 688228 with a new Slurm job behind 687700 (`afterany`, if still active). Preserve its original deployment and submission receipts. Keep the A100 MIG 4g.40gb GPU, 4 CPUs, 24 GiB host RAM and 23-hour-50-minute allocation limit. Preserve the first 22 runs, then append the fourteen final additions in the order above. Reserve at least 6 GiB free and verify quota before resubmission. Shared frozen atlases are not copied or modified. The per-run 75-epoch budget and early stopping are unchanged.

Local verification covers physical mass conservation, padding exclusion, displacement sensitivity, cancellation, missing/extra tissue gradients, class/side separation, reversal invariance, absence of depth-band reweighting, and exact resume before/after calibration. Native 64³ SwinUNETR forward/backward/optimizer checks cover both losses. Eight real-atlas CPU checks cover every branch/loss combination, including nonzero gradients through native, auxiliary encoder and fusion parameters. CUDA bf16 checks use the largest frozen view charts and verify all training cases' loss mass coverage inside the allocation before each run; failures stop the job before subsequent stages.

New checks verify signed physical coordinates under anisotropic rotated affines, exact ROI membership and padding exclusion, input independence from labels and chart fields, preservation of baseline initialization, gradients for all four native input channels, Dice-only absence of calibration, and exact restart for all informed objectives and MRI-only branches. MRI-only encoder output must be unchanged by arbitrary changes to the discarded position channels. Native-view checks verify sample identity, chart membership, exact matching of initialized parameters and valid return mapping. Fresh CPU forward/backward/optimizer checks cover eighteen MRI-only view/loss combinations, three informed objectives, and the two ordinary native losses. CUDA preflight is repeated inside the allocation before every run.
