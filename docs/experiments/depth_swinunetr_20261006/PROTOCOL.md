# Depth-conditioned SwinUNETR experiment

User-authorized 6 October 2026: exactly four objectives at each of two training-data fractions, MSD fold 0, seed 0, from scratch, maximum 75 epochs with early stopping. No other constraints are included. This is an epoch-matched exploratory comparison, not a compute-matched or independent-test study.

## Runs and shared training

| Arm | Training loss |
| --- | --- |
| `dice` | Original MONAI softmax Dice, including background |
| `depth` | Dice + depth-balanced multiclass CE |
| `depth_sdf3d` | Dice + the same depth CE + depth-balanced classwise 3D signed-distance term |
| `depth_sdf_layer` | Dice + the same depth CE + depth-balanced classwise cross-sectional signed-distance term |

The full fold contains 208 training cases. The 5% arm reuses the established ten-case subset (10/208 = 4.81%, the historical rounded-down 5% setting); it does not resample cases independently between objectives. Both settings retain the identical 52 validation cases. Exact IDs and input hashes are in `protocol.json`.

All eight runs use MONAI 1.6.0 SwinUNETR, feature size 24, 15,703,029 parameters, three labels, 64-cubed inputs, batch one, no resizing, no dropout, no activation checkpointing, and no pretrained weights. Preserve the historical nonzero-intensity normalization and central padding/cropping. No training augmentation is introduced in this experiment. AdamW uses LR 1e-4, weight decay 1e-5, and StepLR every 20 epochs with factor 0.5. CUDA bfloat16 autocast uses FP32 losses and optimizer state, without skipped GradScaler updates.

Validation runs every epoch. Early stopping uses patience 8, minimum 25 epochs, and a significant improvement greater than 0.0005. The best checkpoint records every strict improvement in mean per-patient anterior/posterior hard Dice, even below the stopping threshold. All runs are capped at 75 actual passes over their selected training cases; full-data and low-data epochs therefore contain different numbers of updates.

## Geometry and objectives

Compute the physical foreground centroid and longest PCA direction from the union of the training annotation labels. The plane contains that direction; its normal is anatomical RAS right projected orthogonally to it. Preserve the sign of plane distance, separating both sides into 2-mm bands. The cut plane does not assert a planar A/P interface or a fixed number of tissue components.

The auxiliary support includes annotated foreground and background within 8 mm of it. Ordinary full-volume Dice still penalizes errors outside that support. Each sufficiently populated signed-depth band gets equal weight. Bands with fewer than 64 support voxels have mass N/64, limiting tiny-band amplification. Normalize the resulting voxel weights to sum to one per patient. The depth term is the weighted mean of ordinary three-class CE; no full-volume CE is added to the baseline.

For both SDF arms, use two fields, one for each foreground class, with distances negative inside that class and positive outside, half-voxel offset, clipping at +/-8 mm and division by 8. The field-only difference is:

- 3D: original physical Euclidean classwise EDT, matching the repository SDF convention on the verified 1-mm MSD grids.
- Layer: classwise 2D EDT on physical 1-mm cross-sections parallel to the plane. These thin cross-sections are distinct from the 2-mm bands used for loss aggregation. Rasterize labels with nearest-neighbor sampling, interpolate distance magnitudes back to original voxel centers, and set signs from the original labels so resampling cannot reverse the target. Empty class sections get the constant positive clipped distance. Record rasterization sign disagreement before correction.

Both use the same mean across A/P of `(p_class - y_class) * phi_class`, with identical depth voxel weights and support. Consequently this comparison isolates the distance field; the 3D arm changes the historical whole-volume SDF reduction to the agreed depth-balanced reduction. The new fields are discrete cross-sectional distance estimates, not anatomical thickness or screen-space distances. No mesh extraction, argmax, or predicted geometry occurs in the differentiable loss path.

The classwise SDF is a spatially weighted probability loss. Moving probability directly from a wrong to the correct class reduces it, but softmax coupling to the third class means an SDF-only gradient need not improve every individual true-versus-predicted logit margin. Calibration records this diagnostic; there is no claim of guaranteed error correction.

## Training-only coefficient calibration

Every run begins with five Dice-only epochs, included in its 75-epoch cap. At the start of epoch six, each auxiliary run measures logit-gradient RMS on at most 32 seed-0-selected training cases from its own epoch-five state (all ten in low data). This stage has no optimizer updates and restores random state. The shared depth term therefore gets the same coefficient across the three matched auxiliary arms within a cohort.

Each coefficient targets a median auxiliary/Dice RMS ratio of 0.10 with a 95th-percentile cap of 0.50. The depth and SDF terms are calibrated separately, then ramped from 0.2 to 1 over epochs 6–10. These are fixed training-only gradient scales, not validation-tuned hyperparameters. No outer-band, one-cut, equivariance, edge, topology, A/P-plane, geodesic-interface, teacher, or refiner term is added.

## Verification and outputs

Before submission: run geometry and gradient unit checks, interruption/resume equivalence, physical-coordinate checks, original-versus-layer distance controls, geometry preparation on all 260 annotations, and a real 64-cube SwinUNETR backward check. The all-case annotation audit is a preprocessing validity check, not calibration or outcome-based selection. Native NIfTI labels must match the pickle voxel-for-voxel. Local and cluster gzip archives differ in compressed bytes but their uncompressed pickle SHA-256 is identical; bind production to the verified cluster archive hash.

Inside the allocation: rerun unit checks and real CUDA forward/backward/optimizer smoke checks before the eight fresh runs. The smoke model is discarded. Freeze source hashes and protocol, save optimizer/scheduler/RNG/history in the latest checkpoint, and keep best weights separately. Resume must preserve model updates and early-stop selection; completed artifacts are hash checked.

After each run, evaluate the best checkpoint in original 3D coordinates. Save per-case A/P/foreground Dice, foreground false positives/negatives, A/P swaps, mean symmetric boundary-voxel distance and HD95, and error counts by signed depth band. Save predictions, best/latest checkpoints and training curves only in the ignored run directory. Flag empty predicted surfaces rather than reporting zero boundary distance.

## Submission

One Slurm allocation runs all eight experiments sequentially: four low-data runs followed by four full-data runs. It uses the existing `hippocampus` environment, `stud` partition, one A100 `4g.40gb` slice, four CPUs, 24 GiB RAM, and a 12-hour maximum allocation. The scheduler limit is separate from the per-run 75-epoch cap. Queue it after any currently running user job, preserving one running and one queued allocation. A durable submission receipt prevents duplicate submissions. Existing experiments and their files are not modified.

Remote root: `/mnt/beegfsstudents/home/3160552/depth_swinunetr_fold0_seed0_20261006_01`.

Any interrupted runs retain their checkpoints; no extra seeds, folds, objectives, or automatic repeat runs are scheduled. This task requests submission, not completion claims or a new monitoring automation.
