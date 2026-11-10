# GeoFWI3D 5%: four additional loss combinations

The user requested the four combinations in the attached screenshot and confirmed that “Dice” should retain the existing **foreground Dice + cross entropy** base objective. All four arms run sequentially in **one sbatch job on one GPU**:

1. Dice+CE + signed distance (`sdf`).
2. Dice+CE + bands + signed distance (`bands_sdf`).
3. Dice+CE + bands + edge (`bands_edge`).
4. Dice+CE + bands + signed distance + edge (`bands_sdf_edge`).

Each arm starts from scratch with model seed 0, the same 19 salt-containing training cases, the same 96 complete native 96³ validation cases, augmentation and deterministic sample order as baseline job 683068 and bands job 683092. Training is capped at 500 optimizer updates per arm; the 100-update LR warmup and original 5,000-update cosine horizon are retained. Validate at updates 100, 200, 300, 400 and 500; select the exact highest whole-volume mean foreground hard Dice. The configured early-stop patience cannot trigger before this fixed cap. There is no test evaluation or target-Dice stopping rule.

Bands retain the earlier six-connected two-voxel side-balanced BCE, omitting the observation cube's two-voxel frame. The signed-distance and edge objectives are binary adaptations of the current repository implementations in `thesis/new_constraints/signed_distance.py` and `thesis/new_constraints/edge_consistency.py`:

- SDF: target distances are negative inside salt and positive outside, with a half-voxel offset, clipped to ±8 native voxels and normalized by 8. The full-volume loss is `mean((p_salt - y_salt) * phi)`; subtracting the fixed target term makes perfect probabilities have zero loss without changing gradients. Distances use native voxel units, not millimetres. Missing/full foreground is assigned a constant +1/−1 field. Only selected training fields are computed and cached, and they receive the identical XY flips/rotations as the images and masks. No predicted distance head is added.
- Edge: mean squared error between predicted and ground-truth **signed neighbouring-voxel contrasts**, over positive-axis faces with both endpoints in the same inner/outer boundary support. Cube-face framing is excluded. There is no wraparound, class-agnostic edge-only prediction, learned edge head or extra inference input.

Each included auxiliary term has a fixed coefficient of 0.1, ramped linearly over 500 updates, matching the earlier bands policy. These coefficients are fixed before viewing the new arms' validation scores; there is no search for a winning coefficient or guaranteed improvement. Raw supervised/bands/SDF/edge component losses are recorded in the training histories. The different objectives have different computational costs, so equal optimizer-update caps do not claim equal FLOPs or wall time.

The budget itself was chosen using the earlier baseline validation curve. These are therefore single-seed exploratory selected-validation comparisons. The earlier controls remain 0.7374 (short-budget baseline), 0.7684 (short-budget bands), and 0.8327 (longer-budget 5% baseline); none is overwritten.

All 14 local regression checks passed in 57.51 seconds. Checks include SDF sign and gradient direction, transformation alignment, correctly oriented edge contrast and crop exclusion, training all four combinations on complete native volumes without requesting validation SDF supervision, and deterministic checkpoint resume. The existing verified data/model preflight is reused because the dataset and model architecture are unchanged.

Three cluster checks passed in 136.85 seconds, covering binary SDF geometry, signed face-edge supervision and a complete-volume training step with all three auxiliary losses. Frozen control comparisons also confirmed the same data, framework versions, architecture, sampling, augmentation and evaluation functions.

Submitted as **Slurm job 683761**, named `geofwi3d-four-losses`, using one GPU allocation and four sequential arms. The submission guard allowed at most one active job and no waiting jobs, with an `afterany` dependency if an active job remained. Job 683726 finished before the final queue check: the actual submission found zero running and zero queued jobs, so no dependency was needed. The earlier strict launcher's refusal while 683726 was running is retained in `queue_check.json`; the final check and receipt are `dependency_queue_check.json` and `submission.json`.

Campaign: `/mnt/beegfsstudents/home/3160552/geofwi3d_swinunetr_5pct_four_losses_500updates_seed0_20261005_01`. Frozen source, manifests and submission records are copied to `reports/geofwi3d_5pct_four_losses_20261005/cluster/`. No recurring monitor is created. All arms subsequently completed successfully at 11:00:24 Europe/Rome on 5 October 2026; selected validation scores and artifact checks are documented in [RESULTS.md](RESULTS.md).

Immediate scheduler verification showed job 683761 **RUNNING on gnode02**, starting at cluster-reported `2026-10-05T10:50:32`, with no other user jobs in the queue. Startup input verification passed and the error log was empty. All 14 frozen input hashes verified after retrieval; the manifest exactly matches the baseline control, including training and validation identities, and all original protocol fields except the requested methods match.
