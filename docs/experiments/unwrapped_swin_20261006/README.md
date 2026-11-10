# SwinUNETR with an unfolded MRI branch

One Slurm allocation runs 10 configurations at both 5% (10 training cases) and full data (208 training cases), using the same 52 validation cases, MSD fold 0 and seed 0. Each main run has a 75-epoch cap and early stopping: patience 8, minimum 25 epochs, minimum improvement 0.0005. Best-checkpoint selection uses mean per-patient anterior/posterior Dice.

| Configuration | Loss | Purpose |
|---|---|---|
| Native Dice | Dice | Reproduce the original baseline |
| Native depth | Dice + original depth-balanced CE | Reference for every additional branch |
| Plane local / unfolded local | Same depth loss | Isolate the effect of unfolding with a local encoder |
| Plane context / unfolded context | Same depth loss | Add a lower-resolution context path and whole-chart summary |
| Plane position / unfolded position | Same depth loss | Add longitudinal position and signed plane-depth inputs |
| Plane attention / unfolded attention | Same depth loss | Add channel/spatial gating to aligned features |

Each paired row comprises two configurations, for 10 configurations and 20 main runs. There is no SDF, focal weighting, class weighting, topology loss or auxiliary branch loss. Depth coefficients are frozen from the original training-only calibration: 0.0029145569633035585 for 5% and 0.08034038718952953 for full data. Depth activation follows the original 5-epoch Dice warmup and 5-epoch ramp.

## Input construction and fusion

The geometry is the actual cut-shell harmonic chart construction used by the two-part visualization. It is not centerline straightening. Keep both cut sides and additional disconnected surface patches. Every ROI voxel has surface UV coordinates, physical distance from the cutting plane, and a saved native voxel index. The ROI includes an 8-mm margin around predicted tissue within the acquired MRI field of view.

The auxiliary encoder shares weights across chart pieces. Its learned features are sampled at the saved chart coordinates and scattered back to their source MRI voxels. They are normalized and fused with the final full-resolution SwinUNETR decoder features. The final segmentation and every training loss remain in the native 64-cube grid.

The plane control receives exactly the same source MRI samples, chart membership, source voxel indices, grid dimensions, encoder and fusion capacity as its unfolded partner. It uses plane coordinates instead of harmonic UV. Plane pixel spacing is adjusted to match tensor dimensions; physical resolution and interpolation distortion are therefore reported rather than claimed identical.

The chart rasterization is not an invertible volume warp. Projection and splatting can merge nearby samples, particularly in compressed chart regions and background margins. Per-case audits record collisions, MRI reconstruction error, triangle distortion, tissue coverage and sampling support. Native MRI remains available to SwinUNETR throughout.

## Guide preparation

Geometry never uses the case's annotation as an input. Two guide models per data regime train for exactly 25 epochs on opposite halves of that regime's training cases and predict their held-out training cases. Their weights are not selected using held-out labels. Validation geometry comes from the main Dice model's fixed epoch-25 predictions, before and independently of best-checkpoint selection. No full-data guide is used for the 5% experiments.

These four guide preparations are shared by all branch arms and run inside the same allocation. Out-of-fold guides see fewer cases than the validation guide; that difference is a limitation of this first prototype, especially at 5%.

## Validation and outputs

Local checks cover 52 existing real prediction masks, source-voxel correspondence, depth preservation, retained surface pieces, shared-encoder gradients, full SwinUNETR forward/backward and optimizer/checkpoint resume. Before training, a separate process inside the allocation checks MONAI 1.6 and CUDA bf16 forward/backward. Every new guide atlas must pass geometry checks before any corresponding branch run starts; each regime also tests its largest atlas on CUDA.

`status.json` records stages; `logs/` holds per-stage logs; `runs/<fraction>/<arm>/` holds best checkpoints, history, predictions and metrics. Completed runs discard only their own optimizer restart checkpoint to stay within storage quota. `comparison.json` reports paired Dice changes, patient bootstrap intervals, missed/extra tissue and A/P swaps, surface distance and HD95 in run summaries, and corrected/regressed voxels by signed depth bin. These are exploratory comparisons from one fold and seed, not evidence of generalization across training seeds.

`protocol.json` freezes data splits, hashes, geometry, coefficients and configuration. `submission.json` records the confirmed single job ID. The remote campaign root is `/mnt/beegfsstudents/home/3160552/unwrapped_swin_fold0_seed0_20261006_01`.
