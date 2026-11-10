# Native MRI auxiliary branch with anatomical constraints

Authorized scope: SwinUNETR, MSD Task04, fold 0, seed 0, **5% only**. The user selected the six screenshot rows crossed with depth balancing off/on: twelve fresh runs in one sequential GPU allocation.

| Screenshot objective | Depth off | Depth on |
|---|---|---|
| Dice baseline | `dice` | `dice_depth` |
| Dice + bands | `dice_bands` | `dice_bands_depth` |
| Dice + signed distance | `dice_sdf` | `dice_sdf_depth` |
| Dice + bands + signed distance | `dice_bands_sdf` | `dice_bands_sdf_depth` |
| Dice + bands + original edge | `dice_bands_edge` | `dice_bands_edge_depth` |
| Dice + bands + signed distance + original edge | `dice_bands_edge_sdf` | `dice_bands_edge_sdf_depth` |

## Matched branch and data

Use the existing `native_mri_context` architecture from the successful native-MRI control. The main network sees the padded native MRI. The separate encoder sees MRI samples from the same frozen guide-derived hippocampal ROI with 8 mm context, placed at their original voxel centres and original axis orientation. The prior source memberships remain separate for pooling and feature return; no plane-resampled or unwrapped MRI tensors are consumed. Auxiliary convolutions learn one MRI channel only. Support masks remain geometric bookkeeping. The same native decoder, context encoder and ordinary fusion train jointly from scratch in every arm; 15,724,013 trainable parameters. No attention gate, pretrained segmentation weights, learned anatomical prior, or profile loss is added.

Reuse the exact ten training subjects and 52 validation subjects in the previous 5% fold-0/seed-0 split. The original fold has 208 training cases; rounding yields 10/208 (4.81%). Training branch inputs use frozen out-of-fold guides. Validation annotations never construct branch inputs. New GT-derived SDF and depth weights are training targets only; annotation geometry is also used for final evaluation diagnostics.

## Losses and optimization

All losses act on the final three-class native-grid logits. Dice includes background. Bands use the existing two-step, six-neighbour GT inner/outer bands with balanced BCE and no focal/degree weighting. Original edge is the existing signed foreground face-contrast residual, squared and averaged over faces whose endpoints lie in those bands. Signed distance is the original full-volume anterior/posterior `(p-y)*phi` loss with half-voxel EDT, clipped and normalized at eight voxels; the native data are checked to have 1 mm isotropic spacing. Depth balancing is the existing three-class CE weighted over signed 2 mm depth bins in GT foreground plus 8 mm, with the original sparse-bin mass cap at 64 voxels.

Retain the native-branch experiment's 75-epoch maximum, batch size one, 64³ grid, feature size 24, AdamW (learning rate 1e-4, weight decay 1e-5), StepLR every 20 epochs with factor 0.5, CUDA bf16, no augmentation, and early stopping after at least 25 epochs with patience eight and minimum improvement 0.0005. Select the checkpoint with best validation A/P macro Dice.

Each arm starts with five identical Dice-only epochs. At the end of epoch five, use all ten training cases to calibrate active terms in the existing order: bands, original edge, signed distance, depth. Each new coefficient targets median gradient RMS ratio 0.1 relative to the already weighted combined objective, capped at p95 ratio 0.5. Fix these coefficients and ramp all active terms linearly during epochs 6–10. This is an epoch-matched native-branch study; it does not import the late SDF activation or FLOP budget from the separate capped-compute study. The fresh Dice-only control never calibrates auxiliaries.

## Validation and outputs

CPU checks verify all twelve loss values and gradients against the original implementation, sequential weight calibration parity, exact restart before/after calibration, Dice-only prefix parity, and full-network gradient flow and optimizer updates in native/encoder/fusion groups. The job repeats a full native-branch optimizer smoke on real training data under the allocated MONAI 1.6 CUDA bf16 environment, checking all ten training inputs and targets before scientific runs start.

Every run saves a best checkpoint, complete learning curve, coefficients and calibration evidence, validation predictions, per-case A/P and foreground Dice, extra/missed/swapped voxel counts, mean surface distance, HD95, and per-depth diagnostics. The active run keeps an atomic optimizer/RNG/scheduler checkpoint for resume. Completed runs are verified and skipped on restart. `results.json`, `results.csv`, and `RESULTS.md` update after each completed arm, with per-case deltas versus the matched Dice-only branch and depth-on versus depth-off comparisons.

This is one exploratory fold/seed and twelve validation comparisons. The results can prioritize replication but cannot establish a reliable gain by themselves.

## Deployment

Remote root: `/mnt/beegfsstudents/home/3160552/native_branch_constraints_fold0_seed0_5pct_20261008_01`.

One `stud` allocation requests an A100 `4g.40gb` GPU slice, four CPUs, 24 GiB RAM and 12 hours. Stage only source and protocol; share the existing frozen 5% atlas read-only by symlink. The submitter verifies source, payload, dataset and atlas hashes, completed local checks, available quota (minimum 3 GiB), and the user's queue. A durable submission lock/receipt prevents duplicate submissions. If one allocation is already running, this job depends on its completion; it never cancels existing jobs.

Prepare and validate locally using the project Python environment:

```bash
python -m scripts.native_branch_constraints.tests --mode unit --output evaluation_output/native_branch_constraints_20261008/unit_checks.json
python -m scripts.native_branch_constraints.tests --mode model --output evaluation_output/native_branch_constraints_20261008/model_smoke.json
python -m scripts.native_branch_constraints.package
```

The remote staged source provides `stage.py`, `submit.py`, and `run.sh`. See `protocol.json` and `local_checks.json` for the frozen recipe and verification evidence; `submission.json` records the submitted job.
