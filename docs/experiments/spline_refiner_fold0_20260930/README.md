# Learned cubic B-spline boundary refinement, fold 0

**Completed: this first learned refiner is essentially tied with the baseline.**
The selected epoch improves mean validation foreground Dice by only 0.000266
(0.0266 percentage points). Boundary distances barely change. This does not
yet establish useful learned refinement, despite the earlier single-case
annotation-guided spline fit demonstrating representational potential.

Train a small image-conditioned coefficient regressor on all 208 original
fold-0 training cases, with the canonical Dice-only, no-augmentation,
early-stopped SwinUNETR frozen. Evaluate on the original 52 validation cases.
The first objective is outer foreground union correction; it does not train
the anterior/posterior interface.

## Frozen protocol

- Canonical baseline checkpoint SHA256:
  `abbc13be98a720d9c25ed2ee2aefb0a3a7f20811148734e7561886c63a93baf1`.
- Use the original cluster dataset with SHA256
  `3311d2233b99e09080c40ca80f1035781d9c1dc9c5b109dc266c73eb52a3df01`
  and original fold split with SHA256
  `1d6a3fe993359ad85e9c28a29901162251c3359449b710c09cb9e92d66dda472`.
- Existing MONAI normalization and 64-cubed pad/crop preprocessing. Frozen
  segmenter runs once per case in FP32; paired comparisons use those same maps.
- Five inputs: normalized MRI, summed foreground softmax probability, source
  signed distance divided by 4 and clipped to [-1,1], predicted inner band,
  predicted outer band. Hard foreground is the three-class argmax union.
- Existing two-step, six-connected morphology defines the edit bands from
  the initial prediction. Annotation-derived bands are not inputs.
- 222,291-parameter convolutional head: 16/32/64 channels, two downsamplings,
  GroupNorm/GELU; interpolate features to the control grid; 3-channel head.
- Uniform tensor-product cubic B-spline stationary velocity, 4 mm spacing,
  20x20x20 control vectors on the full crop. Each component bounded by
  `4*tanh(raw)`. Zero-initialized last convolution gives the identity.
- Smoothstep taper of prediction-band distance/2 restricts velocity;
  six scaling-and-squaring steps integrate the backward field. Warp source
  signed distance by trilinear sampling; threshold at zero. Hard preservation
  of source signed distance outside the editable bands is explicit.
- Loss: prediction-band mean smooth-L1 target signed-distance error,
  plus 0.25 soft Dice loss, 0.005 velocity magnitude, 0.02 velocity gradient,
  plus band-mean `relu(0.3 - det(J))**2` compression penalty.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 4, seed 0,
  gradient norm clip 5, FP32, no augmentation. Maximum 50 epochs, minimum 25,
  patience 8, meaningful improvement 0.0005. Save the highest validation
  union Dice, including identity epoch 0; report final epoch separately.
- Annotation-independent inference safeguard: fall back to the original
  case if sampled backward Jacobian minimum is <=0.05 or nonfinite.
  Report both raw and safeguarded Dice and fallback counts.
- Preserve initial AP labels on retained tissue. New foreground receives
  the nearest original foreground label. All labels outside bands remain fixed.

## Checks and interpretation

CPU/GPU checks cover the cubic basis against SciPy, axis ordering, analytic
affine Jacobians, identity, exact outside-band preservation, AP preservation,
nonzero network gradients, and finite-difference geometry gradients. A
two-train/two-validation-case smoke test precedes the full run.

Primary comparison is paired per-case foreground Dice. Also report boundary
voxel ASSD/HD95, FP/FN, errors fixed/introduced, case-level wins/losses, AP Dice,
and geometry safeguards. Distances use boundary voxel centres, not continuous
mesh quadrature; compare only like estimators. Saved masks use the padded
64-cubed model coordinates, not native NIfTI coordinates.

The frozen segmenter's training predictions are in-sample. Thus a training
versus validation error distribution shift is a known limitation. Fold 0 has
also already been used for baseline early stopping and exploratory work,
including case 164. This is a development comparison, not a final independent
generalization estimate. Bootstrap intervals are descriptive and do not
correct for checkpoint selection. One seed is a pilot. Positive sampled
Jacobians do not prove continuous injectivity or discrete topology preservation.

Source: `semantic_constraints/spline_refinement/`.
Launcher: `scripts/run_spline_refiner.sh`.
Large checkpoints, prediction archives, and runtime output stay untracked.

## Execution record

- Cluster run directory deleted on 2026-09-30 at the user's request after
  confirming the queue was empty. Local results, selected checkpoint, and
  verification remain in `results_677424/`; the remote paths below are historical.
- Cluster root: `/home/3160552/spline_refiner_fold0_20260930_01`.
- Slurm job 677422 stopped before importing the experiment: the launcher's
  original source-directory calculation used Slurm's copied script location.
  No training ran. The corrected launcher uses the explicit run-root argument.
- Slurm job 677424 runs the corrected, self-contained source bundle, unit
  checks, smoke test, and full experiment. Numerical jobs use `sbatch` only.
- Local pre-submission checks: four geometry tests passed; a full 64-cubed
  case-164 forward/backward step produced finite loss and gradients.
- User-authorized cleanup losslessly compressed 832 old audit `.npy` files in
  `hippopotamus_runs/swin_overfit_audit_665389`. Each gzip copy was decompressed
  and checked against the original SHA256 before removing the redundant `.npy`.
  Logical space saved: 1,227,986,304 bytes. Arrays, summaries, and checkpoints
  were retained. The remote audit root contains `lossless_cleanup_20260930.json`.
  Old scripts reading those audit arrays must decompress `.npy.gz` first.

## Completed results

Training job 677424 completed successfully in 10m29s, including imports,
checks, smoke test, prediction caching, training, and evaluation. Optimization
took about 5m08s. Early stopping ended at epoch 25; the selected checkpoint is
epoch 20. Independent saved-mask verification job 677435 completed successfully
in 13s. Both Slurm exit codes were 0:0.

| Metric, mean over 52 validation cases | Frozen baseline | Selected spline refiner |
| --- | ---: | ---: |
| Whole-hippocampus Dice (primary) | 0.900978 | 0.901243 |
| Boundary-voxel ASSD, mm | 0.421466 | 0.420385 |
| Boundary-voxel HD95, mm | 1.141528 | 1.141528 |
| Mean anterior/posterior class Dice | 0.878614 | 0.878869 |
| Total foreground false positives | 19,159 | 18,424 |
| Total foreground false negatives | 15,438 | 15,957 |
| Total foreground errors | 34,597 | 34,381 |

- 30 cases improved and 22 worsened; none were unchanged.
- 787 errors fixed, 571 introduced: net 216 fewer errors.
- Mean Dice gain: 0.000266; median gain: 0.000309. Descriptive paired bootstrap
  95% interval for the mean gain: [-0.000105, 0.000649], including zero.
- No geometric fallback in any selected validation case. No changes outside
  the prediction bands. No AP changes on retained initial foreground.
- Final epoch 25, reported separately, scored 0.900653 Dice, below baseline.
- Training Dice at the selected epoch: 0.938333 → 0.938786. The small training
  gain means this pilot is not evidence that strong fitted corrections failed
  only at generalization; optimization, target formulation, and model capacity
  remain possible limitations too.

Case 164: Dice 0.907598 → 0.908367; 78 errors fixed and 60 introduced.
ASSD 0.436413 → 0.429799 mm; HD95 remains 1.414214 mm. Minimum sampled
backward Jacobian 0.672909, maximum displacement 0.691975 mm. Its AP mean Dice
changes from 0.783106 to 0.785620 through outer-boundary edits only.

These absolute case-164 numbers differ from the earlier local CPU oracle
experiment. This run uses the canonical historical dataset and CUDA FP32
inference; the local experiment used a different runtime. Use this run's
paired baseline for comparisons, not the earlier CPU mask's absolute Dice.

The independent report reopens all 52 saved mask archives, checks exact split
membership, independently reconstructs the six-connected bands, recomputes
Dice/FP/FN/fixed/introduced counts, and verifies exact outside-band and retained
AP preservation. Results are in `results_677424/verification.json`.

Local untracked artifacts: `results_677424/`, including `results.json`,
`protocol.json`, `history.json`, per-case CSV/JSON, `best.pt`, `last.pt`, the 52
validation archives, and `overview.png`. The code and protocol are versionable;
data and checkpoints remain excluded from Git.

The next diagnostic should test whether this same regressor can deliberately
overfit a small training cohort to substantial boundary gains before changing
the full-cohort model. That would distinguish weak fitting from transfer
limitations. It has not been launched as part of this pilot.
