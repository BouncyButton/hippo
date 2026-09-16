# Protocol-derived A/P plane

The MSD Task04 data descriptor states that the last hippocampal-head slice is
the coronal slice containing the uncal apex and that posterior hippocampus is
the union of body and tail. The released arrays are RAS-oriented; the local
260-case audit verifies that stored spatial axis 1 is posterior-low and
anterior-high in every case.

The encoded rule is therefore

```text
exists integer c such that, for every predicted foreground voxel v:
    y(v) >= c  =>  class(v) = 1 (anterior)
    y(v) <  c  =>  class(v) = 2 (posterior)
```

This is a patient-specific property of the annotation procedure. It is not one
universal coordinate, a PCA/long-axis constraint, or an oblique anatomical
surface.  The descriptor supports the per-scan landmark rule; exact separability
of the released arrays is an empirical audit result, not a verbatim source claim.

## Components

`ExistentialAPPlaneLoss` evaluates every legal per-volume cut and backpropagates
through the lowest-loss one. It uses a stable softplus loss on the class-1 minus
class-2 logit margin, so confidently wrong assignments retain a corrective
gradient. Ground truth supplies foreground support only; ordinary supervised
segmentation loss locates the correct cut.

`BestFitAPPlaneLocationLoss` directly targets cut displacement.  For each
training label it finds the cut with the fewest hard A/P side disagreements,
then evaluates the same stable voxelwise margin loss at that patient-specific
target.  It also derives the model-selected cut from the current logits and
reports cut MAE and exact-cut rate every epoch.  It changes neither the model
head nor inference: the network still produces three voxelwise logits and raw
argmax predictions.

At zero margin this location objective is mathematically foreground-conditional
binary A/P cross-entropy against the best-plane projected labels.  The matched
`OriginalLabelAPConditionalCELoss` control uses the original voxel labels with
the same normalization and gradient-calibration policy.  It is required before
attributing any advantage specifically to plane projection rather than to the
extra conditional A/P supervision.

`hard_project_ap_plane` remains a mechanism-audit utility only. The planned
training experiment does not call it: evaluation uses the network's raw argmax
prediction, with no projection, relabeling, or other A/P post-processing.

Both classes are required by default. This is a separate empirically verified
Task04 property (260/260 masks contain labels 1 and 2), not a logical consequence
of the existential statement alone.

## Released-mask caveat

The final NIfTI masks are not bitwise planar in every case: 202/260 are exact;
58 contain one or two mixed coronal slices. Their best planes disagree with
1,388 of 856,754 foreground voxels (0.162%; median 22.5 in an affected case).
The existential loss therefore does not reject mixed labels or derive its cut
from them.  The location loss keeps all 260 cases and uses their unique
best-fitting cut; its target deliberately treats the small minority remnants as
annotation/raster exceptions to the documented generative rule.

The best cut is unique in all cases, but a unique optimum can still be weak.
After the exact 64-cubed training transform, the median second-best Hamming gap
is 77 voxels; the minimum is one voxel.  Training and final evaluation therefore
record target cost and second-best gap for sensitivity analysis.

## Frozen-cache mechanism check

On the local 52-case fold-0 probability cache, hard projection changed 799
voxels and moved foreground macro Dice from 0.872271 to 0.873642 (+0.001370).
It improved 27 cases and worsened 16. With an oracle best cut but the same
predicted foreground union, macro Dice is 0.891367. Plane enforcement is thus
useful, but accurate cut localization remains the larger opportunity.

## Verification

```bash
/Users/filippofocaccia/anaconda3/bin/python3 -m pytest \
  thesis/new_constraints/ap_plane/test_existential.py \
  thesis/new_constraints/ap_cut/test_posterior.py -q
```

## Seed-0 training protocol

Calibrate the coefficient on 32 deterministic fold-0 training cases using the
existing Dice-only seed-0 checkpoint:

```bash
python thesis/new_constraints/ap_plane/calibrate_weight.py \
  --pkl /absolute/path/to/msd_hippocampus_full.pkl \
  --dataset MSD \
  --splits-json /absolute/path/to/splits_final.json \
  --checkpoint /absolute/path/to/dice_seed0/checkpoint_latest.pt \
  --output /absolute/path/to/ap_plane_calibration_seed0.json \
  --max-cases 32 --seed 0 --spatial-size 64 64 64 \
  --ap-plane-axis 1 --ap-plane-anterior-side high \
  --ap-plane-margin 0 --device cuda --amp
```

The canonical policy targets a weighted plane/Dice logit-gradient RMS ratio of
0.10 at the median and caps the plane-gradient p95 at 0.50 of the median Dice
gradient. Use the exact reported weight and bind the run to the report:

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set ap_plane --supervised-loss dice \
  --ap-plane-weight WEIGHT_FROM_REPORT \
  --ap-plane-calibration-json /absolute/path/to/ap_plane_calibration_seed0.json \
  --ap-plane-axis 1 --ap-plane-anterior-side high \
  --ap-plane-margin 0 --seed 0 --fold 0 --epochs 50 \
  --constraint-warmup-epochs 5 --constraint-eval-every 1
```

The existing seed-0 Dice baseline is the fixed control and is not retrained.

For the direct location experiment, add `--ap-plane-objective location` during
calibration and launch with `--constraint-set ap_plane_location`.  For its
matched control, use `--ap-plane-objective conditional_ce` and
`--constraint-set ap_plane_ce_control`.  All remaining arguments and the
raw-inference policy are identical.
