# Proposal: a learned step surface that directly reinforces segmentation

Status: earlier design sketch. Superseded by [BOUNDARY_FUNCTION_AND_VOXEL_FEEDBACK_PROPOSAL.md](BOUNDARY_FUNCTION_AND_VOXEL_FEEDBACK_PROPOSAL.md), which incorporates the requirement that the fitted predicted function depend on voxel probabilities and send separate feedback to responsible wrong voxels. The earlier fold-0 pilot was withdrawn; neither proposal restarts it or claims a new result.

## What should be represented

Use coordinates `[X,Y,Z]`, with `X` selecting a sagittal section, `Y` running posterior to anterior, and `Z` running inferior to superior. At each `(x,y)`, predict whether foreground is present, its lower `z` edge, and its upper `z` edge. Across all `x`, these are a 2-D presence footprint and two 2-D surfaces enclosing the 3-D hippocampus union. The anterior/posterior class division is a separate prediction.

The previous fold-0 diagnostic found that reference presence plus reference lower/upper edges reconstructed the 52 reference union masks at mean Dice `0.9994`; the interval filled only 192 extra voxels among 174,350 reference foreground voxels. This supports the *two-surface representation* for this data. It does not show that a small number of steps along `Y` is sufficient. Before training, fit the best piecewise-constant lower and upper curves to **TRAIN and INNER-VAL labels only**, for several small step budgets, and measure the resulting oracle mask Dice, edge MAE, and tail/apex error. Choose the smallest acceptable budget on these cohorts; if every small budget damages the shape, retain a bounded non-step residual or abandon the step restriction.

## Learnable function, not independent edge predictions

The network should predict parameters of a function for each sagittal section. For either a lower-edge or thickness latent curve `e`, use a small number `K` of ordered, image-conditioned change points:

```text
f_e(x,y) = base_e(x)
         + Σ[k=1..K] jump_e(x,k) · sigmoid((y - location_e(x,k)) / T_y)
         + residual_e(x,y)
```

`base`, signed `jump`, and ordered `location` are predicted from MRI-derived features. Positive jumps remain possible where anatomy requires them. `T_y` makes knot locations differentiable during training; a lower temperature approaches a step function. Bound and penalize the residual (for example to roughly one voxel) so that a small local deviation can be represented without letting it replace the step model. Predict `K` only after the oracle audit. An alternative first implementation is a cumulative sum of per-position jumps with an L1 penalty; it is simpler but does not fix the maximum number of jumps.

Use one latent curve for the lower edge and one for positive thickness. Map them to ordered voxel coordinates, for example `L=(Z-1)·sigmoid(f_lower)` and `U=L+(Z-1-L)·sigmoid(f_thickness)`. A separate image-conditioned presence branch predicts `q(x,y)`. This guarantees `L≤U` and lets `q` represent disconnected occupied `Y` runs. Use spatial context across neighboring `x` sections so the two sheets form a coherent 3-D surface; do not impose a universal monotone descent on all patients.

The prior pilot's head received only the final decoder features and used a `Z` mean for presence. For a complementary boundary cue, use the full-resolution decoder features together with an early high-resolution image feature, and use max-plus-mean or attention pooling along `Z` for the presence branch. This is a design hypothesis to test, not an established cause of the prior false positives.

## Differentiable rendering and direct reinforcement

Render a soft foreground occupancy at every voxel:

```text
S(x,y,z) = q(x,y)
         · sigmoid((z - L(x,y) + 0.5) / T_z)
         · sigmoid((U(x,y) - z + 0.5) / T_z)
```

The rendered mask remains differentiable with respect to presence, jump heights, and jump locations. During training, supervise `q`, `L`, and `U` with reference presence and edges; also give `S` a union-mask Dice/BCE loss. Empty reference columns receive no edge target. Keep the normal three-class Dice objective. A small boundary-local agreement loss should directly train the segmentation foreground probability `p_fg=p_anterior+p_posterior` toward the **detached, supervised** `S`; this creates the missing gradient into segmentation logits. The contour head stays anchored to labels by its own losses rather than chasing segmentation errors. Calibrate the agreement gradient on TRAIN only.

For an explicit inference contribution, test a separate bounded fusion arm: `p_fg_final=(1-α)p_fg+αS`, where `α(x,y,z)` is learned from image features and uncertainty, confined near the predicted outer boundary, and allowed to approach zero. Obtain final class probabilities as `p_background=1-p_fg_final` and `p_class=p_fg_final·softmax(class_logits_over_foreground)`. This lets the contour reinforce the outer boundary while leaving the anterior/posterior split to the class branch. Train and select the fused output end to end. Record the learned gate and report raw and fused outputs; never substitute a thresholded head mask for the segmentation as the primary test.

## Focused experiment

Keep one patient split, seed, backbone, optimizer, augmentation policy, early stopping rule, and predeclared DEV evaluation. Compare:

1. Canonical Dice-only baseline.
2. Dense contour head with rendered-mask supervision and direct boundary agreement, without the step parameterization.
3. Learned step-function head with the same supervision and agreement.
4. Arm 3 with bounded differentiable fusion.

This separates the value of direct contour-to-segmentation coupling, the step-function representation, and inference-time fusion. Select checkpoints on INNER-VAL segmentation quality plus predeclared contour safety criteria, not on DEV. Report per-patient union and anterior/posterior Dice, HD95 or surface Dice, presence precision/recall, edge MAE, step count/location, tail/apex error, raw-versus-fused changes, gate use, and cases harmed. Include an ablation showing that gradients from the agreement/fusion losses actually reach segmentation logits and the step parameters. The fold-0 DEV cases have been studied repeatedly; confirmation needs a separate patient cohort.

## Scope and precedents

This two-surface model describes the **outer foreground union**. It cannot by itself determine the anterior/posterior internal interface. If the outer-boundary mechanism helps, an internal interface function could be studied separately with its own representation audit and supervision.

Differentiable shape-to-mask training is related in spirit to [shape-aware organ segmentation with predicted signed-distance maps](https://arxiv.org/abs/1912.03849), which included hippocampus data. [Boundary loss](https://arxiv.org/abs/1812.07032) motivates coupling boundary and regional supervision. Neither paper validates the particular sparse step surface proposed here; its feasibility and benefit must be measured by the oracle audit and matched experiment above.
