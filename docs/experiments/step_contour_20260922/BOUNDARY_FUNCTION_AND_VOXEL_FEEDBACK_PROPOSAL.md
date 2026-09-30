# Proposal: fit a boundary function from predicted voxels and correct its causes

Status: design only, 2026-09-22. The withdrawn pilot is not resumed. This proposal refines the earlier [learnable step-surface sketch](LEARNABLE_STEP_SURFACE_PROPOSAL.md) by making the predicted function depend differentiably on the segmentation probabilities and adding a separate loss on voxels responsible for a bad fit. The current implementation choices and experiment arms are specified in the [function-head loss derivation](FUNCTION_HEAD_LOSS_DERIVATION_20260922.md); where they differ, use that later specification.

## 1. Define the reference function before choosing a network

Let the reference foreground be the union of hippocampus labels 1 and 2 in `[X,Y,Z]`. For each sagittal section `x` and anterior–posterior location `y`, extract three exact label-derived targets: occupancy `q*(x,y)`, first foreground position `L*(x,y)`, and last foreground position `U*(x,y)`. Empty columns have no edge target. Along `z`, foreground is approximately the two-step interval `H(z-L*)H(U*-z)`. Across all `x`, the occupancy footprint and the two edge surfaces represent the 3-D outer union. The previous fold-0 audit found mean oracle Dice 0.9994 for this representation, but that result must be checked across the intended training data and other folds.

There is a second, distinct step hypothesis: as `y` changes within one occupied run of a sagittal section, `L*` and `U*` may be approximated by a *small number* of constant plateaus and jumps. Integer voxel labels are trivially a staircase if every adjacent `y` can jump; that imposes no useful prior. Fit the best `K`-step functions to annotated curves by dynamic programming, separately for disconnected occupied runs, without requiring all jumps to descend. On TRAIN and INNER-VAL only, render the fitted curves and record union Dice, edge MAE, tail/apex error, and the number/magnitude of residual deviations for several small `K`. If a compact staircase loses important shape, retain an explicit bounded residual or do not impose a small-step model. Do not replace raw annotated edges by an imperfect fitted target without reporting the approximation error.

## 2. Fit the predicted function from soft voxel predictions

The segmentation branch produces foreground probability `p(x,y,z)=p_1+p_2`. Do **not** fit a function to `argmax` labels: that would sever or destabilize the gradient to the voxel predictions. For each `z` ray, compute differentiable first/last-hit masses:

```text
m_lower(z) = p(z) · Π[t<z](1-p(t))
m_upper(z) = p(z) · Π[t>z](1-p(t))
```

They indicate which predicted voxels cause the apparent outer edges. This project already implements these masses in [`terminal_distributions`](../../../thesis/new_constraints/directional_steps.py). Use stable float32/log-domain arithmetic if needed near saturation.

A small fitting head receives these ray distributions (and image-derived features for ambiguous boundaries) and predicts `q̂(x,y)` plus a *function* for `L̂_x(y)` and `Û_x(y)`. The current design adds flexible, zero-initialized residual logits to the log first/last-hit masses. This starts with an exact tie to the raw voxel path on occupied rays while letting image features rescue missed edges. A quadratic curve fit is the first implementation; differentiable TV and explicit change-point models are later comparisons. Begin with a crossing penalty and measure violations; if they persist, use an ordered parameterization or joint constrained fit. Keep the route from `p` through the ray masses to the fitted function non-detached so function errors train voxel logits as well as the head. Empty columns and disconnected `y` runs need separate handling. Use spatial context across neighboring `x` sections to keep the 3-D surfaces coherent.

## 3. Send two distinct training signals

**Function signal.** Compare the predicted and reference functions *as evaluated curves*, not their possibly non-unique knot parameters. Supervise presence, lower/upper edge positions on occupied columns, and jump/displacement patterns within connected runs; render the predicted interval softly and compare it with the reference union mask. The target is the patient's annotated geometry, including genuine upward excursions. With no detachment between `p` and the fitting head, this loss updates the head and the segmentation network.

**Responsible-voxel signal.** Give extra supervised voxel loss to the specific wrong predictions that distorted the fitted boundary. Let a detached responsibility weight include predicted first/last-hit mass plus a narrow reference-boundary band. Apply a normalized, bounded weighted BCE (or CE) between voxel foreground probability and its true label. A far false-positive voxel can become the predicted first hit and receives greater penalty; a missed true edge is covered by the reference-boundary band. Detach the *weight* so the network cannot lower its penalty merely by moving responsibility elsewhere. Keep ordinary three-class Dice as the anchor and inspect false-positive and false-negative gradients separately. This term is distinct from the function loss: it identifies which voxel decisions caused the function error.

For example, if the true occupied interval is `z=20..26` but a high-probability false positive appears at `z=17`, the fitted lower edge shifts down. The function term penalizes the three-voxel edge error and backpropagates through the first-hit calculation. The responsibility-weighted voxel term additionally tells `p(z=17)` directly that this high-impact foreground prediction is wrong.

## 4. Choose the spatial scope

Start with **all sagittal sections**, not only the central or longest one. Per-`x` step functions along `y`, connected through shared 3-D features and a small target-aware cross-`x` consistency term, produce the two whole-volume boundary sheets. This is an interpretable 2.5-D parameterization of the 3-D **outer foreground union**. A free 3-D step function has no natural single ordering direction and obscures which voxels caused a boundary error. If the two-sheet oracle fails on other data, or cross-`x` surface errors remain dominant, compare a 3-D signed-distance-field alternative. Neither outer-boundary model resolves the internal anterior/posterior class interface; that needs its own target and evaluation.

## 5. Focused evaluation before any large run

First complete the label-only oracle audit and finite-difference/autograd checks. Verify that a mistaken boundary voxel changes the fitted function and receives nonzero gradients from both terms, that an empty column creates no edge target, and that disconnected runs do not interact. Then use one matched split and predeclared weights to compare: Dice baseline; function comparison without responsible-voxel weighting; both terms; and, only if the oracle supports it, the compact `K`-step head against a dense-function control. Select on INNER-VAL, open DEV once, and report per-patient class/union Dice, surface or HD95 distance, presence precision/recall, edge accuracy, step locations, and false-positive/false-negative attribution. Do not use the withdrawn fold-0 experiment as confirmation.

Related precedents include [boundary loss for segmentation](https://proceedings.mlr.press/v102/kervadec19a.html) and [joint shape/segmentation training through signed-distance maps](https://arxiv.org/abs/1912.03849). They motivate explicit boundary supervision; the particular first-hit, step-function, and voxel-responsibility design above is a hypothesis to test.
