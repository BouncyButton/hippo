# Graph-interface discrimination and unary-preserving cuts

Frozen before new feature extraction or intervention results, 23 September 2026.

## Scope and hypothesis

Test whether spatially resolved interface geometry and MRI edge context improve
A/P cut localization beyond current model probabilities and existing pooled MRI
features. Independently measure a uniform-affinity, unary-preserving graph cut.
This is a training-only feasibility screen, not a new training default.

Primary cohort: all 208 fold-0 training cases, unaugmented early-stopped seed-0
model. No fold-0 development cache is opened. The backbone was trained on these
cases; fourfold probe holdout does not make backbone predictions out of fold.
Thus a positive result requires out-of-fold replication, while a negative result
rejects only this finite descriptor/readout experiment. Participant grouping is
unavailable; all splits and uncertainty estimates are case-level.

## Inputs and fixed representations

Recompute CPU inference with the existing no-resize 64-cubed preprocessing and
verify each compact cached candidate feature matrix, cut, target and baseline
metric. Cache dense margins and predicted masks only in ignored experiments/.
Use original labels only for targets and measurement. All graph features use
predicted support. MRI normalization uses the predicted largest component.

Candidate cuts are all interior Y boundaries through the largest predicted
component, matching the existing context probe. Posterior lies at y<cut.
Render each candidate on the entire original foreground for scoring.

New graph descriptors pool face-neighbour geometry and MRI signals at each
candidate interface, globally and separately in four X/Z sectors about the
predicted-component centroid. Geometry uses distance to the exterior and
smoothed signed-distance normals/curvature at scales 1 and 2 voxels. MRI uses
the existing six normalized multiscale maps, recording endpoint means and
absolute differences. Each pooled signal contributes mean and standard deviation;
empty sectors give zeros and an explicit occupancy feature. These are candidate
graph-interface descriptors, not a trained GNN or a per-edge classifier.

Arms: existing base features; base+existing MRI; base+new surface geometry;
base+new geometry+edge MRI; and base+within-case shuffled new descriptors.
The shuffle preserves dimensionality and jointly permutes the new feature rows
with a fixed case-derived seed. It is a negative control, not a permutation test.

Readout: standardized logistic cut ranking, fixed C=0.1, maximum 3000 iterations,
case-balanced positive/negative sample weights. Four case folds, shuffle seed
20260923. No parameter sweep or selection on these out-of-fold outcomes.

## Displacement experiment

In addition to natural model outputs, replace A/P margins by a signed linear
transition centred at target cut -2, 0, or +2 slices, slope 2 per voxel. Keep the
model's predicted foreground and MRI unchanged. Recompute base probability
features from these synthetic margins; image/surface descriptors are unchanged.
All three variants of a case stay in the same fold, and receive equal combined
case weight. This is a synthetic recovery diagnostic using label-derived corruptions,
not deployment validation. Report correct-cut preservation and both displacement
directions. Skip no cases silently if a target/corruption is outside support.

## Uniform graph-cut intervention

Six-face graph on the entire fixed predicted foreground. Conditional A/P unaries
are stable negative log probabilities from the dense logit difference; pairwise
cost is lambda times the number of cut voxel faces. Lambdas: 0, 0.05, 0.2, 1.
No endpoint seeds, volume balance, learned edges, or reference-derived constraints.
Lambda zero preserves original A/P labels. Report degenerate single-class outcomes.

SciPy integer max-flow uses capacities rounded at scale 10000. Report the original
floating objective and a rounding-error bound; optimality is for quantized energy.
For each outer fold choose lambda on the other cases by mean raw A/P Dice, with
ties preferring the smaller lambda. The graph modifies only A/P on fixed foreground.

## Endpoints and advancement

Primary descriptor endpoint: paired held-out cut MAE, new full descriptors versus
base, natural task. Also compare against existing MRI and shuffled new features.
Report raw fixed-support A/P Dice, swaps, helped/harmed cases, and all displacement
strata. Paired bootstrap: 10000 case draws, fixed seed. Variants are never resampled
as independent cases. Report fixed-fold results as a stability diagnostic.

Advance the contextual affinity branch only if it improves natural cut MAE by at
least 0.1 mm versus both base and existing MRI, the paired interval versus base
excludes zero, Dice does not decrease, shuffled features do not explain the gain,
and synthetic recovery improves in both displacement directions without worsening
correct-cut preservation relative to the base readout. These are screening thresholds, not claims of clinical
importance. Positive findings require confirmation with out-of-fold backbone
features and then the augmented model; do not open development cases as a shortcut.

Graph correction advances only if the fold-selected policy beats original raw Dice
by at least 0.1 percentage points with a paired interval excluding zero and improves
cut MAE without more harmed than helped cases. Otherwise keep it a negative control.

Do not launch full structured-network training merely because graph mathematics
works. A failed information/correction screen stops this staged exploration before
costly training; a separate structured-training proposal would require evidence
against conditional CE and the existing supervised cut posterior.

Save source, protocol, checkpoint, split, native input and cache hashes, per-case
results and aggregate decisions. Do not replace original labels, change existing
experiments, or interpret repeated-use data as an untouched test set.
