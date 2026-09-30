# Whole-graph local A/P landmark probe — fixed before fitting

This is an exploratory experiment on the already inspected MSD fold 0, not a
new independent confirmation. It uses 208 training cases and then 52 validation
cases. No participant linkage is assumed. No neural network is trained.

## Question and input boundary

Does local structure of the whole hippocampus locate the A/P transition beyond
simple position and ordinary shape? Extract features only from whole-mask graph
maps produced by the prior property inventory. Never read anterior/posterior map
archives. No semantic labels, fitted A/P cuts, logits, or MRI enter feature
extraction. Reference labels enter only targets and scoring; reference support
is a best-case geometric input, not an inference-time known foreground.

Candidate cuts are coronal planes leaving 10–90% of whole-mask voxels on each
side. Target: closest coronal plane to the released labels, anterior at higher
RAS Y; choose the lower cut in a tie. Report target candidate coverage without
changing the candidate rule. Errors are millimeters on the verified 1-mm grid.

## Descriptors

- Position: normalized RAS Y and posterior volume fraction, with powers 1–3.
- Shape: coronal area, perimeter/area, LR/SI extent, normalized cut; their local
  values, signed derivatives, and change across a +/-2-slice neighborhood.
- Graph: degree distribution, graph depth, digital local thickness, sampled
  betweenness, eccentricity, slice components/branching, and geodesic-bin
  occupancy, cut size, degree-six fraction, thickness, and branching. Local
  intrinsic coordinate summaries are included. Geodesic coordinate is
  d(posterior diameter end)/(d(posterior end)+d(anterior end)); use only finite
  nodes, preserving validity fractions. Diameter ends are geometric proxies.
- Smooth scalar profiles with fixed Gaussian sigma=1 slice or geodesic bin.
  Use a support-relative coronal profile with four zero slices on each side
  to make smoothing independent of image padding. Use 32 geodesic bins.
  Report constant rule profiles and tied maxima, rather than treating every
  deterministic lower-index tie break as an identified landmark.
  Binned component/branch counts are discrete proxies,
  not a topologically verified medial skeleton or anatomical branch count.

Eight fixed rules are evaluated: area rise, thickness rise, degree-six rise,
betweenness peak, coronal branch peak, minimum geodesic normalized cut,
geodesic thickness rise, and standardized multifeature change magnitude.
Maxima/minima select candidates; ties choose lower Y. No reference-centered
search window is permitted.

## Learning and selection

Fourfold shuffled case-level CV on training only, seed 20260924. Each candidate
row stays with its case. StandardScaler + logistic regression (C=0.1, lbfgs,
max_iter=3000, tol=1e-7). Each case has total weight one, split equally between
the target candidate and all negatives. Choose the highest scoring candidate.

Arms: position; position+shape; position+graph; position+shape+graph; and a
negative control with position+shape unchanged and the graph feature rows
circularly shifted by a nonzero case-specific label-independent random offset.
Also evaluate training-median position and volume-fraction priors.

Select one learned arm (excluding shuffled control) and one fixed landmark
rule by training CV mean absolute error, with the listed order breaking ties.
Save all training OOF predictions and selection before extracting/evaluating
validation. Refit on all 208 training cases, then apply frozen models and priors
to reference validation graphs and both existing 52-case prediction graphs.
No validation-driven feature/parameter/model changes are allowed.

## Outcomes and interpretation

Report MAE, median and 95th-percentile error, exact/within-1/within-2-mm fractions,
per-case predictions, and paired case-bootstrap MAE differences (10,000 samples,
seed 20260924). These intervals are descriptive conditional on fitted models;
they omit model-refit and participant-level uncertainty.

Graph increment requires better performance than position+shape, not merely
better performance than a crude midpoint. A promising localization signal
requires >=0.25 mm lower MAE than shape in training CV, a paired interval below
zero, and improvement in at least 3/4 training folds. Validation assesses that
same frozen claim. This gate is an investigative decision, not a clinical margin.

Prediction-graph transfer is a separate robustness check: reference-trained
readouts face foreground domain shift. Compare with the original network cut,
the network's fitted plane, A/P Dice and swaps, and preservation of cases whose
original fitted plane was correct. Do not credit topology/geometry with the
network's existing image-derived localization. No automatic correction or
training loss is deployed. Negative results limit these descriptors and this
readout; they do not prove no graph landmark can exist.
