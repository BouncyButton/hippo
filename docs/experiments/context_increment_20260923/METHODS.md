# What the probe tests

The experiment asks whether a small readout can use MRI or decoder context to
place the A/P boundary more accurately **after it already sees the network's
predictions and the predicted hippocampal geometry**. It does not estimate
conditional mutual information or exhaust every nonlinear contextual model.

## A cut-ranking problem

For each hippocampal case, consider every interior coronal cut `c` through the
largest predicted foreground component. Below `c`, assign posterior (2); at and
above `c`, assign anterior (1). The reference target is the cut minimizing label
disagreement on the reference foreground, using the existing deterministic
first-cut tie rule. Reference labels are supervision only.

The original prediction's fitted cut is one input. Another input is the
conditional likelihood of each candidate. If

`q(v) = sigmoid(logit_anterior(v) - logit_posterior(v))`,

the candidate score supplied to the probe is

`[sum(y < c) log(1-q(v)) + sum(y >= c) log(q(v))] / |predicted foreground|`.

All sums here use predicted foreground. Probabilities are clipped to avoid
logarithms of zero. The remaining base inputs describe position, distance from
the original fitted cut, A/P extent, and probability/area/centroid/span on the
two neighboring slices. There are 24 base features per candidate.

## Added context

MRI context starts with six maps: normalized intensity; Gaussian mean and local
standard deviation at sigma 1 and 2 voxels; and Gaussian gradient magnitude at
sigma 1. Each map uses information from neighboring voxels in three dimensions.

The existing feature-pooling implementation extracts foreground mean/max,
superior exterior-band mean, and surrounding-context mean. It then represents
each candidate using its current value, adjacent-slice changes, and local
contrast. Six maps × four pooling regions × four operations gives 96 additional
features. These are normalized within the case without using its reference cut.

The decoder arm uses 384 cached features from the frozen full-resolution
decoder1 layer. Both checkpoint hashes were checked against the prior decoder
feature audit. All candidates and reference target definitions must match that
cache. The decoder features may encode learned context, geometry, and output
information; this experiment does not identify which of these a coefficient uses.

The shuffled-MRI arm applies one fixed, case-ID-derived permutation to candidate
rows of the added MRI features. Base features are untouched. This is a capacity
control with the same feature dimension and C grid, not a formal permutation test.

## Fitting and measurement

Standardization is fitted on each probe training split. Logistic regression
ranks cuts; its scores are not interpreted as calibrated boundary probabilities.
Within each case, the true cut receives weight 0.5, and the other candidates
share weight 0.5. Thus long hippocampi do not dominate the classifier simply by
having more candidates. Four case-level folds choose C by mean absolute cut error.

Selected cuts are rendered onto the **entire original predicted foreground**,
including any small components, and cannot repair foreground false positives or
false negatives. A/P Dice therefore isolates a fixed-support partition change.
The raw original segmentation and its fitted plane are both retained as controls.

The paired bootstrap resamples the same case indices for both methods in each
comparison, separately for each checkpoint. The two checkpoint results use the
same development cases and are not independent sample replications. See
[METHOD_NOTE.md](METHOD_NOTE.md) for the additional participant-grouping limitation.
