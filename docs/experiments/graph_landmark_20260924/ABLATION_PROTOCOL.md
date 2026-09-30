# Supplementary training-only descriptor ablation

Written after the primary frozen training/validation experiment. This is an
explicitly post-primary diagnostic to understand the feature mixture, not a
new model-selection stage. Do not use validation cases in this script and do
not replace the primary selected model using these results.

Repeat the same four training folds and readout settings while retaining:
(1) all graph features, (2) intrinsic positions only, or (3) all except intrinsic
positions. Also drop each remaining group one at a time: local degree/depth/
thickness, traffic/eccentricity, branching proxies, and geodesic section profiles.
Position and shape remain in every arm. Groups are defined by the existing
feature names before fitting. Report all arms and paired error differences;
correlated features can compensate, so group ablation is not unique causal
attribution. This resolves whether the primary graph increment could merely
be an intrinsic-coordinate prior instead of a local structural cue.
