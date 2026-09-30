# Frozen E: remove presence only from rendering

Written before execution. User requested the proposed single mechanism test.
Use saved E on the same 208 internal outer cases, folds 1–4, 52 cases per fold.
No fitting, tuning, selection, new cohort claim or additional seed runs.

Run E once per case. Reuse its raw logits, fitted lower/upper boundaries, beta,
tau=0.5 and product-renderer implementation. Replace only log presence in the
renderer with exactly zero (presence exactly one). Do not refit boundaries,
change smoothing, alter the head or update parameters. Geometry retains the
historical influence of presence through the fit. This tests the explicit
rendering multiplier, not a model trained without a presence branch.

Primary diagnostics: restoration of raw-detected true thin rays deleted by E,
restoration of true overlap, and additional FP rays/voxels. Count ray presence
and true overlap separately. Report standard nonborder span<=2 thin definition
and an additional count-of-true-voxels 1–2 definition. Do not silently compare
denominators with earlier reports. Secondary: union and per-class Dice, ASSD,
HD95, true voxel recoveries, FN counts, and case/fold heterogeneity. Bootstrap
case differences conditional on fitted models; reused development labels limit
interpretation. No intervention selection on these results.

For remaining true rays without overlap, partition into: raw true overlap existed
(geometry suppresses it even at presence one); no raw true overlap but positive
geometric foreground correction on some true voxel; no raw true overlap and no
positive correction on any true voxel. These are endpoint descriptions, not
independent causal effects of geometry versus raw logits. The last two can have
joint deficits. Labels are used only for metrics and attribution.

Check original renderer exact reproduction, historical E metric reproduction,
unchanged foreground logits/classes, monotone foreground masks when presence
increases, unchanged checkpoints, source/data hashes, and complete nonduplicated
case coverage. Tests cover one-voxel rescue, residual geometric suppression,
label-independent intervention, and false presence versus actual overlap.
Store individual records only on the cluster. Export only aggregate reports
under the user's standing authorization for this investigation.
