# Selected-checkpoint constraint generalization audit: completed

Inference-only job **676160** completed successfully (exit 0) in 2 minutes 4 seconds, after PCGrad job 676122 completed. It trained no models, changed no selected checkpoints, and left the completed pooled/separated experiment results unchanged.

All six selected checkpoint, dataset, split and original source hashes were verified before recomputation. The six new bands-audit files match their completion-manifest hashes. The completed report preserves all 372 selected case-level observations, then averages cases within each seed and seeds equally. Intermediate per-case audits and supplementary historical baseline evidence are also retained.

Report: reports/separated_edge_generalization_20260929/REPORT.md.
Machine-readable summary: reports/separated_edge_generalization_20260929/SUMMARY.json.
Per-case results: reports/separated_edge_generalization_20260929/cases.json.

The original CUDA runtime reproduced saved hard Dice closely (mean absolute case drift 0.000678 percentage points, maximum 0.034318). The earlier local CPU benchmark was excluded because its different runtime produced materially different predictions.

The evidence shows a robust inner hard-correctness gap and bands confidence gaps on both sides. Outer hard edge metrics show a different pattern; crossing is not nearly solved on train. Separation did not meaningfully reduce the gaps. The historical baseline already has a substantial inner gap, so these results do not isolate causally constraint-specific overfitting. No further experiment has been launched.
