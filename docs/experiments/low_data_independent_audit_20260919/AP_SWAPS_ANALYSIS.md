# Fold-0 A/P swaps: training versus validation

Generated from frozen checkpoint evaluations by `analyze_ap_swaps.py`. No jobs, inference, or original experiment modifications.

## Scope and interpretation

Ten identical training cases and 52 identical validation cases across all seeds, arms and schedules. GT foreground totals are 32,517 training and 174,350 validation voxels. Source hashes and every case-level measurement are in AP_SWAPS_ANALYSIS.json. Assertions check case identities, class totals, confusion-matrix sums and reconstruction of reported Dice.

A/P swaps are true anterior predicted posterior or true posterior predicted anterior, excluding foreground misses and background false positives. The retained-foreground rate divides swaps by GT foreground minus foreground misses. This helps expose changes in foreground retention, but uses a different voxel support for each model; it is not a paired same-voxel cut comparison.

The maximum-200 runs stopped early: final epochs are shown below. Best checkpoints were selected for overall validation Dice, not minimum swaps. These are retrospective descriptive comparisons, not an independent held-out test of A/P model selection.

## Best checkpoints

Arrows are baseline → bands. Rate is validation swaps / retained true foreground.

| Max epochs | Seed | Epochs | Train swaps | Validation swaps | Δ swaps | Validation rate (%) | Cases fewer / tied / more swaps |
|---|---|---|---|---|---|---|---|
| 75 | 0 | 73 → 68 | 2 → 3 | 5,768 → 5,354 | -414 | 3.615 → 3.385 | 36 / 3 / 13 |
| 75 | 1 | 74 → 74 | 2 → 2 | 5,376 → 5,014 | -362 | 3.383 → 3.187 | 35 / 0 / 17 |
| 75 | 2 | 71 → 74 | 16 → 12 | 5,690 → 5,257 | -433 | 3.364 → 3.207 | 31 / 3 / 18 |
| 200 | 0 | 107 → 93 | 0 → 1 | 4,448 → 4,573 | +125 | 3.080 → 3.143 | 21 / 1 / 30 |
| 200 | 1 | 99 → 76 | 0 → 3 | 4,894 → 4,943 | +49 | 3.357 → 3.290 | 29 / 0 / 23 |
| 200 | 2 | 167 → 160 | 0 → 0 | 5,449 → 4,972 | -477 | 3.570 → 3.283 | 40 / 2 / 10 |

## Latest checkpoints

Arrows are baseline → bands. Rate is validation swaps / retained true foreground.

| Max epochs | Seed | Epochs | Train swaps | Validation swaps | Δ swaps | Validation rate (%) | Cases fewer / tied / more swaps |
|---|---|---|---|---|---|---|---|
| 75 | 0 | 75 → 75 | 2 → 3 | 5,770 → 5,364 | -406 | 3.616 → 3.397 | 40 / 0 / 12 |
| 75 | 1 | 75 → 75 | 2 → 2 | 5,384 → 5,036 | -348 | 3.384 → 3.197 | 33 / 2 / 17 |
| 75 | 2 | 75 → 75 | 14 → 10 | 5,702 → 5,259 | -443 | 3.368 → 3.204 | 30 / 2 / 20 |
| 200 | 0 | 160 → 160 | 0 → 0 | 4,360 → 4,441 | +81 | 3.098 → 3.169 | 25 / 2 / 25 |
| 200 | 1 | 160 → 160 | 0 → 0 | 4,956 → 4,880 | -76 | 3.443 → 3.369 | 22 / 3 / 27 |
| 200 | 2 | 169 → 160 | 0 → 0 | 5,480 → 4,972 | -508 | 3.591 → 3.283 | 41 / 1 / 10 |

## Every checkpoint: directions, coverage and diagnostic headroom

Swap-only oracle gain is a theoretical relabeling diagnostic, not a forecast of an achievable improvement. It leaves FP/FN unchanged and does not reconstruct or score an anatomical surface. All Dice gains below use the original macro-over-cases metric, not pooled-voxel Dice.

| Schedule | Seed | Checkpoint | Arm | Split | A→P | P→A | FN | Swap / GT (%) | Swap / retained GT (%) | Dice (%) | Swap-only oracle gain (pp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 75 | 0 | best | baseline | train | 0 | 2 | 6 | 0.0062 | 0.0062 | 80.557 | 0.005 |
| 75 | 0 | best | baseline | validation | 2,440 | 3,328 | 14,774 | 3.3083 | 3.6146 | 74.137 | 2.745 |
| 75 | 0 | best | bands | train | 0 | 3 | 11 | 0.0092 | 0.0092 | 84.160 | 0.008 |
| 75 | 0 | best | bands | validation | 2,038 | 3,316 | 16,170 | 3.0708 | 3.3848 | 76.856 | 2.655 |
| 75 | 0 | latest | baseline | train | 0 | 2 | 5 | 0.0062 | 0.0062 | 80.561 | 0.005 |
| 75 | 0 | latest | baseline | validation | 2,360 | 3,410 | 14,763 | 3.3094 | 3.6156 | 74.137 | 2.748 |
| 75 | 0 | latest | bands | train | 0 | 3 | 8 | 0.0092 | 0.0092 | 84.249 | 0.008 |
| 75 | 0 | latest | bands | validation | 2,025 | 3,339 | 16,427 | 3.0766 | 3.3966 | 76.772 | 2.664 |
| 75 | 1 | best | baseline | train | 1 | 1 | 24 | 0.0062 | 0.0062 | 85.808 | 0.005 |
| 75 | 1 | best | baseline | validation | 2,709 | 2,667 | 15,423 | 3.0835 | 3.3827 | 77.287 | 2.677 |
| 75 | 1 | best | bands | train | 1 | 1 | 38 | 0.0062 | 0.0062 | 88.525 | 0.005 |
| 75 | 1 | best | bands | validation | 2,350 | 2,664 | 17,000 | 2.8758 | 3.1865 | 79.144 | 2.568 |
| 75 | 1 | latest | baseline | train | 1 | 1 | 21 | 0.0062 | 0.0062 | 85.715 | 0.005 |
| 75 | 1 | latest | baseline | validation | 2,641 | 2,743 | 15,245 | 3.0880 | 3.3839 | 77.150 | 2.674 |
| 75 | 1 | latest | bands | train | 1 | 1 | 38 | 0.0062 | 0.0062 | 88.449 | 0.005 |
| 75 | 1 | latest | bands | validation | 2,371 | 2,665 | 16,840 | 2.8884 | 3.1973 | 79.112 | 2.577 |
| 75 | 2 | best | baseline | train | 1 | 15 | 0 | 0.0492 | 0.0492 | 70.395 | 0.034 |
| 75 | 2 | best | baseline | validation | 2,448 | 3,242 | 5,225 | 3.2636 | 3.3644 | 68.016 | 2.325 |
| 75 | 2 | best | bands | train | 0 | 12 | 5 | 0.0369 | 0.0369 | 80.832 | 0.028 |
| 75 | 2 | best | bands | validation | 1,815 | 3,442 | 10,414 | 3.0152 | 3.2067 | 75.244 | 2.460 |
| 75 | 2 | latest | baseline | train | 1 | 13 | 0 | 0.0431 | 0.0431 | 70.065 | 0.028 |
| 75 | 2 | latest | baseline | validation | 2,566 | 3,136 | 5,064 | 3.2704 | 3.3683 | 67.619 | 2.314 |
| 75 | 2 | latest | bands | train | 0 | 10 | 5 | 0.0308 | 0.0308 | 80.573 | 0.023 |
| 75 | 2 | latest | bands | validation | 1,875 | 3,384 | 10,200 | 3.0163 | 3.2038 | 75.031 | 2.451 |
| 200 | 0 | best | baseline | train | 0 | 0 | 149 | 0.0000 | 0.0000 | 96.136 | 0.000 |
| 200 | 0 | best | baseline | validation | 1,795 | 2,653 | 29,930 | 2.5512 | 3.0799 | 80.359 | 2.507 |
| 200 | 0 | best | bands | train | 0 | 1 | 153 | 0.0031 | 0.0031 | 95.559 | 0.003 |
| 200 | 0 | best | bands | validation | 1,666 | 2,907 | 28,859 | 2.6229 | 3.1431 | 80.532 | 2.578 |
| 200 | 0 | latest | baseline | train | 0 | 0 | 62 | 0.0000 | 0.0000 | 97.987 | 0.000 |
| 200 | 0 | latest | baseline | validation | 1,879 | 2,481 | 33,604 | 2.5007 | 3.0978 | 79.772 | 2.517 |
| 200 | 0 | latest | bands | train | 0 | 0 | 55 | 0.0000 | 0.0000 | 98.182 | 0.000 |
| 200 | 0 | latest | bands | validation | 1,810 | 2,631 | 34,198 | 2.5472 | 3.1687 | 79.633 | 2.568 |
| 200 | 1 | best | baseline | train | 0 | 0 | 153 | 0.0000 | 0.0000 | 96.975 | 0.000 |
| 200 | 1 | best | baseline | validation | 1,947 | 2,947 | 28,545 | 2.8070 | 3.3565 | 80.275 | 2.750 |
| 200 | 1 | best | bands | train | 3 | 0 | 150 | 0.0092 | 0.0093 | 95.322 | 0.009 |
| 200 | 1 | best | bands | validation | 2,248 | 2,695 | 24,129 | 2.8351 | 3.2905 | 80.669 | 2.705 |
| 200 | 1 | latest | baseline | train | 0 | 0 | 80 | 0.0000 | 0.0000 | 98.268 | 0.000 |
| 200 | 1 | latest | baseline | validation | 1,936 | 3,020 | 30,412 | 2.8426 | 3.4431 | 79.914 | 2.806 |
| 200 | 1 | latest | bands | train | 0 | 0 | 42 | 0.0000 | 0.0000 | 98.593 | 0.000 |
| 200 | 1 | latest | bands | validation | 1,991 | 2,889 | 29,514 | 2.7990 | 3.3693 | 80.218 | 2.754 |
| 200 | 2 | best | baseline | train | 0 | 0 | 3 | 0.0000 | 0.0000 | 89.084 | 0.000 |
| 200 | 2 | best | baseline | validation | 1,676 | 3,773 | 21,737 | 3.1253 | 3.5705 | 77.220 | 2.817 |
| 200 | 2 | best | bands | train | 0 | 0 | 5 | 0.0000 | 0.0000 | 92.839 | 0.000 |
| 200 | 2 | best | bands | validation | 1,446 | 3,526 | 22,923 | 2.8517 | 3.2834 | 78.871 | 2.638 |
| 200 | 2 | latest | baseline | train | 0 | 0 | 3 | 0.0000 | 0.0000 | 89.110 | 0.000 |
| 200 | 2 | latest | baseline | validation | 1,688 | 3,792 | 21,728 | 3.1431 | 3.5906 | 77.201 | 2.832 |
| 200 | 2 | latest | bands | train | 0 | 0 | 5 | 0.0000 | 0.0000 | 92.839 | 0.000 |
| 200 | 2 | latest | bands | validation | 1,446 | 3,526 | 22,923 | 2.8517 | 3.2834 | 78.871 | 2.638 |

## Findings and limits

- Training A/P assignment is already almost perfect on retained GT foreground; every final maximum-200 checkpoint has zero training swaps. This is not equivalent to perfect overall training segmentation.
- At the original 75-epoch best checkpoints, bands reduces validation swaps for all three seeds. However, it also increases foreground misses for all three. Lower retained-foreground rates do not rule out selective removal of difficult-to-label voxels.
- At maximum 200, best-checkpoint bands effects are mixed: seed 0 worsens both raw count and retained-foreground rate; seed 1 increases the raw count slightly while improving the rate and retaining more foreground; seed 2 improves both A/P measures but misses more foreground. Final checkpoints confirm that seed 0 is worse and seed 2 better; seed 1 has fewer swaps and a better rate.
- P→A exceeds A→P for every maximum-200 validation checkpoint. This is a net anterior overassignment within recovered true foreground, not proof of a consistently posterior-shifted anatomical plane. Missing foreground and false positives can alter full predicted class volumes.
- Longer training does not uniformly reduce A/P confusion after accounting for retained foreground: seed 2 baseline best-checkpoint retained rates rise from 3.364% at maximum 75 to 3.570% at maximum 200; bands rises from 3.207% to 3.283%. Both raw counts fall, but many more GT foreground voxels are missed. Different schedules and selected epochs prevent attributing this change to duration alone.
- Direction varies by case. For example, maximum-200 seed 2 baseline has more P→A than A→P swaps in 35 validation cases, but the reverse in 17. An aggregate directional imbalance is not evidence that a single global cut shift would fix all cases.
- The archived bands loss supervises grouped foreground versus background and is invariant under exchanging A/P labels. Any A/P improvement is indirect, through the shared network or changed foreground support; these experiments do not show a direct cut-placement penalty.
- Confusion matrices cannot establish cut displacement, tilt, roughness, disconnected islands, or error distance from the true interface. The local audited archives contain no spatial prediction volumes. Spatial claims require predictions and GT in the same physical coordinate system.
- Next diagnostic: evaluate A/P mistakes on GT foreground predicted as foreground by BOTH arms; separately track swap→correct, swap→background and background→swap transitions. Inspect all 52 cases, not only selected successes. With spatial masks, report signed AP displacement and surface distances of the internal interface, coverage/missing-interface failures, and errors near versus far from the true interface. Use plane-fit offset/tilt only if a plane is justified by the labeling protocol.
- Report paired case-level changes and seed-specific effects. Voxels are not independent samples; repeated seeds do not multiply the number of independent cases. Case identifiers alone do not establish independent subjects.

## Validation case concentration at best checkpoints

| Schedule | Seed | Arm | Cases with swaps | Top 5 cases’ share of swaps (%) | Median case retained rate (%) |
|---|---|---|---|---|---|
| 75 | 0 | baseline | 52/52 | 28.21 | 2.782 |
| 75 | 0 | bands | 52/52 | 29.08 | 2.469 |
| 75 | 1 | baseline | 52/52 | 30.97 | 2.814 |
| 75 | 1 | bands | 52/52 | 31.59 | 2.427 |
| 75 | 2 | baseline | 52/52 | 30.28 | 2.459 |
| 75 | 2 | bands | 52/52 | 30.09 | 2.630 |
| 200 | 0 | baseline | 52/52 | 29.47 | 2.136 |
| 200 | 0 | bands | 52/52 | 28.95 | 2.409 |
| 200 | 1 | baseline | 52/52 | 29.55 | 2.368 |
| 200 | 1 | bands | 51/52 | 30.18 | 2.563 |
| 200 | 2 | baseline | 52/52 | 28.17 | 2.729 |
| 200 | 2 | bands | 52/52 | 29.18 | 2.657 |
