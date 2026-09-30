# Degree-weighted bands: boundary-error audit

**Finding: neither degree-weighted variant reduced net foreground/background boundary mistakes relative to the early-stopped Dice baseline in this run.** Both corrected many baseline errors but introduced more new errors. Small A/P macro-Dice gains do not establish improved outer-boundary segmentation.

Audit job **667725 completed successfully** in 2m03s. Inference only: four selected checkpoints, the same 52 validation cases, fold 0, seed 0; no baseline retraining. The Dice reference uses the same mild augmentation, optimizer, input preprocessing and early-stopping policy. Its selected epoch is 24, after stopping at epoch 32. A and B selected epochs 22 and 30, stopping at 30 and 38.

## Boundary mistakes

Counts below are pooled over the ground-truth two-step six-connected inner and outer bands. FN means missing hippocampus; FP means adding foreground outside it. A/P swaps are counted separately.

| Model | Inner FN | Outer FP | FN + FP | Change vs Dice |
|---|---:|---:|---:|---:|
| Early-stopped Dice | 15,500 | 15,821 | 31,321 | +0 |
| Existing Dice + bands | 13,861 | 17,852 | 31,713 | +392 |
| A: inner normalization | 16,121 | 15,401 | 31,522 | +201 |
| B: surface normalization | 15,423 | 16,395 | 31,818 | +497 |

**A:** 420 fewer outer false positives, offset by 621 additional inner misses: **201 more boundary errors**. **B:** 77 fewer inner misses, offset by 574 additional outer false positives: **497 more boundary errors**.

| Compared with Dice | Baseline boundary errors corrected | New boundary errors introduced | Net corrected |
|---|---:|---:|---:|
| A | 5,125 | 5,326 | -201 |
| B | 4,308 | 4,805 | -497 |

These are paired voxel transitions, not differences inferred only from overall Dice. A recovered foreground voxel with the wrong A/P class counts as a union correction, but not a full-class correction. Including A/P swaps, A has 45 more errors inside the bands and B has 255 more than Dice.

## Where the errors occur

| Boundary layer | Dice errors | A errors | B errors |
|---|---:|---:|---:|
| inner layer1 | 14,786 | 15,287 | 14,668 |
| inner layer2 | 714 | 834 | 755 |
| outer layer1 | 14,662 | 14,405 | 15,332 |
| outer layer2 | 1,159 | 996 | 1,063 |

**99.51%** of the Dice model's foreground/background errors lie inside these bands (31,321/31,474). Only 153 lie beyond them: 44 missed foreground and 109 false positives. A and B have 132 and 128 foreground/background errors beyond the bands. This supports the user's observation that extending supervision farther from the boundary is not the principal unmet need.

The statement applies to the outer foreground/background boundary. A/P label swaps can lie deep inside the union: Dice has 1,174 swaps beyond the outer bands.

## Did exposed-voxel recovery improve?

Whole-foreground graph degree, including all GT foreground voxels. Percentages are pooled fractions predicted as background; swaps remain separate.

| GT degree | Voxels | Dice miss rate | Ordinary bands | A miss rate | B miss rate |
|---|---:|---:|---:|---:|---:|
| 0–1 | 286 | 81.82% | 82.87% | 82.87% | 83.92% |
| 2–3 | 21,061 | 40.69% | 36.40% | 40.97% | 39.64% |
| 4–5 | 50,186 | 11.92% | 10.64% | 12.80% | 12.11% |
| 6 | 102,817 | 0.74% | 0.61% | 0.86% | 0.76% |

The most exposed degree-0/1 group does not improve: 234 misses with Dice, 237 with A, 240 with B, out of 286 voxels. B reduces degree-2/3 misses by 221 versus Dice, but this is region-dependent: anterior improves by 279 while posterior worsens by 58. A does not improve any whole-foreground degree group in aggregate. Ordinary bands recover more foreground in the common degree groups, at the cost of more outer false positives.

## Segmentation and surface metrics

Equal-case means, not pooled voxel Dice. Surface distances use six-face surface voxels and pooled bidirectional distances in the 1-mm isotropic grid.

| Model | Macro A/P Dice | Union Dice | ASSD mm | HD95 mm | Surface Dice @1mm | A/P swaps, all voxels |
|---|---:|---:|---:|---:|---:|---:|
| Early-stopped Dice | 0.88766 | 0.90900 | 0.38888 | 1.09373 | 96.625% | 3,630 |
| Existing Dice + bands | 0.88769 | 0.90896 | 0.39041 | 1.08762 | 96.740% | 3,651 |
| A: inner normalization | 0.88794 | 0.90834 | 0.38855 | 1.08296 | 96.722% | 3,505 |
| B: surface normalization | 0.88818 | 0.90792 | 0.39338 | 1.07169 | 96.638% | 3,333 |

Both variants reduce A/P swaps but have lower union Dice than the Dice baseline. A is nearly unchanged on ASSD; B is worse. HD95 is slightly lower, but is tied with Dice in 49/52 cases for A and 47/52 for B, so its mean change reflects only a few cases.

## Paired uncertainty

Candidate minus Dice, with 10,000 paired case-bootstrap resamples. Error/distance increases are worse; Dice increases are better.

| Comparison | Metric | Mean change | Descriptive 95% interval |
|---|---|---:|---:|
| A vs Dice | macro_dice | +0.000275 | [-0.003492, +0.003996] |
| A vs Dice | whole/dice | -0.000654 | [-0.002609, +0.001361] |
| A vs Dice | boundary_union_errors | +3.865385 | [-8.942308, +16.307692] |
| A vs Dice | balanced_boundary_error | +0.001255 | [-0.001117, +0.003565] |
| B vs Dice | macro_dice | +0.000521 | [-0.002889, +0.003971] |
| B vs Dice | whole/dice | -0.001077 | [-0.002722, +0.000650] |
| B vs Dice | boundary_union_errors | +9.557692 | [-0.903846, +19.846154] |
| B vs Dice | balanced_boundary_error | +0.001289 | [-0.000713, +0.003233] |

All these intervals include zero. The results do not demonstrate a reliable improvement over Dice, nor establish a reproducible harm claim. They describe this single seed and repeatedly used validation fold; they do not account for experiment selection, training-seed variation or checkpoint selection.

Balanced boundary error is the equal-case mean of half the inner FN rate plus half the outer FP rate. It rises from 10.724% for Dice to 10.850% for A and 10.853% for B. Ordinary bands achieve 10.633%. A and B are worse than ordinary bands on this balanced metric; their descriptive paired intervals exclude zero, but this remains an exploratory development-fold result.

## Integrity and numerical repeatability

Source, checkpoint, export and input hashes were verified; the full selected-checkpoint metadata match each config. All models were evaluated on the same 3g.40gb GPU profile. Historical Dice and ordinary-bands training used 4g.40gb; that resource difference is recorded.

The first audit, 667718, stopped at an exact count comparison with the prior audit. The follow-up measured the discrepancy: A differs by one FN and B by one FP. Repeated inference within the completed audit changed two Dice predictions (one voxel in each of two cases) and zero predictions for ordinary bands, A and B. These are observed AMP-inference differences; their exact kernel-level cause was not established. The hundreds-of-voxel boundary differences above are not explained by this variation. Reported paired statistics use the first pass uniformly.

Three focused numerical tests passed: exact agreement with training-band construction, error-transition semantics, and perfect/one-error score behavior. Pooled correction-minus-regression counts were checked against direct error-count differences in every reported support.

## Interpretation

The motivating exposed-voxel failure remains real, but the tested alpha=2 degree-weighting formulations have not solved it. B has a small anterior degree-2/3 recovery signal; neither variant improves the rare degree-0/1 group, and both introduce more foreground/background boundary mistakes than they remove overall. This result concerns these loss formulations, strengths and calibration policy; it does not settle the separate hypothesis of explicit degree prediction.

Artifacts: [aggregate results and provenance](BOUNDARY_AUDIT_V2_RESULTS.json), [protocol](BOUNDARY_AUDIT_PROTOCOL.md), [submission receipt](BOUNDARY_AUDIT_V2_SUBMISSION.json). Per-case transitions remain on the cluster; predictions were kept only in memory and image arrays were not exported.
