# Seed 0 PCGrad, 50 training cases: anatomical and constraint audit

**The 50-case solution is substantially more anatomically plausible than the earlier 10-case PCGrad solution, and outperforms the earlier 10-case summed-loss model on this validation fold. It does not improve every constraint, and substantial train–validation gaps remain.**

Training job 676272 completed successfully in 16m54s, stopping at epoch 60 (3,000 updates). The unchanged selected checkpoint is epoch 30 (1,500 updates). Supplementary job 676288 performed inference only on three selected checkpoints; no model was trained or modified.

## Definitions and verification

Foreground is the union of anterior/posterior classes: y=1[label>0], p=P(class 1)+P(class 2), H=1[argmax of all three classes>0]. H is not obtained by thresholding p at 0.5. Inner and outer bands are the foreground shell removed by two 6-neighbor erosions and the background shell added by two dilations. Face pairs are counted once along positive axes, with both endpoints in these bands. These are local boundary-shell constraints, not global hippocampal connectivity or anterior/posterior anatomical constraints.

- Inner/outer hard equality satisfaction = mean 1[H_i=H_j] = 1−disagreement. Both-wrong pairs satisfy equality. Pair correctness = mean 1[H_i=y_i and H_j=y_j]; we report it separately.
- Crossing correctness is `correct_transition` = mean 1[H_i−H_j=y_i−y_j] on crossing ground-truth faces. It is not its complement, not Dice, and not 1−MSE.
- Soft edge error = mean ((p_i−p_j)−(y_i−y_j))² per pair group. The separated edge objective gives the three group means equal weight. Low same-side MSE can occur with an incorrectly constant foreground prediction.
- Bands use the original zero-gamma, zero-degree balanced BCE: B=(mean_inner[−log p]+mean_outer[−log(1−p)])/2 (implementation denominator epsilon 1e−6). Existing fuzzy truth is exp(−B) per case, averaged after exponentiation. It is a confidence score, not a percentage of correct voxels or edges. The existing binary case-level flag is exp(−B)≥0.90.
- Hard macro Dice averages anterior/posterior class Dice within each case; union Dice measures the foreground envelope. Logged training `train_dice` is a soft loss measured during updates and is not used as audited hard Dice.
- Connectivity is computed on the predicted foreground with 6-neighbor adjacency. Ground truth is not always a single component, so one component is supportive but not proof of perfect anatomy.
- Supplementary symmetric surface distances concatenate both directional distances between 6-neighbor surface voxels. The mean and 95th percentile are reported in evaluation-grid voxels, not millimetres; no patient-specific spacing claim is made.

All original selected-case hashes and selected-checkpoint hashes were checked; both frozen source manifests were verified and shared source files are byte-identical. Supplementary inference used the original Torch/MONAI/NumPy versions and CUDA AMP. Original saved edge/Dice values remain primary; newly computed bands and surface measurements are separate. Drift is quantified in AUDIT_SUMMARY.json. Cases are averaged equally; repeated cases or epochs are not independent samples.

## Selected epoch 30: train and validation

| Metric | Train (50) | Validation (52) | Train − validation |
|---|---:|---:|---:|
| Macro Dice (%) | 94.6433 | 85.3531 | 9.2902 |
| Union Dice (%) | 94.7294 | 87.9573 | 6.7721 |
| Inner pair correctness (%) | 91.6201 | 78.5361 | 13.0840 |
| Outer pair correctness (%) | 92.2128 | 84.9777 | 7.2351 |
| Correct crossing transition (%) | 63.5360 | 40.2760 | 23.2600 |
| Inner equality satisfaction (%) | 92.8668 | 87.8014 | 5.0653 |
| Outer equality satisfaction (%) | 93.3847 | 90.2136 | 3.1711 |
| Bands fuzzy truth (0–1) | 0.8378 | 0.5896 | 0.2482 |
| Bands inner BCE | 0.1887 | 0.6818 | -0.4931 |
| Bands outer BCE | 0.1672 | 0.4082 | -0.2410 |
| Bands balanced BCE | 0.1780 | 0.5450 | -0.3670 |
| Bands confidence-adherent cases (%) | 0.0000 | 0.0000 | 0.0000 |
| Inner soft MSE | 0.0434 | 0.0762 | -0.0327 |
| Outer soft MSE | 0.0436 | 0.0641 | -0.0205 |
| Crossing soft MSE | 0.3145 | 0.5356 | -0.2212 |
| Balanced edge soft MSE | 0.1338 | 0.2253 | -0.0915 |

Gaps in percentage rows are percentage points; fuzzy-truth gaps are score units. For BCE/MSE, lower is better, so negative train−validation indicates lower error on training.

Bands confidence improves substantially overall, but its inner and outer terms move differently: compared with the original ten-case models, inner BCE worsens while outer BCE improves. The existing confidence-adherent flag is zero for every selected case in all three runs because none reaches the predefined 0.90 threshold. This is a floor effect of a case-level confidence criterion, not zero correctly classified voxels.

## Validation: same 52 cases

| Metric | 10-case sum (epoch 64) | 10-case PCGrad (epoch 61) | 50-case PCGrad (epoch 30) |
|---|---:|---:|---:|
| Macro Dice (%) | 77.4542 | 52.6761 | 85.3531 |
| Union Dice (%) | 80.1058 | 47.4598 | 87.9573 |
| Inner pair correctness (%) | 84.9827 | 98.0165 | 78.5361 |
| Outer pair correctness (%) | 59.3204 | 7.1784 | 84.9777 |
| Correct crossing transition (%) | 23.2591 | 2.2286 | 40.2760 |
| Bands fuzzy truth | 0.5462 | 0.1818 | 0.5896 |
| Bands balanced BCE | 0.6110 | 1.7269 | 0.5450 |
| False positives per case | 1186.9038 | 7307.2500 | 364.7115 |
| False negatives per case | 316.3077 | 45.0000 | 431.0769 |
| Precision (%) | 71.8925 | 31.3815 | 88.7326 |
| Recall (%) | 90.8681 | 98.7687 | 87.3854 |
| Predicted/GT volume ratio | 1.2710 | 3.2124 | 0.9871 |
| Components per case | 2.6538 | 11.8846 | 1.0000 |
| Voxels outside largest component | 6.8654 | 122.2500 | 0.0000 |
| Enclosed background voxels | 0.0192 | 0.8654 | 0.0192 |
| Mean surface distance (voxels) | 0.8739 | 3.3060 | 0.5037 |
| Surface-distance 95th percentile (voxels) | 1.9866 | 7.1993 | 1.2707 |

## Training comparison on the identical original ten cases

| Metric | 10-case sum | 10-case PCGrad | 50-case PCGrad: original ten |
|---|---:|---:|---:|
| Macro Dice (%) | 85.3798 | 56.8363 | 94.1795 |
| Union Dice (%) | 85.3712 | 46.6152 | 94.2594 |
| Inner pair correctness (%) | 99.9514 | 100.0000 | 91.9460 |
| Outer pair correctness (%) | 52.9587 | 0.0000 | 91.2257 |
| Correct crossing transition (%) | 27.2942 | 0.0000 | 61.4641 |
| Inner equality satisfaction (%) | 99.9514 | 100.0000 | 93.1232 |
| Outer equality satisfaction (%) | 71.0869 | 100.0000 | 92.7677 |
| Bands fuzzy truth (0–1) | 0.6492 | 0.1266 | 0.8243 |
| Bands inner BCE | 0.0241 | 0.0023 | 0.1896 |
| Bands outer BCE | 0.8410 | 4.1478 | 0.1988 |
| Bands balanced BCE | 0.4325 | 2.0751 | 0.1942 |
| Bands confidence-adherent cases (%) | 0.0000 | 0.0000 | 0.0000 |
| Inner soft MSE | 0.0017 | 0.0000 | 0.0430 |
| Outer soft MSE | 0.0668 | 0.0007 | 0.0474 |
| Crossing soft MSE | 0.6091 | 0.9882 | 0.3378 |
| Balanced edge soft MSE | 0.2259 | 0.3296 | 0.1427 |

## Does training improvement extend to the added cases?

| Metric | Original ten | Added forty |
|---|---:|---:|
| Macro Dice (%) | 94.1795 | 94.7592 |
| Union Dice (%) | 94.2594 | 94.8469 |
| Inner pair correctness (%) | 91.9460 | 91.5387 |
| Outer pair correctness (%) | 91.2257 | 92.4596 |
| Correct crossing transition (%) | 61.4641 | 64.0540 |
| Inner equality satisfaction (%) | 93.1232 | 92.8027 |
| Outer equality satisfaction (%) | 92.7677 | 93.5390 |
| Bands fuzzy truth (0–1) | 0.8243 | 0.8412 |
| Bands inner BCE | 0.1896 | 0.1885 |
| Bands outer BCE | 0.1988 | 0.1593 |
| Bands balanced BCE | 0.1942 | 0.1739 |
| Bands confidence-adherent cases (%) | 0.0000 | 0.0000 |
| Inner soft MSE | 0.0430 | 0.0435 |
| Outer soft MSE | 0.0474 | 0.0427 |
| Crossing soft MSE | 0.3378 | 0.3086 |
| Balanced edge soft MSE | 0.1427 | 0.1316 |

## Train minus validation gaps (training cohort differs for the 50-case run)

| Metric | 10-case sum | 10-case PCGrad | 50-case PCGrad |
|---|---:|---:|---:|
| Macro Dice (pp) | 7.9255 | 4.1602 | 9.2902 |
| Inner correctness (pp) | 14.9687 | 1.9835 | 13.0840 |
| Outer correctness (pp) | -6.3617 | -7.1784 | 7.2351 |
| Crossing correctness (pp) | 4.0351 | -2.2286 | 23.2600 |
| Bands fuzzy truth (score units) | 0.1030 | -0.0552 | 0.2482 |

## Case-level consistency on validation

| Metric | Versus 10-case PCGrad: improved / worse / tied | Versus 10-case sum: improved / worse / tied |
|---|---:|---:|
| macro_dice | 52 / 0 / 0 | 51 / 1 / 0 |
| union_dice | 52 / 0 / 0 | 52 / 0 / 0 |
| inner_both_correct | 0 / 52 / 0 | 7 / 45 / 0 |
| outer_both_correct | 52 / 0 / 0 | 52 / 0 / 0 |
| cross_correct_transition | 52 / 0 / 0 | 51 / 1 / 0 |
| all_fp | 52 / 0 / 0 | 52 / 0 / 0 |
| all_fn | 0 / 52 / 0 | 9 / 43 / 0 |
| bands_case_loss | 52 / 0 / 0 | 33 / 19 / 0 |
| symmetric_mean_surface_distance_voxels | 52 / 0 / 0 | 52 / 0 / 0 |
| foreground_components_6 | 45 / 0 / 7 | 10 / 0 / 42 |

These are descriptive paired counts, not 52 independent experimental replications. A lower component count is counted here as a directional change toward one; component count alone is not an anatomical ground-truth metric.

## Evolution during training

Entries are train / validation percentages. These are full 50/52-case audits, unlike the earlier experiment’s two-case training probes.

| Epoch | Updates | Macro Dice | Inner correct | Outer correct | Crossing correct |
|---:|---:|---:|---:|---:|---:|
| 5 | 250 | 44.36 / 44.21 | 100.00 / 100.00 | 0.00 / 0.09 | 0.00 / 0.03 |
| 15 | 750 | 68.71 / 64.78 | 99.97 / 98.84 | 3.23 / 10.69 | 0.55 / 3.29 |
| 30 | 1500 | 94.64 / 85.35 | 91.62 / 78.54 | 92.21 / 84.98 | 63.54 / 40.28 |
| 45 | 2250 | 96.04 / 85.20 | 95.41 / 79.48 | 92.98 / 83.82 | 71.68 / 39.97 |
| 60 | 3000 | 96.14 / 85.17 | 95.51 / 79.28 | 93.19 / 83.93 | 72.31 / 39.90 |

At epoch 5, inner correctness is effectively perfect on both splits while outer/crossing correctness is near zero: this is extensive foreground overprediction, not complete anatomical learning. At epoch 15 it persists, with roughly 3,800 false positives per case. The major validation Dice jump occurs at epoch 20 (1,000 updates), from 65.59% at epoch 19 to 82.26%. By epoch 30 the foreground envelope is substantially corrected. Intermediate full anatomical audits are at epochs 15 and 30; they do not locate every boundary-metric change precisely at epoch 20.

From epoch 30 to 60, train Dice rises from 94.64% to 96.14% and train crossing correctness from 63.54% to 72.31%. Validation Dice changes from 85.35% to 85.17%, and crossing correctness from 40.28% to 39.90%. The crossing gap widens from 23.26 to 32.41 percentage points; the inner gap from 13.08 to 16.23. This is evidence of continued training fit without matching validation improvement. It supports a remaining generalization limitation, especially for exact boundary transitions, alongside ordinary segmentation overfitting. It does not establish a uniquely constraint-caused overfitting mechanism.

## Optimization and attribution

All 3,000 optimizer steps completed without skips; projection occurred at every update. Median projected/original gradient norm ratios by windows 1–15, 16–19, 20–30, 31–60 are 0.973, 0.786, 0.927, 0.573. These observations do not isolate the cause of the transition.

The new selected checkpoint has 1,500 updates versus 610 for the old PCGrad selected checkpoint (690 at its stop). The successful epoch-20 transition occurs after 1,000 updates, beyond the old run’s entire update budget. This is consistent with a data/update-budget limitation, but cannot distinguish it from effects of more diverse cases, the adaptive LR, or the five-epoch warmup now lasting 250 rather than 50 updates. The LR was still 1e−4 when Dice jumped.

## Interpretation

1. **Anatomically promising: yes, relative to the measured 10-case solutions.** The model predicts a much more accurate foreground envelope, with fewer false positives, better outer correctness and correct boundary transitions, and no detached foreground islands at the selected checkpoint. Both union and anterior/posterior macro Dice improve. This is evidence of improved segmentation geometry, not direct evidence of a learned internal anatomical representation.
2. **Every constraint learned better: no.** Crossing and outer hard correctness improve on both train and validation, but inner hard correctness decreases and false negatives increase. Same-side probability MSE is not equivalent to correctness: a constant overlarge foreground can have almost zero MSE. The report keeps equality satisfaction, correct labels, soft error and unary bands confidence separate.
3. **Constraints generalize perfectly: no.** The selected model retains meaningful train–validation gaps, especially in crossing correctness, and the gaps grow with later training. Better absolute validation performance coexists with a larger gap than the poorly fitting old PCGrad model.
4. **PCGrad is established as superior: no.** There is no matched 50-case summed-loss control, only one seed, and the reused validation set drove both LR adaptation and selection. More examples, more updates, and LR changes are confounded. The result shows that this PCGrad setup can reach a much better solution under the new regime; it does not identify PCGrad as the source of the advantage.
5. **No further experiment was launched.** The supplementary job only evaluated frozen selected checkpoints; original training outputs and checkpoints remain untouched.

Machine-readable selected, paired, temporal, optimization and provenance results: AUDIT_SUMMARY.json. Per-case measurements: AUDIT_CASES.json and AUDIT_TEMPORAL_CASES.json.

Maximum absolute supplementary-inference hard Dice drift across all three checkpoints is 0.04693 percentage points for an individual case; maximum crossing-correctness drift is 0.11503 percentage points. These small CUDA inference differences do not approach the measured effects.

![Training trajectory](/Users/filippofocaccia/Desktop/hippo/reports/pcgrad_50cases_seed0_20260929/audit_trajectory.png)
