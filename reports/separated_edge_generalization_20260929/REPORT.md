# Do the constraints generalize?

**There is a large and reproducible inner-boundary correctness gap, but the evidence does not show that all constraints are learned nearly perfectly on training and then fail on validation.** Outer hard pair correctness and edge agreement are generally better on validation, **but both inner and outer bands confidence are worse on validation**. Crossing correctness remains low even on training. Separating the rules changes these gaps very little. The measurements support rule-dependent constraint-generalization problems coexisting with ordinary segmentation generalization error, rather than establishing that the constraint loss itself caused overfitting.

These are the original selected checkpoints of the **fold-0, three-seed, 75-epoch-cap** experiment. The selected epochs are pooled **73, 63, 74** and separated **73, 67, 73** for seeds 0, 1, 2. Both arms stopped at **75, 71, 75** respectively; neither a fixed epoch-75 comparison nor a new checkpoint selection is being presented.

## Mean across seeds

Entries are **train / validation / train-minus-validation gap**. Correctness, agreement, Dice and bands scores are multiplied by 100; gaps are percentage points for fractions and score points for continuous fuzzy truth. The bands fuzzy truth is a continuous score; case adherence is the percentage of cases passing the implementation's predefined threshold. They are not edge-correctness percentages.

| Metric | Pooled | Separated |
| --- | --- | --- |
| Inner pair correctness | 99.920480 / 85.495579 / 14.424902 | 99.926806 / 85.552757 / 14.374048 |
| Inner equality satisfaction (agreement) | 99.924505 / 92.700539 / 7.223966 | 99.930181 / 92.734832 / 7.195350 |
| Outer pair correctness | 54.699127 / 60.237922 / -5.538795 | 55.000073 / 60.171538 / -5.171465 |
| Outer equality satisfaction (agreement) | 71.637232 / 82.763320 / -11.126089 | 71.628813 / 82.687128 / -11.058314 |
| Crossing correct_transition | 29.297945 / 24.873928 / 4.424017 | 29.543952 / 24.862338 / 4.681613 |
| Bands fuzzy truth score | 64.882573 / 52.698920 / 12.183653 | 64.959884 / 52.533620 / 12.426265 |
| Bands case adherence (truth >=0.90) | 0.000000 / 0.000000 / 0.000000 | 0.000000 / 0.000000 / 0.000000 |
| Macro Dice | 85.932425 / 77.775051 / 8.157374 | 86.028516 / 77.859763 / 8.168753 |
| Foreground-union Dice | 85.962584 / 80.416390 / 5.546194 | 86.067382 / 80.464410 / 5.602972 |

The previously quoted **99.92–99.93% train versus about 85.5% validation is correct for inner *pair correctness***: pooled **99.920480% versus 85.495579%**, separated **99.926806% versus 85.552757%**. These are means of case means for the three selected checkpoints above, specifically the new fold-0 pooled/separated comparison, not the earlier nine-run, three-fold audit. They are not deep-interior voxel accuracy, the edge MSE, or literal same-side equality satisfaction. Both-wrong neighbors satisfy equality but are not correct. On validation, **7.204960%** of pooled inner pairs and **7.182075%** of separated inner pairs are coherently wrong; this accounts for much of the difference between agreement and correctness.


The original bands confidence-adherence threshold (truth >=0.90) is met by **zero cases in either split in every seed**. That binary metric is at its floor and cannot distinguish these arms or their generalization; the continuous bands truth and raw side-specific BCE remain informative.

For bands specifically, mean inner BCE is 0.021676 train versus 0.195053 validation for pooled, and 0.021010 versus 0.195302 for separated. Outer BCE is 0.854791 versus 1.113068, and 0.852707 versus 1.119195, respectively. Both sides have worse validation BCE in every seed. The outer bands confidence gap therefore remains present even though outer pair agreement and hard pair correctness are better on validation on average.

## Answers to the seven questions

1. **Is the gap mainly inner?** Inner pair correctness has the largest consistent positive hard-correctness gap: 14.424902 pp pooled and 14.374048 pp separated. Crossing also has a positive mean gap (4.424017 / 4.681613 pp), but its sign depends on the seed. Outer correctness and agreement are better on validation on average, and outer probability MSE is lower on validation in every seed. Bands fuzzy-truth gaps are 12.183653 / 12.426265 score points; the separate inner and outer BCE/truth tables below show which side contributes. Both inner and outer bands BCE are worse on validation in every seed. Therefore the gap is **not only an inner-constraint problem**: the particularly large hard-correctness gap is inner, while confidence-based bands gaps extend to both sides. Better outer hard correctness/edge smoothness can coexist with worse outer unary confidence because these measures evaluate different properties; counts, pair variation and log-probability error are not interchangeable.

2. **Did separating the rules reduce gaps?** Only slightly for the main inner hard metrics. Inner-correctness gap shrinks by 0.050853 pp; inner-equality gap shrinks by 0.028617 pp. Crossing-correctness gap grows by 0.257596 pp; macro-Dice gap changes by +0.011379 pp. The bands-truth gap grows by 0.242612 score points, worsening in all three seeds. Inner soft-MSE train advantage actually becomes larger in all three seeds: validation error is higher in every seed, while mean training error is slightly lower (training error itself increases in seed 2). There is no broad or substantial generalization-gap reduction. Moving an already negative outer gap toward zero is not inherently a benefit: training and validation values must be inspected separately.

3. **Are constraints almost perfectly satisfied on train?** Only the inner hard metrics approach 100%. Pooled/separated outer pair correctness is 54.699127% / 55.000073%; outer equality satisfaction is 71.637232% / 71.628813%; crossing correctness is 29.297945% / 29.543952%. Crossing has substantial training soft error too (MSE 0.596285 / 0.594567). The claim that all constraints are solved on train is contradicted by the measurements. Crossing correctness is a strict exact-boundary-face measure, not the fraction of correctly segmented hippocampal voxels, so its low value is compatible with substantially higher Dice. Bands truth/adherence must likewise be read as their own confidence metrics, not as binary segmentation correctness.

4. **Is the gap disproportionately larger than Dice?** Inner pair-correctness gap (~14.4 pp) exceeds macro-Dice gap (~8.16 pp). However, **literal inner equality-satisfaction gap is only ~7.2 pp**, similar in size to macro Dice. Foreground-union Dice gaps are 5.546194 / 5.602972 pp. Macro Dice additionally penalizes anterior/posterior confusion, which the foreground-union edge rules do not address. Pair correctness also fails whenever either endpoint is wrong, which can amplify a regional voxel error compared with a whole-mask overlap measure. Different measures condition on different regions and count voxels, edges, classes or confidence differently; a larger numerical gap is descriptive, not proof of a separate overfitting mechanism.

5. **Does this establish constraint-specific overfitting?** No. It is consistent with localized failure to generalize inner correctness and probability consistency. There is also ordinary segmentation generalization error, and crossing still has substantial error on train. The temporal evidence below shows a regional tradeoff while validation Dice improves, so global Dice alone misses the deterioration. A supplementary historical Dice-only baseline on the same cases and matched settings already has a **10.733824 pp inner-correctness gap**, a **5.331831 pp inner-equality gap**, and a **5.773682 pp macro-Dice gap**. Its selected epochs are 73/74/71. Thus the inner discrepancy is not unique to explicit constraints. The constrained arms enlarge the gap while improving validation Dice and boundary accuracy, consistent with a changed error balance; a matched baseline trajectory, controlled train/validation anatomy distribution, or independent test cohort would be needed to isolate a causal mechanism. The historical comparison is supplementary, not mixed into the two-arm tables. Gradient conflict and this generalization gap are different observations; neither proves the other caused the measured deterioration. The rules are jointly consistent: ideal p_i=y_i makes all three edge MSE terms zero. The observed competition concerns optimization and the learned error balance, not logically incompatible constraints.

6. **At which epochs?** Pooled 73/63/74 and separated 73/67/73. Exact checkpoint hashes are retained in the machine-readable provenance; checkpoint selection was unchanged.

7. **Are conclusions seed-sensitive?** The positive inner-correctness, inner-equality, bands fuzzy-truth (both sides), and macro-Dice gaps occur in all three seeds. The direction of the crossing-correctness gap changes: seeds 0/1 favor train, seed 2 favors validation. Outer pair correctness favors validation in seeds 0/2 and very slightly favors train in seed 1. Separation slightly reduces the inner-correctness gap for seeds 0/1 but increases it for seed 2. Consequently, the large inner gap is robust across these initializations; a general claim that separation improves constraint generalization is not. All individual seed values, including bands, appear in the tables below.

## When does the discrepancy develop?

The complete hard train–validation trajectory is unavailable: only validation was audited at intermediate checkpoints, and those weights were not retained. Full-training hard-gap onset cannot be reconstructed without retraining, which was not performed.

The available evidence nevertheless shows that the inner problem is **not confined to the selected checkpoint**. From epoch 15 to 60, validation inner correctness falls in every seed of both arms while validation Dice and crossing correctness rise in every seed. Means change from **96.19% to 86.68%** (pooled inner correctness) and **96.01% to 86.67%** (separated), while macro Dice improves from **53.58% to 77.39%** and **54.00% to 77.26%**. Some early predictions obtain high inner correctness by predicting too much foreground; inner correctness must therefore be considered together with outer and crossing accuracy.

Two fixed training cases have comparable *soft* rule audits at epochs 5/15/30/45/60. The inner MSE train–validation difference is small at epoch 5 (~0.00036), then widens progressively: training-probe inner MSE stays around 0.002 while validation rises to ~0.018 by epoch 60. Crossing MSE also develops a growing training advantage, although both splits improve; outer MSE becomes worse on train than validation. This supports gradually diverging, rule-dependent behavior rather than a uniform optimization failure. It is limited to two training cases and different probe/inference precision; the precision effect is not isolated, so this remains secondary evidence and cannot substitute for a full-cohort hard-gap timeline.

![Validation trajectories](/Users/filippofocaccia/Desktop/hippo/reports/separated_edge_generalization_20260929/validation_trajectories.png)

![Limited soft trajectories](/Users/filippofocaccia/Desktop/hippo/reports/separated_edge_generalization_20260929/soft_probe_trajectories.png)

The most defensible conclusion is **rule-dependent generalization deterioration plus optimization tradeoffs**, with insufficient evidence to label it causally constraint-specific overfitting. Separation alone barely changes that conclusion. No future experiment or additional training was launched as part of this analysis.


---

# Selected-checkpoint constraint generalization audit

This is a descriptive analysis of the completed fold-0, three-seed experiment, using the original selected checkpoints. No training, checkpoint reselection, additional folds, or changes to prior result files were performed.

## Definitions and aggregation

Let y_i=1 for either annotated hippocampal class and 0 for background, p_i=P(anterior)+P(posterior), and h_i=1[argmax of the three class probabilities is anterior or posterior]. **The hard decoder is not a threshold of p_i at 0.5.** The inner band I is foreground removed by two six-connected erosions; the outer band O is background added by two six-connected dilations. Every positive-axis face adjacency is counted once, with both endpoints in I union O. E_in has both endpoints in I; E_out both in O; E_cross connects I and O.

- **Same-side hard equality satisfaction (agreement):** mean over E_in or E_out of 1[h_i=h_j]. This is 1-disagreement. Both endpoints can be wrong and still satisfy this relation.
- **Inner/outer pair correctness:** mean of 1[h_i=y_i and h_j=y_j] over the appropriate edge group. Inner requires two foreground predictions; outer requires two background predictions. This is stricter than agreement. Agreement = both-correct + both-wrong; correctness + both-wrong + disagreement = 1.
- **Crossing hard satisfaction/correctness:** mean over E_cross of 1[h_i-h_j=y_i-y_j]. This is exactly the saved `cross_correct_transition`, also equal to crossing both-correct. It is **not its complement**. Its complement is transition failure; an inverted transition is a subset of failures. This tests the exact annotated boundary face and orientation, so a roughly 25% rate does not mean only 25% of hippocampal voxels are correctly segmented; a small boundary displacement can fail many such faces while Dice remains high.
- **Soft edge error:** MSE_k = mean over E_k of ((p_i-p_j)-(y_i-y_j))^2. On same-side edges this is probability-difference MSE; on crossing edges it is signed-contrast MSE. Lower is better. No hard percentage is inferred from it. Hard squared error is another metric: a reversed crossing contributes 4, so it is not the complement of correct_transition.
- **Bands:** the frozen `OuterBoundaryBandLoss` uses B_in=mean_I[-log p_i], B_out=mean_O[-log(1-p_i)] and B=(B_in+B_out)/2 (implementation denominator count+1e-6). Focal and degree weights are zero in these runs. Its existing fuzzy truth is T=exp(-B), with side truths exp(-B_in) and exp(-B_out). Its existing **case-level** binary `confidence_adherent` is 1[T >= 0.90]. This is a predefined confidence threshold, not voxel classification or pair correctness. Reported adherence is the percentage of cases passing it. Fuzzy truth multiplied by 100 is a score, not a percent of correct edges. Mean case truth is computed before averaging, not exp(-mean B). B is unweighted BCE; the tiny weighted bands contribution in training logs is not this raw loss.
- **Dice:** hard macro Dice averages anterior and posterior class Dice equally per case. Binary union Dice is supplied separately for context; neither is the training soft Dice loss (which includes background).

Every selected-case metric is evaluated with the same definition on train and validation. Each seed first averages 10 train or 52 validation cases equally, then the three seed means are averaged equally. Edge/voxel counts do not weight patients. Gap always means train minus validation; positive gaps favor train for correctness/truth/Dice, but **negative gaps favor train for losses/disagreement**. Percentage metrics have gaps in percentage points (pp); fuzzy truths scaled by100 have gaps in score points. No repeated case, seed, or epoch is treated as an independent sample for a confidence claim. The same validation set selected checkpoints; this is not an independent test-set estimate.

## Selected checkpoints


| Arm | Seed | Selected epoch | Stopped epoch |
| --- | --- | --- | --- |
| pooled | 0 | 73 | 75 |
| pooled | 1 | 63 | 71 |
| pooled | 2 | 74 | 75 |
| separated | 0 | 73 | 75 |
| separated | 1 | 67 | 71 |
| separated | 2 | 73 | 75 |

## Pooled: selected checkpoint values

All values below are percentages or fuzzy-truth score times100; gaps are percentage points or fuzzy score points, respectively. Bands adherence is case-level, not edge-level.

| Constraint / metric | Seed | Train | Validation | Gap |
| --- | --- | --- | --- | --- |
| Inner pair correctness | 0 | 99.965771 | 83.858404 | +16.107367 |
| Inner pair correctness | 1 | 99.829743 | 83.955919 | +15.873823 |
| Inner pair correctness | 2 | 99.965927 | 88.672412 | +11.293514 |
| Inner pair correctness | Mean | 99.920480 | 85.495579 | +14.424902 |
| Inner equality satisfaction | 0 | 99.965771 | 92.361563 | +7.604208 |
| Inner equality satisfaction | 1 | 99.839689 | 91.716425 | +8.123264 |
| Inner equality satisfaction | 2 | 99.968056 | 94.023629 | +5.944427 |
| Inner equality satisfaction | Mean | 99.924505 | 92.700539 | +7.223966 |
| Outer pair correctness | 0 | 54.139825 | 60.580150 | -6.440324 |
| Outer pair correctness | 1 | 66.358280 | 65.911344 | +0.446936 |
| Outer pair correctness | 2 | 43.599275 | 54.222271 | -10.622995 |
| Outer pair correctness | Mean | 54.699127 | 60.237922 | -5.538795 |
| Outer equality satisfaction | 0 | 71.458429 | 82.692165 | -11.233736 |
| Outer equality satisfaction | 1 | 76.437719 | 83.920660 | -7.482941 |
| Outer equality satisfaction | 2 | 67.015547 | 81.677136 | -14.661589 |
| Outer equality satisfaction | Mean | 71.637232 | 82.763320 | -11.126089 |
| Crossing correct_transition | 0 | 28.466202 | 23.284981 | +5.181222 |
| Crossing correct_transition | 1 | 41.318667 | 28.198059 | +13.120609 |
| Crossing correct_transition | 2 | 18.108966 | 23.138744 | -5.029778 |
| Crossing correct_transition | Mean | 29.297945 | 24.873928 | +4.424017 |
| Bands fuzzy truth score | 0 | 65.714067 | 55.021083 | +10.692984 |
| Bands fuzzy truth score | 1 | 72.267677 | 57.154857 | +15.112820 |
| Bands fuzzy truth score | 2 | 56.665974 | 45.920819 | +10.745155 |
| Bands fuzzy truth score | Mean | 64.882573 | 52.698920 | +12.183653 |
| Bands case adherence >=0.90 | 0 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | 1 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | 2 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | Mean | 0.000000 | 0.000000 | +0.000000 |
| Dice (macro) | 0 | 85.739919 | 77.525262 | +8.214657 |
| Dice (macro) | 1 | 89.233357 | 79.539364 | +9.693992 |
| Dice (macro) | 2 | 82.823998 | 76.260526 | +6.563472 |
| Dice (macro) | Mean | 85.932425 | 77.775051 | +8.157374 |

## Separated: selected checkpoint values

All values below are percentages or fuzzy-truth score times100; gaps are percentage points or fuzzy score points, respectively. Bands adherence is case-level, not edge-level.

| Constraint / metric | Seed | Train | Validation | Gap |
| --- | --- | --- | --- | --- |
| Inner pair correctness | 0 | 99.973769 | 84.080579 | +15.893190 |
| Inner pair correctness | 1 | 99.835746 | 84.039855 | +15.795890 |
| Inner pair correctness | 2 | 99.970902 | 88.537837 | +11.433065 |
| Inner pair correctness | Mean | 99.926806 | 85.552757 | +14.374048 |
| Inner equality satisfaction | 0 | 99.973769 | 92.444068 | +7.529701 |
| Inner equality satisfaction | 1 | 99.843744 | 91.772671 | +8.071073 |
| Inner equality satisfaction | 2 | 99.973031 | 93.987757 | +5.985274 |
| Inner equality satisfaction | Mean | 99.930181 | 92.734832 | +7.195350 |
| Outer pair correctness | 0 | 54.150981 | 60.339465 | -6.188484 |
| Outer pair correctness | 1 | 66.412811 | 65.811212 | +0.601599 |
| Outer pair correctness | 2 | 44.436426 | 54.363937 | -9.927511 |
| Outer pair correctness | Mean | 55.000073 | 60.171538 | -5.171465 |
| Outer equality satisfaction | 0 | 71.371361 | 82.608736 | -11.237375 |
| Outer equality satisfaction | 1 | 76.325229 | 83.823080 | -7.497850 |
| Outer equality satisfaction | 2 | 67.189849 | 81.629567 | -14.439718 |
| Outer equality satisfaction | Mean | 71.628813 | 82.687128 | -11.058314 |
| Crossing correct_transition | 0 | 28.360381 | 23.405103 | +4.955278 |
| Crossing correct_transition | 1 | 41.237508 | 28.048478 | +13.189031 |
| Crossing correct_transition | 2 | 19.033966 | 23.133435 | -4.099469 |
| Crossing correct_transition | Mean | 29.543952 | 24.862338 | +4.681613 |
| Bands fuzzy truth score | 0 | 65.637002 | 54.891952 | +10.745050 |
| Bands fuzzy truth score | 1 | 72.304355 | 56.822766 | +15.481589 |
| Bands fuzzy truth score | 2 | 56.938297 | 45.886142 | +11.052155 |
| Bands fuzzy truth score | Mean | 64.959884 | 52.533620 | +12.426265 |
| Bands case adherence >=0.90 | 0 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | 1 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | 2 | 0.000000 | 0.000000 | +0.000000 |
| Bands case adherence >=0.90 | Mean | 0.000000 | 0.000000 | +0.000000 |
| Dice (macro) | 0 | 85.749417 | 77.541545 | +8.207872 |
| Dice (macro) | 1 | 89.265688 | 79.587062 | +9.678626 |
| Dice (macro) | 2 | 83.070443 | 76.450682 | +6.619760 |
| Dice (macro) | Mean | 86.028516 | 77.859763 | +8.168753 |

## Soft losses and bands side metrics

Each entry is **train / validation / gap**. These are raw losses or fuzzy truths; MSE and BCE are not hard satisfaction fractions. Lower loss is better.

### Pooled

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| Inner edge MSE | 0.001682 / 0.019303 / -0.017621 | 0.002692 / 0.027009 / -0.024318 | 0.001100 / 0.012131 / -0.011031 | 0.001825 / 0.019481 / -0.017657 |
| Outer edge MSE | 0.066694 / 0.041022 / 0.025672 | 0.086051 / 0.059027 / 0.027024 | 0.048970 / 0.029173 / 0.019797 | 0.067239 / 0.043074 / 0.024164 |
| Crossing signed MSE | 0.598950 / 0.715868 / -0.116917 | 0.507524 / 0.652159 / -0.144635 | 0.682380 / 0.749109 / -0.066728 | 0.596285 / 0.705712 / -0.109427 |
| Bands inner BCE | 0.024232 / 0.215253 / -0.191022 | 0.025184 / 0.242266 / -0.217082 | 0.015611 / 0.127638 / -0.112027 | 0.021676 / 0.195053 / -0.173377 |
| Bands outer BCE | 0.816512 / 0.991742 / -0.175230 | 0.625955 / 0.893310 / -0.267355 | 1.121905 / 1.454152 / -0.332247 | 0.854791 / 1.113068 / -0.258277 |
| Bands balanced BCE | 0.420372 / 0.603498 / -0.183126 | 0.325570 / 0.567788 / -0.242219 | 0.568758 / 0.790895 / -0.222137 | 0.438233 / 0.654060 / -0.215827 |
| Bands inner fuzzy truth | 0.976060 / 0.809686 / 0.166374 | 0.975131 / 0.789126 / 0.186005 | 0.984511 / 0.881288 / 0.103223 | 0.978568 / 0.826700 / 0.151868 |
| Bands outer fuzzy truth | 0.442897 / 0.380694 / 0.062202 | 0.536416 / 0.421868 / 0.114548 | 0.326658 / 0.245223 / 0.081435 | 0.435324 / 0.349262 / 0.086062 |
| Union Dice | 0.857237 / 0.801693 / 0.055543 | 0.892349 / 0.821850 / 0.070499 | 0.829292 / 0.788948 / 0.040344 | 0.859626 / 0.804164 / 0.055462 |

### Separated

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| Inner edge MSE | 0.001651 / 0.019514 / -0.017863 | 0.002540 / 0.027260 / -0.024720 | 0.001132 / 0.012427 / -0.011295 | 0.001774 / 0.019733 / -0.017959 |
| Outer edge MSE | 0.068909 / 0.041818 / 0.027091 | 0.088652 / 0.060588 / 0.028064 | 0.051081 / 0.030649 / 0.020432 | 0.069547 / 0.044352 / 0.025195 |
| Crossing signed MSE | 0.600182 / 0.714677 / -0.114495 | 0.505756 / 0.654165 / -0.148409 | 0.677763 / 0.748533 / -0.070769 | 0.594567 / 0.705791 / -0.111224 |
| Bands inner BCE | 0.023720 / 0.213920 / -0.190200 | 0.023888 / 0.243264 / -0.219376 | 0.015421 / 0.128721 / -0.113300 | 0.021010 / 0.195302 / -0.174292 |
| Bands outer BCE | 0.819324 / 0.998240 / -0.178917 | 0.626250 / 0.903859 / -0.277609 | 1.112547 / 1.455487 / -0.342940 | 0.852707 / 1.119195 / -0.266488 |
| Bands balanced BCE | 0.421522 / 0.606080 / -0.184558 | 0.325069 / 0.573561 / -0.248492 | 0.563984 / 0.792104 / -0.228120 | 0.436858 / 0.657249 / -0.220390 |
| Bands inner fuzzy truth | 0.976560 / 0.810723 / 0.165837 | 0.976396 / 0.788378 / 0.188018 | 0.984699 / 0.880371 / 0.104327 | 0.979218 / 0.826491 / 0.152727 |
| Bands outer fuzzy truth | 0.441610 / 0.378702 / 0.062908 | 0.536269 / 0.417655 / 0.118615 | 0.329755 / 0.245258 / 0.084497 | 0.435878 / 0.347205 / 0.088673 |
| Union Dice | 0.857460 / 0.801865 / 0.055595 | 0.892751 / 0.822252 / 0.070499 | 0.831811 / 0.789816 / 0.041995 | 0.860674 / 0.804644 / 0.056030 |

## Change when separating the rules

Separated minus pooled, averaging seeds. A smaller positive correctness gap is only beneficial if validation improves; reducing a gap by making training worse is not improved generalization.

| Metric | Change train (points) | Change validation (points) | Change gap (points) |
| --- | --- | --- | --- |
| Inner pair correctness | +0.006325 | +0.057178 | -0.050853 |
| Inner equality satisfaction | +0.005676 | +0.034293 | -0.028617 |
| Outer pair correctness | +0.300946 | -0.066384 | +0.367329 |
| Outer equality satisfaction | -0.008418 | -0.076193 | +0.067774 |
| Crossing correct_transition | +0.246007 | -0.011589 | +0.257596 |
| Bands fuzzy truth score | +0.077312 | -0.165300 | +0.242612 |
| Bands case adherence >=0.90 | +0.000000 | +0.000000 | +0.000000 |
| Dice (macro) | +0.096091 | +0.084712 | +0.011379 |

## Temporal evidence

Only validation has full-cohort hard audits at intermediate epochs. Intermediate model checkpoints were not retained, so full-training hard correctness cannot be reconstructed over time. The saved gradient audits provide exactly the same raw rule MSE definitions on **two fixed training cases**, in FP32 eval mode, at matching epochs; validation MSE uses the saved CUDA-AMP inference. This restricted comparison is informative about direction but is not a full-training trajectory or proof of when a full-cohort hard gap opens. Epoch-level training losses were measured during changing model updates; they are not substituted for same-checkpoint evaluations. All per-case temporal inputs and per-seed summaries are preserved. No trajectory p-values are computed.

### Pooled temporal means

Each MSE entry is **two-training-case mean / validation mean / gap**.

| Epoch | Inner MSE | Outer MSE | Crossing MSE | Val inner correct % | Val macro Dice % |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.007470 / 0.007833 / -0.000363 | 0.018309 / 0.019172 / -0.000864 | 0.839927 / 0.845997 / -0.006069 | 86.968 | 17.854 |
| 15 | 0.002246 / 0.005917 / -0.003671 | 0.031028 / 0.026500 / 0.004527 | 0.771111 / 0.792300 / -0.021189 | 96.189 | 53.576 |
| 30 | 0.001857 / 0.010036 / -0.008179 | 0.046878 / 0.032764 / 0.014114 | 0.701648 / 0.747888 / -0.046240 | 91.807 | 73.812 |
| 45 | 0.001868 / 0.014549 / -0.012681 | 0.058408 / 0.039019 / 0.019390 | 0.638644 / 0.722151 / -0.083507 | 88.547 | 76.781 |
| 60 | 0.001727 / 0.017780 / -0.016053 | 0.065153 / 0.042612 / 0.022540 | 0.604528 / 0.711769 / -0.107240 | 86.683 | 77.387 |

### Separated temporal means

Each MSE entry is **two-training-case mean / validation mean / gap**.

| Epoch | Inner MSE | Outer MSE | Crossing MSE | Val inner correct % | Val macro Dice % |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.007600 / 0.007961 / -0.000361 | 0.018568 / 0.019418 / -0.000850 | 0.838941 / 0.845135 / -0.006195 | 86.941 | 17.859 |
| 15 | 0.002263 / 0.006145 / -0.003882 | 0.032303 / 0.027428 / 0.004875 | 0.765420 / 0.789153 / -0.023733 | 96.011 | 54.001 |
| 30 | 0.001914 / 0.010090 / -0.008176 | 0.047892 / 0.033650 / 0.014241 | 0.695432 / 0.747371 / -0.051939 | 91.710 | 73.915 |
| 45 | 0.001892 / 0.014829 / -0.012937 | 0.060057 / 0.039739 / 0.020318 | 0.636294 / 0.721370 / -0.085076 | 88.430 | 76.818 |
| 60 | 0.001811 / 0.017822 / -0.016011 | 0.067483 / 0.043960 / 0.023523 | 0.605738 / 0.715201 / -0.109462 | 86.674 | 77.256 |

## Provenance and numerical limits

All 562 original result-manifest files were rehashed, and original source payload hashes were verified. Each selected audit matches its completion-manifest hash. All six downloaded checkpoint hashes and embedded run/epoch fields match their original completion manifests; dataset and split hashes match. Only retained selected weights were used. The original frozen dataset builder gives train and validation identical deterministic spatial preprocessing; augmentation is disabled. Recomputed bands metrics use the original frozen bands implementation and matching configuration, evaluated with the original CUDA-AMP model inference and FP32 bands computation for both splits. Original selected edge/Dice tables retain the original saved audit values. Recomputed full per-case edge/Dice metrics allow numerical-drift inspection. A local CPU-FP32/MONAI-version-mismatched benchmark showed substantial drift and was excluded; its diagnostic file is retained separately.

| Recomputed versus saved metric | Mean absolute difference (pp) | Maximum case absolute difference (pp) |
| --- | --- | --- |
| macro_dice | 0.000678 | 0.034318 |
| inner_both_correct | 0.000723 | 0.058377 |
| outer_both_correct | 0.001453 | 0.084429 |
| cross_correct_transition | 0.002669 | 0.099338 |

The CUDA reproduction differences are small relative to the approximately 10–15-point correctness/confidence gaps, but they are not zero: no bitwise-reproducibility claim is made, and tiny arm differences remain descriptive. Recomputed soft edge-MSE mean absolute differences are approximately 2.08e-7 (inner), 3.41e-7 (outer), and 2.08e-6 (crossing); maximum case differences are 9.63e-6, 1.14e-5, and 4.77e-5 respectively. `NUMERICAL_REPRODUCTION.json` preserves these checks. There were no previously saved selected-checkpoint bands outputs against which to compute direct bands drift.

The new `cases.json` preserves every original selected-case metric plus the separately recomputed bands metrics and drift checks. `bands_*_seed*.json` retain all inference outputs; `temporal_cases.json` preserves intermediate case-level evidence; `SUMMARY.json` contains full precision per-seed and mean gaps, arm differences, temporal summaries and checkpoint provenance. Source definitions: [edge audit](/Users/filippofocaccia/Desktop/hippo/reports/separated_edge_20260929/source/scripts/audit_edge_coherence.py), [original bands](/Users/filippofocaccia/Desktop/hippo/reports/separated_edge_20260929/source/thesis/new_constraints/bands/outer_boundary.py), and [separated edge loss](/Users/filippofocaccia/Desktop/hippo/reports/separated_edge_20260929/source/thesis/new_constraints/separated_edge.py).


---

# Supplementary historical Dice-only baseline

This context uses only the existing hash-verified coherence audit; it is not part of the requested two-arm tables. Same fold-0 case IDs (10 train, 52 validation), data/split hashes, runtime, no augmentation, 75-epoch cap, optimizer, learning-rate schedule and early-stopping settings were verified. Original configuration hashes match saved checkpoint bindings. Model construction and preprocessing functions are AST-identical; validation hard-Dice calculation is unchanged (the only evaluator difference is an inactive added constraint flag). Band/face geometry counts also match case by case. Baseline selected epochs are **73, 74, 71** for seeds 0, 1, 2. This remains a historical comparison at independently selected epochs, not an exact contemporaneous baseline rerun or a matched temporal comparison.

Values are percentages and gaps are train minus validation in pp.

| Metric | Seed | Train | Validation | Gap |
| --- | --- | --- | --- | --- |
| Inner pair correctness | 0 | 99.985637 | 86.681236 | 13.304402 |
| Inner pair correctness | 1 | 99.923868 | 85.760140 | 14.163728 |
| Inner pair correctness | 2 | 100.000000 | 95.266658 | 4.733342 |
| Inner pair correctness | Mean | 99.969835 | 89.236011 | 10.733824 |
| Inner equality satisfaction | 0 | 99.985637 | 93.746614 | 6.239023 |
| Inner equality satisfaction | 1 | 99.923868 | 92.705546 | 7.218321 |
| Inner equality satisfaction | 2 | 100.000000 | 97.461852 | 2.538148 |
| Inner equality satisfaction | Mean | 99.969835 | 94.638004 | 5.331831 |
| Outer pair correctness | 0 | 38.342057 | 50.452531 | -12.110474 |
| Outer pair correctness | 1 | 55.196748 | 59.553309 | -4.356561 |
| Outer pair correctness | 2 | 12.555419 | 30.494454 | -17.939035 |
| Outer pair correctness | Mean | 35.364742 | 46.833431 | -11.468690 |
| Outer equality satisfaction | 0 | 70.762498 | 82.631570 | -11.869072 |
| Outer equality satisfaction | 1 | 73.426517 | 83.362568 | -9.936051 |
| Outer equality satisfaction | 2 | 78.290178 | 83.086509 | -4.796331 |
| Outer equality satisfaction | Mean | 74.159731 | 83.026882 | -8.867151 |
| Crossing correct_transition | 0 | 16.189370 | 18.010074 | -1.820704 |
| Crossing correct_transition | 1 | 31.022460 | 24.605736 | 6.416723 |
| Crossing correct_transition | 2 | 2.232159 | 11.847124 | -9.614965 |
| Crossing correct_transition | Mean | 16.481330 | 18.154312 | -1.672982 |
| Macro Dice | 0 | 80.550744 | 74.136267 | 6.414477 |
| Macro Dice | 1 | 85.813609 | 77.285962 | 8.527647 |
| Macro Dice | 2 | 70.394980 | 68.016059 | 2.378922 |
| Macro Dice | Mean | 78.919778 | 73.146096 | 5.773682 |
The baseline already has an inner-correctness gap of **10.733824 pp** and an inner-equality gap of **5.331831 pp**, alongside a macro-Dice gap of **5.773682 pp**. This directly argues against interpreting the entire inner gap as unique to explicit constraints. The constrained arms enlarge the inner gap while improving validation Dice and boundary accuracy; the observations are consistent with a changed false-positive/false-negative balance and localized generalization difficulty. They do not isolate memorization caused by the constraint formulation. Baseline gaps vary considerably across seeds, especially seed 2. No new baseline inference or training was performed.
