"""Write a concise interpretation from verified measurement summaries."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent

def main():
    s=json.loads((OUT/'SUMMARY.json').read_text())
    a=s['selected'];changes=s['separated_minus_pooled']
    def v(arm,k,f):return a[arm][k][-1][f]
    def pct(x):return f'{100*x:.6f}'
    def trip(arm,k):return ' / '.join(pct(v(arm,k,f)) for f in ['train','validation','gap'])
    intro=f'''# Do the constraints generalize?

**There is a large and reproducible inner-boundary correctness gap, but the evidence does not show that all constraints are learned nearly perfectly on training and then fail on validation.** Outer hard pair correctness and edge agreement are generally better on validation, **but both inner and outer bands confidence are worse on validation**. Crossing correctness remains low even on training. Separating the rules changes these gaps very little. The measurements support rule-dependent constraint-generalization problems coexisting with ordinary segmentation generalization error, rather than establishing that the constraint loss itself caused overfitting.

These are the original selected checkpoints of the **fold-0, three-seed, 75-epoch-cap** experiment. The selected epochs are pooled **73, 63, 74** and separated **73, 67, 73** for seeds 0, 1, 2. Both arms stopped at **75, 71, 75** respectively; neither a fixed epoch-75 comparison nor a new checkpoint selection is being presented.

## Mean across seeds

Entries are **train / validation / train-minus-validation gap**. Correctness, agreement, Dice and bands scores are multiplied by 100; gaps are percentage points for fractions and score points for continuous fuzzy truth. The bands fuzzy truth is a continuous score; case adherence is the percentage of cases passing the implementation's predefined threshold. They are not edge-correctness percentages.

| Metric | Pooled | Separated |
| --- | --- | --- |
| Inner pair correctness | {trip('pooled','inner_both_correct')} | {trip('separated','inner_both_correct')} |
| Inner equality satisfaction (agreement) | {trip('pooled','inner_hard_agreement')} | {trip('separated','inner_hard_agreement')} |
| Outer pair correctness | {trip('pooled','outer_both_correct')} | {trip('separated','outer_both_correct')} |
| Outer equality satisfaction (agreement) | {trip('pooled','outer_hard_agreement')} | {trip('separated','outer_hard_agreement')} |
| Crossing correct_transition | {trip('pooled','cross_correct_transition')} | {trip('separated','cross_correct_transition')} |
| Bands fuzzy truth score | {trip('pooled','bands_truth')} | {trip('separated','bands_truth')} |
| Bands case adherence (truth >=0.90) | {trip('pooled','bands_confidence_adherent')} | {trip('separated','bands_confidence_adherent')} |
| Macro Dice | {trip('pooled','macro_dice')} | {trip('separated','macro_dice')} |
| Foreground-union Dice | {trip('pooled','union_dice')} | {trip('separated','union_dice')} |

The previously quoted **99.92–99.93% train versus about 85.5% validation is correct for inner *pair correctness***: pooled **{pct(v('pooled','inner_both_correct','train'))}% versus {pct(v('pooled','inner_both_correct','validation'))}%**, separated **{pct(v('separated','inner_both_correct','train'))}% versus {pct(v('separated','inner_both_correct','validation'))}%**. These are means of case means for the three selected checkpoints above, specifically the new fold-0 pooled/separated comparison, not the earlier nine-run, three-fold audit. They are not deep-interior voxel accuracy, the edge MSE, or literal same-side equality satisfaction. Both-wrong neighbors satisfy equality but are not correct. On validation, **{pct(v('pooled','inner_both_wrong','validation'))}%** of pooled inner pairs and **{pct(v('separated','inner_both_wrong','validation'))}%** of separated inner pairs are coherently wrong; this accounts for much of the difference between agreement and correctness.

## Answers to the seven questions

1. **Is the gap mainly inner?** Inner pair correctness has the largest consistent positive hard-correctness gap: {pct(v('pooled','inner_both_correct','gap'))} pp pooled and {pct(v('separated','inner_both_correct','gap'))} pp separated. Crossing also has a positive mean gap ({pct(v('pooled','cross_correct_transition','gap'))} / {pct(v('separated','cross_correct_transition','gap'))} pp), but its sign depends on the seed. Outer correctness and agreement are better on validation on average, and outer probability MSE is lower on validation in every seed. Bands fuzzy-truth gaps are {pct(v('pooled','bands_truth','gap'))} / {pct(v('separated','bands_truth','gap'))} score points; the separate inner and outer BCE/truth tables below show which side contributes. Both inner and outer bands BCE are worse on validation in every seed. Therefore the gap is **not only an inner-constraint problem**: the particularly large hard-correctness gap is inner, while confidence-based bands gaps extend to both sides. Better outer hard correctness/edge smoothness can coexist with worse outer unary confidence because these measures evaluate different properties; counts, pair variation and log-probability error are not interchangeable.

2. **Did separating the rules reduce gaps?** Only slightly for the main inner hard metrics. Inner-correctness gap shrinks by {-100*changes['inner_both_correct'][-1]['gap']:.6f} pp; inner-equality gap shrinks by {-100*changes['inner_hard_agreement'][-1]['gap']:.6f} pp. Crossing-correctness gap grows by {100*changes['cross_correct_transition'][-1]['gap']:.6f} pp; macro-Dice gap changes by {100*changes['macro_dice'][-1]['gap']:+.6f} pp. The bands-truth gap grows by {100*changes['bands_truth'][-1]['gap']:.6f} score points, worsening in all three seeds. Inner soft-MSE train advantage actually becomes larger in all three seeds: validation error is higher in every seed, while mean training error is slightly lower (training error itself increases in seed 2). There is no broad or substantial generalization-gap reduction. Moving an already negative outer gap toward zero is not inherently a benefit: training and validation values must be inspected separately.

3. **Are constraints almost perfectly satisfied on train?** Only the inner hard metrics approach 100%. Pooled/separated outer pair correctness is {pct(v('pooled','outer_both_correct','train'))}% / {pct(v('separated','outer_both_correct','train'))}%; outer equality satisfaction is {pct(v('pooled','outer_hard_agreement','train'))}% / {pct(v('separated','outer_hard_agreement','train'))}%; crossing correctness is {pct(v('pooled','cross_correct_transition','train'))}% / {pct(v('separated','cross_correct_transition','train'))}%. Crossing has substantial training soft error too (MSE {v('pooled','cross_soft_squared_error','train'):.6f} / {v('separated','cross_soft_squared_error','train'):.6f}). The claim that all constraints are solved on train is contradicted by the measurements. Crossing correctness is a strict exact-boundary-face measure, not the fraction of correctly segmented hippocampal voxels, so its low value is compatible with substantially higher Dice. Bands truth/adherence must likewise be read as their own confidence metrics, not as binary segmentation correctness.

4. **Is the gap disproportionately larger than Dice?** Inner pair-correctness gap (~14.4 pp) exceeds macro-Dice gap (~8.16 pp). However, **literal inner equality-satisfaction gap is only ~7.2 pp**, similar in size to macro Dice. Foreground-union Dice gaps are {pct(v('pooled','union_dice','gap'))} / {pct(v('separated','union_dice','gap'))} pp. Macro Dice additionally penalizes anterior/posterior confusion, which the foreground-union edge rules do not address. Pair correctness also fails whenever either endpoint is wrong, which can amplify a regional voxel error compared with a whole-mask overlap measure. Different measures condition on different regions and count voxels, edges, classes or confidence differently; a larger numerical gap is descriptive, not proof of a separate overfitting mechanism.

5. **Does this establish constraint-specific overfitting?** No. It is consistent with localized failure to generalize inner correctness and probability consistency. There is also ordinary segmentation generalization error, and crossing still has substantial error on train. The temporal evidence below shows a regional tradeoff while validation Dice improves, so global Dice alone misses the deterioration. A supplementary historical Dice-only baseline on the same cases and matched settings already has a **10.733824 pp inner-correctness gap**, a **5.331831 pp inner-equality gap**, and a **5.773682 pp macro-Dice gap**. Its selected epochs are 73/74/71. Thus the inner discrepancy is not unique to explicit constraints. The constrained arms enlarge the gap while improving validation Dice and boundary accuracy, consistent with a changed error balance; a matched baseline trajectory, controlled train/validation anatomy distribution, or independent test cohort would be needed to isolate a causal mechanism. The historical comparison is supplementary, not mixed into the two-arm tables. Gradient conflict and this generalization gap are different observations; neither proves the other caused the measured deterioration. The rules are jointly consistent: ideal p_i=y_i makes all three edge MSE terms zero. The observed competition concerns optimization and the learned error balance, not logically incompatible constraints.

6. **At which epochs?** Pooled 73/63/74 and separated 73/67/73. Exact checkpoint hashes are retained in the machine-readable provenance; checkpoint selection was unchanged.

7. **Are conclusions seed-sensitive?** The positive inner-correctness, inner-equality, bands fuzzy-truth (both sides), and macro-Dice gaps occur in all three seeds. The direction of the crossing-correctness gap changes: seeds 0/1 favor train, seed 2 favors validation. Outer pair correctness favors validation in seeds 0/2 and very slightly favors train in seed 1. Separation slightly reduces the inner-correctness gap for seeds 0/1 but increases it for seed 2. Consequently, the large inner gap is robust across these initializations; a general claim that separation improves constraint generalization is not. All individual seed values, including bands, appear in the tables below.

## When does the discrepancy develop?

The complete hard train–validation trajectory is unavailable: only validation was audited at intermediate checkpoints, and those weights were not retained. Full-training hard-gap onset cannot be reconstructed without retraining, which was not performed.

The available evidence nevertheless shows that the inner problem is **not confined to the selected checkpoint**. From epoch 15 to 60, validation inner correctness falls in every seed of both arms while validation Dice and crossing correctness rise in every seed. Means change from **96.19% to 86.68%** (pooled inner correctness) and **96.01% to 86.67%** (separated), while macro Dice improves from **53.58% to 77.39%** and **54.00% to 77.26%**. Some early predictions obtain high inner correctness by predicting too much foreground; inner correctness must therefore be considered together with outer and crossing accuracy.

Two fixed training cases have comparable *soft* rule audits at epochs 5/15/30/45/60. The inner MSE train–validation difference is small at epoch 5 (~0.00036), then widens progressively: training-probe inner MSE stays around 0.002 while validation rises to ~0.018 by epoch 60. Crossing MSE also develops a growing training advantage, although both splits improve; outer MSE becomes worse on train than validation. This supports gradually diverging, rule-dependent behavior rather than a uniform optimization failure. It is limited to two training cases and different probe/inference precision; the precision effect is not isolated, so this remains secondary evidence and cannot substitute for a full-cohort hard-gap timeline.

![Validation trajectories]({OUT}/validation_trajectories.png)

![Limited soft trajectories]({OUT}/soft_probe_trajectories.png)

The most defensible conclusion is **rule-dependent generalization deterioration plus optimization tradeoffs**, with insufficient evidence to label it causally constraint-specific overfitting. Separation alone barely changes that conclusion. No future experiment or additional training was launched as part of this analysis.
'''
    bands_note=''
    if all(r[f]==0 for arm in ['pooled','separated'] for r in a[arm]['bands_confidence_adherent'] for f in ['train','validation']):
        bands_note+='\nThe original bands confidence-adherence threshold (truth >=0.90) is met by **zero cases in either split in every seed**. That binary metric is at its floor and cannot distinguish these arms or their generalization; the continuous bands truth and raw side-specific BCE remain informative.\n'
    bands_note+=f"\nFor bands specifically, mean inner BCE is {v('pooled','bands_inner_loss','train'):.6f} train versus {v('pooled','bands_inner_loss','validation'):.6f} validation for pooled, and {v('separated','bands_inner_loss','train'):.6f} versus {v('separated','bands_inner_loss','validation'):.6f} for separated. Outer BCE is {v('pooled','bands_outer_loss','train'):.6f} versus {v('pooled','bands_outer_loss','validation'):.6f}, and {v('separated','bands_outer_loss','train'):.6f} versus {v('separated','bands_outer_loss','validation'):.6f}, respectively. Both sides have worse validation BCE in every seed. The outer bands confidence gap therefore remains present even though outer pair agreement and hard pair correctness are better on validation on average.\n"
    intro=intro.replace('## Answers to the seven questions',bands_note+'\n## Answers to the seven questions')
    (OUT/'INTERPRETATION.md').write_text(intro)
    methods=(OUT/'TABLES_AND_METHODS.md').read_text()
    (OUT/'REPORT.md').write_text(intro+'\n\n---\n\n'+methods+'\n\n---\n\n'+(OUT/'HISTORICAL_BASELINE.md').read_text())

if __name__=='__main__':main()
