# Renderer-state inputs help fitting, but selective preservation still fails

Follow-up: the [completed loss-flow audit](OVERFIT_LOSS_FLOW_RESULTS_20260923.md)
decomposes the improvement and measures activation gaps and shared-parameter
gradient interference at these saved heads.

Job **666786** completed all 24 fits in **4m37s**. Explicit renderer-state inputs
reduced the final fitting objective in all 12 matched fold/seed pairs and improved
rescue-relevant ranking. The resulting masks changed little compared with the
control, and every final head increased both false-positive burdens relative to
E. No model was promoted and no inner/outer evaluation was performed.

## Frozen comparison

The [protocol](RENDERER_STATE_OVERFIT_PROTOCOL_20260923.md) fixed the first four
sorted fitting cases per fold before training. Across folds these are **seven
distinct cases and 16 case/fold exposures**, repeatedly used across three head
seeds. These are small, overlapping fitting subsets, not independent validation.

Both arms started from the original E residual head, with width 24, unchanged
mean pooling over z, identical initial outputs and **8,185 trainable parameters**.
Both allocated six extra input channels. The control received zeros; the state
arm received frozen presence logit, signed offsets to the fitted boundaries,
raw/rendered argmax margins and beta. No ground-truth-derived feature was used.
The extra weights are inactive in the control, so matched allocated size does
not mean identical effective input capacity.

E's backbone, original head, fitted geometry and renderer strength stayed frozen.
Only the new correction head was fitted. Both arms used the same interval
targets, group balancing, initial weights, case order per seed and AdamW
settings: lr=3e-4, weight decay=0, 1,024 case updates. This is 256 passes over
four cached cases. Steps 0,128,512,1024 were reported; step 1024 was fixed as
primary, without selecting a checkpoint or extending training after inspection.

The optimizer settings, data size and repeated exposure differ from the previous
full fitting experiment. Cross-experiment improvements cannot be attributed to
input features alone; the within-experiment comparison is matched.

## Final fitting results

Values below average cases within each subset and then the 12 fold/seed fits.
AUROC compares feasible missed thin rays with hard-empty rays, using the learned
correction as score. Labels define these diagnostic groups only. Thin means an
untruncated one- or two-voxel true z-span. An AUC of 0.5 is chance ranking.

| Metric | E / zero correction | Original-input control | Renderer-state inputs |
|---|---:|---:|---:|
| Interval fitting objective | 0.185186 | 0.126168 | **0.118788** |
| Rescue-relevant correction AUC | 0.5000 | 0.5119 | **0.6093** |
| Thin overlap recall | 66.4372% | 69.9222% | 70.1443% |
| Thin-voxel recall | 64.2963% | 68.2529% | 68.4159% |
| FP rays/case | 7.0625 | 8.5000 | 8.6042 |
| FP voxels/case | 239.2500 | 246.8750 | 247.0417 |
| Recovered true voxels/case versus E | 0 | 8.7083 | 8.8958 |
| Additional true deletions versus E | 0 | 0 | 0 |
| Union Dice | 0.9392503 | 0.9395385 | 0.9395391 |

The best representable constant's mean objective was 0.130460. All 24 heads beat
their subset's constant at the final step. This revises any suggestion that the
original-input architecture can only learn a constant: it can do better with
this small-subset fitting setup, although its rescue-relevant ranking remains
weak.

The state arm's fitting loss was **2.92–9.32% lower** than its paired control in
all 12 comparisons. Its average loss was 5.85% lower than the control's average.
However, the state-minus-control change was only **+0.2221 percentage points**
in thin overlap, +0.1631 points in thin-voxel recall, +0.1875 recovered true
voxels/case and **+0.1667 FP voxels/case**. The mean FP-ray change was +0.1042.
No added true deletions occurred, but E's existing missed anatomy remains.

All 24 final heads increased both mean FP ray and FP voxel counts relative to
E. Therefore neither arm satisfies the preservation criterion even on these
fitting subsets. No ASSD improvement is claimed; ASSD was not recomputed in
this diagnostic.

## Heterogeneity and learning curves

| Fold, averaging seeds | Control objective | State objective | Control AUC | State AUC |
|---|---:|---:|---:|---:|
| 1 | 0.155004 | 0.144170 | 0.5225 | 0.7421 |
| 2 | 0.110965 | 0.107337 | 0.5728 | 0.6067 |
| 3 | 0.089513 | 0.081738 | 0.3906 | 0.4911 |
| 4 | 0.149189 | 0.141907 | 0.5618 | 0.5971 |

The extra thin-overlap gain relative to the control came entirely from fold 1;
the other three folds had identical thin-overlap recall between arms at the
final step. Fold 3 illustrates the limitation of loss improvements: its state
arm had lower loss, but rescue-relevant ranking remained around chance.

| Updates | Control objective | State objective | Control AUC | State AUC |
|---|---:|---:|---:|---:|
| 0 | 0.185186 | 0.185186 | 0.5000 | 0.5000 |
| 128 | 0.130084 | 0.123209 | 0.4670 | 0.5664 |
| 512 | 0.127795 | 0.121181 | 0.4949 | 0.5979 |
| 1024 | 0.126168 | 0.118788 | 0.5119 | 0.6093 |

Both mean objectives were still decreasing at the final checkpoint. This test
does not establish convergence or impossibility of fitting these examples.

## What was learned

**Observed:** explicit prediction-derived renderer inputs make this head fit
the interval objective better and improve ranking on these examples. A blanket
claim that the extra information is useless is contradicted by this comparison.

**Not established:** the original features lack the information, any individual
added channel is responsible, or a bigger head would/would not help. The bundle
changes input access, active connections and optimization conditioning. These
effects were not separated. The experiment also does not show transfer to
unseen cases.

**Observed:** lower objective and better ranking do not translate into preserved
FP burden. In the state arm, a case-averaged **78.50%** of feasible missed thin
rays still fall below their allowed rescue interval, versus 79.33% in the
control. Hard-empty upper-bound violations are **4.53%** versus 4.42%. Bounds
include the 0.001-logit safety margin; these violation rates are diagnostics,
not substitutes for the actual voxel counts above.

**Inference:** input access is part of the problem, but insufficient to explain
or repair the tradeoff. The next uncertainty is how improvements in this
group-balanced objective distribute across rays and whether they place the
right corrections across the actual voxel activation thresholds. This requires
a per-group loss/gradient and activation-margin analysis, not promotion on
fitting loss or another unstructured width search. Such an analysis should
distinguish feasible missed thin rays from already-correct thin rays and hard
empty rays. No further training or held-out screen was launched here.

## Verification and artifacts

Three tests passed: label-free renderer feature/margin construction; matched
initial outputs/weights/parameter counts with gradient access to the state
channels; and both-arm fitting/save smoke tests with unchanged original weights.
The launcher passed shell syntax checks. Input/source hashes and the frozen E
checkpoints were verified. The job completed with exit code 0.

The user explicitly authorized this and future non-identifying aggregate reports
for this investigation. Images, case identifiers, individual case records and
new head weights remain on the cluster. Local reports:

- [Full aggregate histories](RENDERER_STATE_OVERFIT_AGGREGATES_666786.json)
- [Compact numerical comparison](RENDERER_STATE_OVERFIT_SUMMARY_666786.json)
- Implementation: `experiments/presence_decisive_20260923/renderer_state_overfit.py`
- Remote root: `/home/3160552/renderer_state_overfit_20260923_01a0cd/results_666786`

These results are internal mechanistic evidence from seven fitting cases,
overlapping subsets, three head-ordering seeds and the original E backbones.
They provide no independent generalization, clinical or reliable boundary-
position-learning evidence. Original E remains the retained model.
