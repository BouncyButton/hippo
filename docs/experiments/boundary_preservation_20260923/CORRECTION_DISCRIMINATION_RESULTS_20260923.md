# Saved-head audit: weak selective correction already on fitting data

The proposed fitting-only comparison is now [completed](RENDERER_STATE_OVERFIT_RESULTS_20260923.md).
Explicit renderer inputs improved fitting loss and ranking, but neither arm
preserved E's false-positive burden on the small fitting subsets.

Job **666780** completed successfully in **3m43s**, auditing all 24 final heads
from job 666769: two objectives, four folds and three head-fitting seeds. No
weights were updated, no outer images were evaluated, and no selections changed.
The [protocol](CORRECTION_DISCRIMINATION_PROTOCOL_20260923.md) was written before
execution. The user explicitly authorized retrieval of the non-identifying
[aggregate report](CORRECTION_DISCRIMINATION_AGGREGATES_666780.json).

**Conclusion:** the interval head has weak discrimination on its own fitting
data for the rays that need rescue. Its effect is largely reproduced by a
constant renderer-control shift. This undermines a purely held-out
generalization explanation, but does not establish whether optimization, target
design or available features cause the limitation.

## Discrimination where rescue is needed

The main diagnostic compares true thin rays that E misses but that admit a
label-dependent, protection-preserving rescue within ±4 logits, against hard
empty rays. Hard empty means a true-empty ray marked present by either the raw
mask or original head. These groups are defined for analysis only; feasibility
is not an inference feature. Thin means an untruncated z-span of one or two
voxels, consistently with the preceding experiment.

AUROC below is computed within each case and averaged over cases with both
groups, then equally across the 12 interval fold/seed fits. A constant score has
AUROC 0.5. No significance claim follows from these descriptive averages.

| Score / positive group versus hard empty | Fitting | Inner validation |
|---|---:|---:|
| Learned correction, feasible missed thin | **0.5567** | **0.5285** |
| Learned correction, all true thin | 0.6332 | 0.5789 |
| Original presence-logit anchor, feasible missed thin | 0.7874 | 0.5866 |
| Anchor plus learned correction, feasible missed thin | 0.7870 | 0.5868 |

The head contains some ranking information, especially over all thin rays. It
is less discriminative on the subset actually needing feasible rescue, and
barely changes the original anchor's ranking. AUROC alone does not determine
voxel outcomes because activation thresholds differ between rays.

For the main correction-only AUC, fitting values range from 0.5133 to 0.6125
across the 12 heads; inner values range from 0.4501 to 0.6083.

| Fold, averaging three head seeds | Fitting AUC | Inner AUC |
|---|---:|---:|
| 1 | 0.5256 | 0.5274 |
| 2 | 0.5774 | 0.5924 |
| 3 | 0.5990 | 0.4597 |
| 4 | 0.5246 | 0.5344 |

Fold 3 has a clear descriptive fitting/inner gap. The original anchor also has
a substantial gap across the four folds. Generalization issues coexist with
weak fitting discrimination; it would be incorrect to rule them out entirely.

## Corrections overlap, and a constant largely reproduces the masks

Average group correction means over the 12 interval heads:

| Group | Fitting correction | Inner correction |
|---|---:|---:|
| All true thin | +0.6342 | +0.6335 |
| Feasible missed thin | +0.6278 | +0.6285 |
| Hard empty | +0.6252 | +0.6297 |
| Other empty | +0.4691 | +0.4689 |

These are not literally constant outputs: easier empty rays receive different
corrections. Within each head, the average ray-level standard deviation is
about 0.0186 for fitting feasible-missed thin rays and 0.0196 for hard-empty
rays. The corresponding means differ by only 0.0026 logits.

Two controls were chosen using fitting data only and reused unchanged on inner
data: the case-averaged learned correction and the constant minimizing the
original fitting objective. The objective-optimal interval constants were
0.6653, 0.6581, 0.6860 and 0.6947 logits for folds 1–4 respectively.

**Eight of 12 interval heads had worse fitting loss than this best constant.**
The four better heads improved the objective by only 0.0795%, 0.4648%, 0.3483%
and 0.2231%. The constant is representable by this architecture: leave the
initial branch unchanged and add `4*atanh(c/4)` to the trainable final bias.
A unit test verifies this. The eight worse heads therefore did not reach even
the fitting loss of this simple representable control at their final checkpoint.
This is evidence of suboptimal final fitting for that objective, not proof that
optimization alone explains the failure to distinguish anatomy.

Learned versus objective-optimal-constant masks disagree by only **2.5753 voxels
per fitting case** and **2.7460 per inner case** on average. Mask similarity alone
can be misleading in a boundary task, so the actual error changes are also shown:

| Interval diagnostic | Fitting learned | Fitting constant | Inner learned | Inner constant |
|---|---:|---:|---:|---:|
| Thin overlap recall | 66.8714% | 67.0843% | 62.2001% | 62.3124% |
| Recovered true voxels/case versus E | 6.7249 | 7.2455 | 6.5040 | 6.9048 |
| Added FP voxels/case versus E | 6.9704 | 7.5271 | 8.5913 | 9.0476 |
| Additional true deletions versus E | 0 | 0 | 0 | 0 |

The head makes a slightly smaller correction with slightly less rescue and
slightly fewer false positives. This comparison does not establish identical
performance at exactly matched rescue or FP burden. The fitting-mean constant
likewise gives less rescue and fewer FPs than the learned head. Neither constant
nor learned head solves selective rescue under the original guards.

## How the interval constraints fail

Among feasible missed thin rays, the learned correction lies below the
protection-preserving rescue interval in a case-averaged **80.77%** on fitting
data and **82.43%** on inner data. On hard-empty rays it exceeds the permitted
upper bound in **4.73%** and **6.61%** respectively. Bounds include the frozen
0.001-logit safety margin; interval violations are not identical to voxel-error
counts, which are reported separately above.

This is a concrete combination of insufficient correction on many rescuable
rays and excessive correction on some empty rays. A uniform strength increase
does not distinguish these two populations.

The existence-BCE controls behave similarly at much larger amplitude. Their
rescue-relevant correction AUC averages 0.4827 fitting and 0.3990 inner. Their
best constants are approximately +3.90 to +4 logits. Learned BCE fitting loss
beats its constant by only 0.13–0.99%; masks disagree by 0.8348 fitting and 2.3948
inner voxels/case on average. Neither objective has demonstrated useful enough
selective correction in this representation and training setup.

## Code-level explanation to test next

The current correction is algebraically

`delta = 4*tanh((residual(inputs) - initial_residual(inputs))/4)`.

Although `anchor` is an argument to `PresenceProbe.forward`, it is added to the
output and subtracted again by `bounded_delta`. It is not a direct input to the
learned correction branch. A unit test verifies anchor independence up to
float32 rounding. The seven stem inputs contain voxel foreground probability,
first/last-hit channels and four projected backbone channels. The branch pools
over z; fitted lower/upper positions, beta and the actual three-class argmax
margin are not supplied explicitly to it. Those quantities influence which
correction is safe, but the branch must infer relevant information indirectly.

This is a verified information path, not proof that the necessary information
is absent from the existing features. The head may infer some of it. Likewise,
good backbone body segmentation does not prove that this projection and pooling
retain all signals needed to select a boundary correction.

The next small discriminating experiment should use a fixed fitting-only subset
and compare original inputs against explicit frozen renderer-state inputs,
keeping head size, targets and optimization matched. First check whether the
existing head can overfit those examples and beat the representable constant.
Add the original anchor, fitted interval and local raw class margins as
deployable inputs in the matched arm; no GT-derived feasibility inputs. If only
the explicit-state arm fits selective corrections, that supports an input-path
limitation. If both fail, inspect objective gradients and optimization before
adding capacity. If both fit but fail on inner data, the remaining failure is
primarily about transferring the learned rule to held-out cases. This follow-up
has not been launched.

## Scope and verification

Each fold used 166 fitting and 42 inner cases. Fitting and inner sets are disjoint
within a fold; their reuse across folds and across seeds does not yield
independent observations. Three head-fitting seeds do not replace independent
backbone seeds. All data remain part of the repeatedly used development pool.
ASSD was not recomputed for this diagnostic, and no clinical or external
generalization conclusion is warranted.

Original split, source, data and saved-checkpoint hashes were checked. All 24
heads were evaluated without gradient updates. Four diagnostic tests pass; the
eight existing correction-control tests also passed during this audit. The
constant-representability/anchor-independence test was added after the first
results to check the code interpretation and does not modify the experiment.

Implementation: `experiments/presence_decisive_20260923/audit_correction_discrimination.py`.
Remote results: `/home/3160552/correction_discrimination_20260923_01a0cd/results_666780`.
