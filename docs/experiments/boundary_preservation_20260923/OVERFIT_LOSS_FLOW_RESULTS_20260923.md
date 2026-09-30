# Loss-flow audit: intended learning, large rescue gaps, opposing shared gradients

Follow-up completed: the [matched boundary-pooling experiment](BOUNDARY_POOL_OVERFIT_RESULTS_20260923.md)
improved fitting discrimination but did not meet the preservation criteria.

Job **666826** completed the audit of all 24 frozen heads in **1m38s**, with no
weight updates or inner/outer evaluation. The audit used the same four fitting
cases per fold as job 666786: seven distinct cases, 16 case/fold exposures and
three head-fitting seeds. All findings below remain local, internal diagnostics.

The [protocol](OVERFIT_LOSS_FLOW_PROTOCOL_20260923.md) separates three questions:
which groups explain the loss reduction, how far missed anatomy remains below
its activation threshold, and how the shared head parameters respond to the
different group losses. The principal results concern the 12 renderer-state
heads, compared with their 12 original-input controls.

## 1. The objective improved mainly on the intended rays

All entries retain the original four-group normalization. Subdividing thin rays
for analysis does not increase their loss weight. Values average the four cases
within each subset and then the 12 fold/seed fits.

| Weighted loss contribution | Original inputs | Renderer-state inputs | State minus control |
|---|---:|---:|---:|
| Initially missed, feasible thin | 0.087773 | 0.078165 | **−0.009608** |
| Initially missed, infeasible thin | 0.024271 | 0.026694 | +0.002423 |
| Initially overlapping thin | 0.002700 | 0.003586 | +0.000886 |
| Other true rays | 0.007092 | 0.006461 | −0.000632 |
| Hard-empty rays | 0.003460 | 0.003480 | +0.000020 |
| Other empty rays | 0.000872 | 0.000403 | −0.000469 |
| Total | 0.126168 | 0.118788 | **−0.007380** |

The feasible-missed-thin contribution decreases in every fold mean. Its reduction
exceeds the net objective reduction because losses on other thin groups increase.
Thus the hypothesis that the improvement came mainly from easy empty rays is
not supported. The new features help the intended training target, but the
remaining deficits are still substantial.

In the state arm, almost all of the feasible-missed-thin contribution is the
lower-bound hinge: 0.077361 of 0.078165. The direct gradient asks for greater
correction on these rays. Infeasible missed thin rays receive the predefined
zero-correction target; their loss is entirely the squared deviation from zero.
They are true anatomy, not absence labels. Infeasible means that the frozen
bounded scalar action cannot satisfy all the protection/rescue constraints,
including its ±4 domain and numerical safety requirements.

This does not make the per-ray target vector mathematically inconsistent:
different rays can have different actions. The question is whether shared
parameters can learn those differences. Removing the infeasible-ray penalty
would also remove a restraint on corrections that may violate protected
background; this audit does not justify that change by itself.

## 2. Most remaining misses are well short of the rescue threshold

For feasible thin rays still lacking true overlap under the learned state head:

- Mean shortfall to the safe rescue lower bound: **1.3938 residual logits**.
- Mean fraction within 0.01 logits: **0%**; within 0.1: **2.04%**.
- Within 0.25 logits: **7.27%**; within 0.5: **19.73%**.
- Within 1 logit: **37.91%**; within 2: **72.59%**.
- Per-head median gaps range from **0.6568 to 1.8138 logits**.
- Mean derivative of the bounded tanh correction: **0.9519**, where its maximum
  is 1. This does not indicate strong tanh saturation on average for these rays.

These fractions and means are computed per head and then averaged, not treated
as independent pooled observations. Logit gaps are not voxel or millimetre
distances. The findings argue against a merely numerical near-threshold miss
or correction-tanh saturation as the principal explanation in this subset.
They do not rule out the narrow-interval penalty of the product renderer.

Fixed diagnostic increments confirm the FP tradeoff:

| State-head intervention | Thin overlap recall | FP rays/case | FP voxels/case |
|---|---:|---:|---:|
| Saved learned output | 70.1443% | 8.6042 | 247.0417 |
| Uniform +0.25 logits, clipped to domain | 71.3054% | 9.0833 | 250.0417 |
| Uniform +0.5 logits, clipped to domain | 73.1709% | 9.5208 | 252.5625 |
| GT-selective feasible-thin oracle | **82.1818%** | **8.6042** | **247.0417** |

The last row is explicitly **non-deployable**. It uses labels to increase only
currently unrescued feasible thin rays to their lower bound, leaving every
other correction unchanged. It recovers another **5.8542 true voxels/case**
without adding FP voxels or true deletions relative to the learned output.
Assertions checked these protections on every evaluated mask.

This quantifies available selective rescue within the frozen action space;
it does not show that the network can identify those rays. It also retains
the learned output's existing excess FPs: E's FP burden on these cases was
7.0625 rays and 239.25 voxels/case. The oracle does not itself satisfy the
overall requirement to preserve E's FP burden, and it says nothing about the
share of ASSD attributable to thin anatomy. ASSD was not computed here.

## 3. Shared parameter gradients remain almost directly opposed

For each final head, gradients are calculated from the same four-case objective.
The average cosine between group gradients in the renderer-state arm is:

| Gradient pair | Mean cosine |
|---|---:|
| Feasible missed thin vs hard-empty | **−0.999649** |
| Feasible missed thin vs infeasible missed thin | **−0.999888** |
| Feasible missed thin vs already-overlapping thin | **−0.999334** |

The original-input controls are similarly opposed: −0.999955 for rescue versus
hard empty. Adding explicit renderer information improved fitting, but did not
remove this near-opposition of the group-averaged gradients at the saved heads.
Competing gradients are not intrinsically a bug; the important additional
finding is their effect through the shared parameter-to-output mapping.

Using derivatives of mean correction over the receiver rays:

- In **all 12 state heads**, the feasible-thin loss's descent direction would
  increase correction on both remaining feasible thin rays and hard-empty rays.
- The hard-empty and infeasible-thin loss contributions oppose correction
  increases on the rescuable thin receivers in **all 12** state heads.
- The total objective's local SGD direction raises the remaining-thin mean
  correction in **4/12** state heads and lowers it in **8/12**. It moves the
  hard-empty mean in the same direction in each head.
- Across both arms, **none of the 24 computed total directions** raises the
  remaining-thin mean without also raising the hard-empty mean.

This does not prove that no selective parameter direction exists. It describes
these particular gradients and group-mean readouts at the saved parameter
values. Per-ray changes can differ from the mean. The calculations are
first-order derivatives along an L2-normalized SGD direction, not a full unit
step, an AdamW update or a reconstruction of historical training. They support
local shared-parameter interference, not a causal claim about prior fold failures.

## Interpretation and next discriminating change

The evidence now separates useful learning from successful decisions. The new
inputs reduce the intended thin-ray loss, but learned corrections remain far
below many rescue thresholds. A stronger shared rescue signal would also raise
empty-ray corrections in this local diagnostic. Weight tuning alone therefore
has no demonstrated route to selective preservation here.

The next narrowly controlled architecture test should change how local boundary
features are pooled while holding the state inputs, interval objective, width
and fitting subsets fixed. For example, compare current whole-ray mean pooling
with two fixed, prediction-defined pools around the fitted lower and upper
boundaries, using the existing two-block presence readout. Keep initial rendered
outputs identical via initial-branch subtraction and freeze the pooling width
before running. This tests whether retaining local boundary evidence reduces
the coupling seen here; it is a hypothesis, not an established cause or fix.

Require better actual thin recovery without increases in either FP burden or
true deletions on the fitting test, alongside improved group response. Lower
loss, better AUC, or less opposed gradient cosines alone would not be success.
Failure would weaken this pooling hypothesis without proving that all function
heads fail. No such training comparison was launched during this audit.

## Numerical checks and provenance

Job 666811 stopped on a componentwise float32 gradient-reconstruction check
after one completed head. Its outputs were not used as the final report. The
amended audit records vector reconstruction error relative to contributing
group gradients, uses the directly computed original-objective gradient for the
total direction, and suppresses responses if reconstruction error is too large.
This changes diagnostic validation only; all model files and loss definitions
remain unchanged.

All 24 completed diagnostics passed the revised check. Maximum reconstruction
L2 error relative to the sum of group norms was **8.12e-6**; maximum relative
error to the total norm was **0.0002868 (0.0287%)**. Every signed response passed
the numerical reliability rule. Saved fitting losses were reproduced within
1e-5. Data/source/checkpoint hashes were verified, and no saved weights changed.

Three diagnostic tests passed: exact loss/gradient decomposition on a synthetic
case, signed response versus a finite parameter perturbation, and complete
audit/oracle smoke checks with unchanged weights. The user authorized current
and future non-identifying aggregate downloads for this investigation. Images
and individual case records remain on the cluster.

- [Full aggregates](OVERFIT_LOSS_FLOW_AGGREGATES_666826.json)
- [Compact summary](OVERFIT_LOSS_FLOW_SUMMARY_666826.json)
- Implementation: `experiments/presence_decisive_20260923/audit_overfit_loss_flow.py`
- Remote root: `/home/3160552/overfit_loss_flow_v2_20260923_01a0cd/results_666826`

E remains the retained model. These seven-case fitting diagnostics do not
establish external generalization, clinical usefulness or reliable boundary-
position learning.
