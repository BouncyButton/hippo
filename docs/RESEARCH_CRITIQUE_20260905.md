# Research critique and falsifiers — 2026-09-05

The handoff supports retiring the unreplicated bands headline, preserving matched
configuration/exposure comparisons, and testing stronger supervised controls. Its
claims that all differentiable constraints are blocked by saturation, that CE must
remove saturation, and that a detached translation teacher cannot saturate do not
follow from the evidence. Treat the three proposed routes as hypotheses.

## 1. Probability saturation does not imply train-time immobility

For `p = softmax(z)`, `d p / d z = diag(p) - p p^T`. A loss with a bounded derivative
with respect to probabilities, such as squared probability error, can have a very
small logit gradient near a one-hot prediction. However, for a target distribution
`q`, stable cross-entropy `L = -sum(q * log_softmax(z))` has `dL/dz = p - q`.
A confidently wrong class therefore retains an order-one corrective gradient.
This follows directly by differentiating the official
[cross-entropy definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

For example, logits `(40,-40)` with the second class as target have essentially
unit-magnitude opposing gradients, despite numerically saturated probabilities.
Logit-space likelihoods, margin objectives and structured log posteriors are smooth
counterexamples to the handoff's universal statement. Computing `log(clamp(p))`
after a low-precision softmax is not equivalent: underflow and clipping can erase
the relevant gradient or original logit margin.

A weak post-hoc plane-prior intervention measures that specific intervention at a
fixed checkpoint. It does not test the reachability of the optimizer trajectory
from initialization, nor all differentiable losses. Even a failed frozen-logit
gradient audit can only reject the tested formulation, checkpoint, scale and metric;
a passing audit establishes a necessary mechanism, not generalization.

## 2. Dice+CE is an appropriate control, not a proven prerequisite

Hard-label CE supplies corrective gradients for confident errors, but its
per-example optimum also favors the true class with probability one. Adding it
does not mathematically guarantee less saturation, calibration, a later Dice peak,
or greater TTA gain. Modern neural networks can be overconfident; see
[Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599). There are theoretical
reasons for CE and soft Dice to differ under label uncertainty, but they do not
determine the behavior of this specific finite-data Dice+CE mixture;
[Nordström et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Nordstrom_The_Impact_Label_Noise_and_Choice_of_Threshold_has_on_CVPR_2025_paper.html).

Measure saturation and calibration separately on foreground, A/P-interface,
outer-boundary, correct and incorrect voxels. A percentage over every voxel is
dominated by background in a 64-cube crop. Use the conditional A/P probabilities
when diagnosing A/P assignment. Preserve the saturation threshold and denominator,
and report NLL/Brier plus confidence histograms alongside hard Dice and errors.
Temperature scaling alone preserves argmax for every positive scalar temperature;
better calibration and better segmentation are distinct questions.

The smallest clean supervised-control comparison is Dice-only versus an explicitly
specified Dice+CE mixture with every other setting matched, using two paired seeds
and the registered exposure/endpoint. An eventual constraint gain must be measured
against its own supervised-loss control. Changing both the supervised loss and the
constraint in one run cannot identify either contribution. A full 2-by-2 loss ×
constraint design would test an interaction, but is not necessary to start the
baseline control experiment.

## 3. A detached translation teacher can still agree while wrong

For `q = stopgrad(mean_s inverse(T_s) f(T_s x))`, CE/KL student gradients vanish when
`p=q`, including a translation-equivariant but wrong predictor. Squared consistency
error also vanishes there. Stop-gradient changes where derivatives flow; it does
not eliminate wrong fixed points or guarantee a teacher improves with training.
The proposed current-model view average is also different from an EMA-weight
teacher. [Mean Teacher](https://arxiv.org/abs/1703.01780) averages model weights;
its empirical benefits are motivation, not a non-saturation theorem.

Distilling an ensemble can work empirically
([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), but a measured inference-time
TTA improvement is not automatically transferable to one model's weights. Teacher
KL/CE should use stable student log probabilities and detached targets. Audit the
gradient, valid non-wrapped translation support, teacher quality and late-epoch
behavior. A matched supervised translation-augmentation control is needed before
attributing gains specifically to the consistency relation when ordinary
augmentation was previously absent.

## 4. Sharpening a cut distribution does not identify the correct cut

If candidate scores are `s`, replacing them by `a*s` with `a>1` increases the
top-two margin and reduces `H(softmax(a*s))`, while preserving their ranking and
argmax exactly. In fact `dH/da = -a * Var_softmax(a*s)(s) <= 0`. Entropy minimization
or self-selected margin maximization can therefore make the wrong cut more
confident. Top-five oracle recall says that an acceptable candidate is nearby; it
does not establish a learnable, inference-available signal that ranks it correctly.
Same tissue at the local interface also does not prove that global image landmarks
carry no location information.

A GT-anchored negative log posterior over candidate planes is a coherent
**supervised structural auxiliary**, with a correctness signal supplied by training
labels. It is not a pure anatomical prior. Candidate support, class orientation,
normalization and temperature need explicit definitions. The new
`thesis/new_constraints/ap_cut/` implementation uses conditional A/P log
probabilities and detached GT support, and tests corrective gradients on saturated
wrong logits. It does not claim remaining Dice headroom.

## 5. New local geometry audit rejects a universal exact-plane assumption

On the current raw Dataset101 NIfTI files and the handoff's matching fold-0 split
SHA256 (`1d6a3fe993359ad85e9c28a29901162251c3359449b710c09cb9e92d66dda472`):

| Cohort | Cases | Exact unique planes | Cases with mixed A/P slices | Best planar disagreements |
|---|---:|---:|---:|---:|
| Train | 208 | 161 | 47 | 1,104 foreground voxels |
| Validation | 52 | 41 | 11 | 284 foreground voxels |

There are 51 mixed slices in train and 12 in validation, at most two per case.
The deviations are small in voxel count (about 0.16% of foreground) but they falsify
exact planarity for these arrays. All 260 NIfTI sform linear parts are identity;
anterior occupies the high side of stored spatial axis 1. All arrays exactly match
the previous band-locality audit's NIfTI parser. Independent direct byte-offset
checks confirm both class values in every mixed slice.

All 52 existing validation ground-truth exports in each of the baseline and
translation error-map folders equal these raw arrays after symmetric 64-cube
padding/cropping. This confirms the finding for those exported targets too. The
training pickle remains unverified in the local environment. The handoff's previous
plane-fit statistic may describe a derived interface mask or a different convention;
that computation has not been reproduced here, so the discrepancy is not attributed
to a specific earlier bug.

The strict candidate must therefore fail its geometry gate on the current arrays.
It should not silently project labels to a plane or omit exceptions. An explicit
skip policy would define a different, partial-cohort experiment. A subsequent
preregistered F1–F7 investigation tested the tolerance-aware direction and closed
it on the official checkpoint: slice-pooled conditional A/P log-odds degenerate
to the hard vote count, while predicted cut shrinkage is appropriate under weak
image evidence. Evidence,
case IDs, exact conflicting coordinates, source hashes and a reproducible CPU
audit are in `experiments/loss_constraint_followup_20260905/gt_geometry_audit.json`
and `audit_gt_geometry.py`.

The closure and corrected official-pool statistics are recorded in
`experiments/loss_constraint_followup_20260905/A_P_BRANCH_CLOSURE_20260905.md`.
This empirical closure does not restore the handoff's original universal claim:
stable logit-space losses still have corrective gradients on confident errors.
The failure is that the tested aggregate reads no additional localization signal.

## 6. Statistical and causal limits

Two paired seeds are a useful minimum replication check, but their gap is not an
estimate precise enough to establish a universal significance threshold. The rule
`effect > 2 * 0.0012` is a practical heuristic, not a formal claim criterion. Epochs
within one training run are correlated and are not independent seed replications.
Report both seed-specific paired effects, their mean and uncertainty appropriate to
the actual experimental unit; do not pool repeated epochs or telemetry duplicates
as independent runs.

Oracle swap budgets and correctly placed planes are conditional ceilings, not
achievable gains or evidence of generalization. Failure of one local corrector
does not prove an information-theoretic boundary ceiling. Repeated fold-0 search,
checkpoint selection and inspecting the same 52 validation cases make subsequent
findings exploratory until a locked evaluation protocol confirms them. Cross-paper
Dice comparisons additionally require matching data, splits, preprocessing, label
definitions, exposure and pooling before attributing a gap to the loss.
