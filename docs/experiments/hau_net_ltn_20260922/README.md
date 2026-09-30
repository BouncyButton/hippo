# HAU-Net paper: relevance to nnU-Net and an LTN experiment

Source: Gong et al., *2.5D HAU-Net with gated spatial attention for automatic
hippocampus segmentation in MRI*, Journal of Neuroscience Methods 431 (2026),
110730, DOI: [10.1016/j.jneumeth.2026.110730](https://doi.org/10.1016/j.jneumeth.2026.110730).
This note uses the PDF supplied by the user on 22 September 2026.

## What the paper establishes

HAU-Net is a **binary whole-hippocampus 2D U-Net**, trained on three adjacent
image slices as channels. Decoder features gate encoder skip features. It uses
Dice + BCE and samples 20 slices per volume from start, middle and end. Its
reported Decathlon Dice is 91.05% in Tables 4, 5, 7 and 8, and 91.00% in the
factorial ablation (Table 3). Table 3 attributes +0.75 percentage points to
2.5D input alone, +0.60 points to attention alone, and +1.92 points to their
combination relative to its 2D U-Net. Its HarP Table 3 Dice is 90.62%.

The anatomical information is supplied by **input context and sampling**, not
by an explicit anatomical proposition, segmentation constraint, or LTN. The
paper mentions nnU-Net as a desirable reference but does not test against it.
Its large comparison gains should not be treated as a gain over our trained
3D nnU-Net. The paper's 3D U-Net comparison uses a depth-16 patch and scores
65.06% (Table 8); it does not represent a tuned nnU-Net.

Its Decathlon masks combine anterior and posterior labels into one foreground
(Methods, Eq. 32). Our Dataset101 labels have background, anterior and
posterior classes. Whole-hippocampus gains cannot establish any gain on the
anterior/posterior interface. Our archived nnU-Net full-volume audit reports
0.8934 mean class Dice and 0.9126 foreground-union Dice on its 52-case
validation split; the largest class-specific issue is A/P confusion near the
interface. These figures use different splits, labels, and metric aggregation
from the paper and **must not be compared numerically** as model rankings.

Interpretation cautions from the PDF: its U-Net Decathlon Dice is 89.08% in
Table 3, 80.38% in Table 7, and 83.19% in Table 8, without a clear mapping
between the different protocols. HarP HAU-Net Dice is 90.62% in Tables 3/6
and 90.39% in Table 7. The text describes axis 0 as superior-inferior while
Fig. 1 labels normalized position from anterior to posterior. The methods
describe an 8:1:1 split and the statistical section describes five-fold
cross-validation, so reported scores need protocol-specific interpretation.

## Local, training-label-only feasibility audit

Using fold 0's 208 training labels in `datasets/Dataset101_MSD`, without
looking at validation labels or predictions, we found:

| Axis in native RAS NIfTI | Median depth | Median foreground voxels in paper's start/mid/end sampled slices | Cases with at least one occupied slice omitted | Cases with a hole in the foreground occupancy interval |
| --- | ---: | ---: | ---: | ---: |
| A/P (axis 1) | 50 | 23.6% | 208/208 | 0/208 |
| S/I (axis 2) | 36 | 59.1% | 208/208 | 0/208 |

Sampling used indices 0:5, `N//2-5:N//2+5`, and `N-5:N`, with first and last
slices excluded to permit adjacent-slice input. Thus the audit has at most 18
valid centers per case; the PDF calls the intended budget 20, while its
inclusive middle-window formula could produce 11 rather than 10 middle
indices. This is only a label-coverage calculation. The paper's slice policy
may suit its 2D training budget, but
copying it into our 3D nnU-Net would discard a large amount of supervision.
The two anatomically distinct axes must not be interchanged.

For adjacent S/I slice pairs where **both** ground-truth union areas are at
least 10% of that case's maximum slice area, hard-mask overlap Dice across
4,366 training pairs had 1st/5th/50th/95th percentiles of
0.620/0.715/0.841/0.913. For A/P pairs (7,710), they were
0.567/0.733/0.895/0.956. The high typical overlap makes an interior
continuity hypothesis plausible; the lower tail means an aggressive fixed
smoothness target would contradict genuine anatomy. These are pooled-pair
descriptives, not calibrated patient-level guarantees.

## Candidate LTN-style rule

The nearest honest translation is **conditional inter-slice continuity of
the whole hippocampus**. It is an extrapolation motivated by the paper's
adjacent-slice context; it is not a loss proposed or tested by the authors.

Let `P = softmax(logits)[:,1] + softmax(logits)[:,2]` and let `P_z` be a slice
on the physical S/I axis after preprocessing and augmentation. For adjacent
slices, define a soft overlap:

```text
D_z = 2 sum_xy min(P_z, P_(z+1)) /
      (sum_xy P_z + sum_xy P_(z+1) + epsilon)
```

Let `G_z` be a **detached** applicability predicate derived from training GT
union masks: both slices have sufficient foreground area, neither intersects
the patch edge, and physical spacing/augmentation is valid. Fit the area
cutoff and lower overlap threshold `tau` using training cases only; the
observed 1st-percentile S/I value 0.620 is a conservative starting point,
not a selected hyperparameter. The fuzzy proposition is:

```text
for all eligible adjacent S/I slice pairs:
    G_z  ->  D_z >= tau
truth_z = 1 - G_z * (1 - clamp(D_z / tau, 0, 1))
L_rule  = mean_{G_z=1} [-log(clamp(truth_z, epsilon, 1))]
L_total = L_nnunet + lambda * L_rule
```

Aggregate per case before averaging cases, so large crops do not dominate.
Report ineligible pairs separately; an empty active set has zero auxiliary
loss and no claim of satisfaction. Use FP32 for the ratio and log. The
grounding is label-informed and is therefore a supervised geometric
regularizer expressed in fuzzy logic, **not independent anatomical
knowledge**. At inference time the rule supplies no new image evidence.
Preserve the A/P channels and their dense nnU-Net loss: this candidate
targets outer continuity, not their interface.

The current `semantic_constraints` DSL has only whole-mask scalar
primitives and one threshold per rule; it cannot express this conditional
slice-pair quantifier. This would need a new tested primitive/grounder or a
separate nnU-Net trainer variant. Do not modify the existing SwinUNETR
constraint trainer for this question.

## Decision sequence before a training claim

1. Locate the archived nnU-Net checkpoint and preprocessed fold-0 cache.
   They reside on the authorized cluster, under
   `/mnt/beegfsstudents/home/3160552/nnunet_recovery_20260920_01/`.
   They are not present in this local checkout. Reproduce the archived
   0.8934/0.9126 full-volume validation metrics before comparing new training
   experiments.
2. Freeze that checkpoint. On a training-only development subset, measure
   GT rule truth, baseline prediction violations, where violations occur,
   and correlation with union-boundary errors. Exclude patch-edge and
   naturally tapering pairs. If baseline violations are rare or unrelated to
   errors, stop; the loss has little plausible training signal.
3. Counterfactually optimize frozen prediction logits with `L_rule` plus a
   trust-region term, using training cases only. Require improved union Dice
   or boundary distance without harming A/P Dice or creating islands.
   This screens a direction; it does not prove network learnability.
4. If the direction passes, fine-tune from **identical checkpoint and
   optimizer schedule** with `lambda=0` and one preregistered nonzero
   `lambda`. Use the same data, patch sampling, augmentations, epoch budget,
   and checkpoint-selection rule. Transform the physical-axis coordinate
   with geometric augmentation or use an augmentation-compatible grounding.
5. Evaluate on a fresh participant-grouped holdout. Report case-level
   paired changes with confidence intervals for foreground-union Dice,
   anterior and posterior Dice, boundary distance, A/P swap voxels, islands,
   and worst-decile cases. The existing 52-case fold has already informed
   other project decisions and is developmental, not a new confirmatory set.

## Frozen-baseline screen, 22 September 2026

`scripts/audit_nnunet_slice_continuity.py` evaluated the saved **hard-mask
training predictions** from the archived stopped checkpoint. It read all 208
fold-0 training cases on the cluster and returned aggregate metrics only. A
separate 1-minute-46-second GPU job (Slurm 665738) exported the same
checkpoint's training probability maps (89 MB, retained on the cluster under
`nnunet_recovery_20260920_01/continuity_probabilities_20260922/`). Each
probability map's argmax was checked against its hard segmentation after
axis alignment. The
reproduced mean training foreground-union Dice was 0.9236, consistent with the
earlier full-volume training audit (0.9238 pooled; aggregation differs).

| Training GT threshold | Eligible S/I pairs | GT violations | Hard-mask violations | Soft-probability violations | Soft-only violations vs GT | Soft violation rate vs union error, case Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1st percentile, 0.6203 | 4,366 | 44 (1.0%) | 61 (1.4%) | 57 (1.3%) | 33 | 0.116 |
| 5th percentile, 0.7155 | 4,366 | 219 (5.0%) | 236 (5.4%) | 210 (4.8%) | 63 | 0.134 |

The median case has **zero violations** at the conservative threshold. At the
5th-percentile threshold, the soft baseline violates **fewer** pairs than GT
(210 versus 219), while 5% of genuine anatomy already violates the rule.
At the conservative threshold, soft-violating pairs have more local voxel
error (0.330 versus 0.164), but the case-level association is weak. The soft
prediction 1st/5th/median overlap percentiles are 0.595/0.718/0.841,
compared with 0.620/0.715/0.841 for GT: their typical overlap is nearly
identical. These are paired descriptives, not an independent validation
estimate or a causal test of a gradient step. Per-case results and exact
settings are in the four `continuity_audit_train*.json` files beside this
note.

**Decision:** do not train nnU-Net with this fixed-threshold overlap rule. Its
signal is too sparse at a low-contradiction threshold. Increasing the
threshold increases violations of correct ground truth without exposing a
clear excess of baseline violations, so the frozen
screen does not justify the compute or the risk of over-smoothing. This does
not rule out an image-conditioned local continuity teacher or a different
anatomical constraint. Counterfactual repair and network fine-tuning remain
untested; the screen does not prove that the proposed loss has zero gradient
or zero possible benefit.
