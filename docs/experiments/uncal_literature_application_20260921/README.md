# Uncal apex: literature, MSD experiment, and LTN proposal

Research date: 21 September 2026. This is a targeted primary-source literature
review and an exploratory application to existing local predictions. It is not a
systematic review, an expert landmark annotation, or a completed training trial.

## Main conclusion

The anatomical premise is correct: MSD explicitly used the uncal apex to mark the
last slice of the hippocampal head. A better independently grounded landmark could
substantially improve the A/P boundary. However, the available handcrafted locator
does not yet reliably outperform the segmentation model, and the augmented model's
frozen-feature probe reproduces its native cut in all 52 validation crops. Injecting
that probe as an anatomical teacher would add little new information.

I ran a new counterfactual evaluation that keeps the predicted hippocampus foreground
fixed and changes only its A/P partition. With the augmented checkpoint, the
label-derived oracle raises mean boundary-band Dice from **0.7140 to 0.9025**.
The handcrafted shape+MRI locator reaches **0.7408**, but its case-bootstrap interval
for improvement includes zero and whole-region A/P Dice decreases. This motivates
an independent landmark detector and a soft local constraint, not a hard replacement
of all A/P labels.

## What the literature actually supports

| Source | Relevant finding | Consequence for this project |
|---|---|---|
| [Simpson et al., 2019, MSD data descriptor](https://arxiv.org/html/1902.09063v1#S2.SS1.SSS4) | Task04 uses 1-mm T1 MPRAGE. The last head slice contains the uncal apex; posterior merges body and tail. | The label interface is a weak proxy for the landmark slice, not an explicit 3-D landmark annotation. |
| [Woolard & Heckers, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3289761/) | Original anatomical/functional anterior–posterior volumetry study, cited by MSD's tracing protocol; uncus formation involves anterior hippocampal folding. | Inspect a 3-D folding transition, rather than expecting a tissue-intensity boundary between two hippocampal classes. |
| [Lerma-Usabiaga et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27159325/) | Separates manual landmark placement from automatic PCA/Bézier alignment and percentage-based partitioning. Orientation and manual placement both contribute variability. | PCA and a fixed percentage are useful controls, but do not demonstrate patient-specific apex detection. |
| [Poppenk, 2020](https://onlinelibrary.wiley.com/doi/10.1002/hipo.23196) | Identifies the last visible uncus going posteriorly, using sagittal folding and coronal double-level morphology. Apex position changes with age across 4,434 hippocampi. | Use adjacent slices and orthogonal views. Treat relative position as a defeasible prior, especially across populations. |
| [Canada et al., quality-control guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC10705396/) | Reviews landmark visibility, including the uncus, in high-resolution T2 subfield segmentation. | Add an explicit visibility/abstention decision. T2 internal landmarks cannot simply be assumed visible in MSD's T1 crops. |
| [Rekik et al., 2026, EMBC paper/preprint](https://arxiv.org/html/2605.14221v1) | Uses separately annotated landmarks, global-to-local detection and anatomical rules to split coarse subcortical regions. | Supports the architectural idea of landmark-guided segmentation. Its landmark list does **not** include the uncal apex; its performance is not evidence for an MSD apex detector. |

The targeted search did not establish an off-the-shelf uncal-apex detector validated
on MSD. The existing automatic longitudinal-axis method is often easy to misread as
such a detector: its automatic partition is percentage-based.

## How to identify the landmark on MRI

1. Confirm orientation and inspect coronal slices together with sagittal context.
   In the local native arrays, axis 1 increases anteriorly. Scrolling anterior to
   posterior therefore means **decreasing y**. RAS storage does not by itself prove
   AC–PC or hippocampal-axis alignment.
2. Follow the folded anterior hippocampus across successive slices. The useful
   evidence is the continuation and disappearance of the superior/medial folded
   component, seen as a double-level structure coronally and folding sagittally.
3. Record the most posterior slice on which the uncus remains visible. In the MSD
   convention this slice belongs to anterior/head. The next posterior slice starts
   body/posterior. Keep the landmark-slice coordinate distinct from the separator
   plane between voxel centres.
4. Record uncertainty or nonvisibility. A notch, widening cross-section or superior
   intensity edge alone is not a validated apex criterion. Do not assign a medial
   side to these crops without reliable hemisphere information.

For our 1-mm RAS arrays, define c as the first anterior index in increasing array
order: posterior y<c; anterior y>=c; the separating plane is y=c−0.5. The biological
landmark is 3-D; this experiment estimates only its A/P slice coordinate.

## Data and geometry checked in this session

- 260 local labeled crops, with 208 training and 52 validation case IDs in fold 0.
  All 260 linear affine parts are identity and native spacing is 1 mm. The published
  MSD challenge lists 263 training cases; why three are absent locally was not
  resolved here. Results describe the local 260-case cohort.
- **58/260** labels are not representable by one exact axis-1 plane. Their optimal
  planes disagree with **1,388 voxels** in total. Preserve the original mixed slices
  for evaluation; do not silently replace the reference masks with planes.
- Each cached ground truth was cropped back from the padded 64³ grid and checked
  for exact equality with its native label. Native and padded cut offsets were
  verified separately for every case and model.
- These are crop/case splits. The code's `subject_id` is a filename identifier.
  A mapping of left/right crops to original participants was not available in the
  inspected metadata. Participant-level independence is therefore **unverified**;
  earlier reports calling these splits subject-wise should be read cautiously.
- The affines do not establish MNI registration. A fixed MNI y coordinate must not
  be applied directly to these cropped arrays.

## New application: does the landmark actually improve segmentation?

Inputs: existing early-stopping seed-0 baseline and augmented predictions; existing
frozen feature-probe cuts; existing compact shape+T1+soft-position cuts. No model
was fitted, no validation threshold selected, and no checkpoint modified here.
Original foreground support, including small islands, is identical across all
interventions. Native cut localization uses the largest component, as in the
previous audits, but corrections preserve the full original support.

The band is fixed at ±2 mm around the label-derived separator plane, i.e. four
coronal slices at 1-mm spacing. Band Dice averages the two class Dice scores and
includes outer-boundary errors. A/P swap rate is reported separately, conditional
on foreground shared by prediction and reference. Each case receives equal weight.

| Augmented model, 52 crops | Cut MAE (mm) | Whole A/P Dice | Band A/P Dice | Band A/P swap rate |
|---|---:|---:|---:|---:|
| Original segmentation | 0.904 | 0.8866 | 0.7140 | 18.43% |
| Repartition at its own cut | 0.904 | 0.8872 | 0.7189 | 17.64% |
| Repartition at frozen-feature cut | 0.904 | 0.8872 | 0.7189 | 17.64% |
| Repartition at shape+MRI cut | 1.115 | 0.8786 | 0.7408 | 16.31% |
| Repartition at training-median relative cut | 1.038 | 0.8830 | 0.6528 | 22.17% |
| Repartition at label-derived oracle cut | 0.000 | 0.9061 | 0.9025 | 1.47% |

The shape+MRI band improvement is +0.0268, with a descriptive paired case-bootstrap
95% interval of [−0.0100, +0.0666]. Its global Dice reduction and 19-slice maximum
localization error in the earlier audit make hard enforcement unattractive.
The feature result is identical to simply planarizing the native augmented cut.
The unaugmented model also shows no clear gain: feature correction changes band
Dice by +0.0033 [−0.0355, +0.0430].

The oracle's augmented band gain is +0.1885 [+0.1438, +0.2358]. This is useful
headroom evidence, **not** a forecast of a learnable gain. The oracle uses validation
labels and a best-fitting plane; it is neither an independent anatomical annotation
nor a strict upper bound over arbitrary nonplanar segmentations. The residual swaps
reflect mismatch between the plane representation and original labels.

Repeated inspection of this same validation fold makes all results exploratory.
Bootstrap intervals use crops, not known independent participants. No external-site
generalization or training benefit has been demonstrated.

## A concrete LTN formulation to pursue

The missing grounding is an independent, uncertainty-aware image predicate.
Use a small 2.5-D detector with adjacent coronal slices and sagittal context to
estimate q(c|I), a distribution over the last visible-uncus slice, plus a visibility
score r. Supervise it initially with blinded expert slice/interval annotations.
MSD A/P interfaces can provide weak targets, but supervision from them alone should
be described as structural auxiliary learning, not new anatomical knowledge.

One interpretable intermediate predicate is `FoldVisible(s)`: is the folded upper
component visible here? Its posterior disappearance defines a candidate cut.
Require sequence context; a scalar whole-mask curvature or solidity score is too
ambiguous. Mask-derived features may supply context, but avoid requiring a perfect
GT union at deployment. Train and test with predicted-support perturbations.

For voxel v, let d(v,c) be signed distance in mm from candidate plane c, positive
anteriorly. Let δ define a tolerance band for mixed labels and annotation uncertainty,
and τ control spatial softness:

    A(v) = Σ_c q(c|I) sigmoid(( d(v,c) − δ)/τ)
    P(v) = Σ_c q(c|I) sigmoid((−d(v,c) − δ)/τ)

Use q normalized across all supported candidates; existing logistic peak scores
are not calibrated landmark probabilities. Select δ, τ and calibration on training
development data. Do not reinterpret a peak score as visibility confidence.

Let H(v) be detached trustworthy hippocampus support, Q(v) a fixed/detached local
band around the predicted landmark distribution, and r an independently trained
visibility/reliability gate. Define conditional segmentation probabilities
π_A,π_P = softmax(z_A,z_P). Candidate fuzzy clauses are:

    ∀v: [H(v) ∧ Q(v) ∧ Reliable(I) ∧ A(v)] → Anterior(v)
    ∀v: [H(v) ∧ Q(v) ∧ Reliable(I) ∧ P(v)] → Posterior(v)

With product conjunction and Reichenbach implication, each clause truth has form
1−w(v)(1−π_class(v)). Aggregate within active support and then equally across cases
(for example p-mean error). Inactive cases should contribute a differentiable zero
loss and be counted, not dominate satisfaction through background vacuity.
Always specify the exact fuzzy operators when comparing LTN implementations.

Prevent trivial satisfaction: freeze/detach the landmark teacher and support for
this loss, or train them with separately supervised objectives. Otherwise the model
can shift the landmark, shrink support or lower reliability to evade the rule.
Conditional class probabilities remove the direct background-logit gradient, but
shared network updates can still alter foreground geometry; monitor union Dice.

This is a proposed grounding, **not yet wired into the training objective**. The
counterfactual evidence does not justify activating the present locators as hard
constraints. It also does not rule out a soft learned landmark auxiliary.

## Next experiment and acceptance criteria

1. Annotate approximately 30–50 **training** crops, selected before viewing model
   errors, with two blinded raters: last visible-uncus slice, uncertainty interval,
   and visibility. Include both hemispheres if original metadata can be recovered.
   Measure rater agreement; resolve coordinate conventions before model fitting.
2. Verify participant grouping and checkpoint-selection provenance. Reserve fresh
   cases or a participant-grouped outer fold before detector or LTN tuning. The
   reused fold 0 is development evidence now, not a fresh test set.
3. Compare an image-only landmark head, a positional control, and a weakly supervised
   A/P-cut head. Require improvement over the **native segmentation cut**, not just
   over the positional prior, with uncertainty coverage and abstention reported.
4. Compare matched-seed segmentation runs: baseline, auxiliary landmark supervision,
   LTN geometry alone, landmark-grounded LTN, and oracle landmark diagnostic. Keep
   segmentation supervision, augmentation, training budget and stopping rules equal.
5. Primary endpoint: A/P error or Dice within a predeclared physical band. Also report
   per-class global Dice, union Dice, cut MAE, tail errors and calibration. Test an
   external dataset only after confirming a compatible annotation convention.

## Reproduce and inspect

```bash
rtk proxy .venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_boundary_utility.py -q
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_boundary_utility
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.visualize_boundary_utility
```

New outputs live in `experiments/uncal_boundary_utility_20260921/`: `summary.json`
(input hashes, all-case geometry and aggregate results), `per_case.csv`, `RESULTS.md`,
`boundary_utility.png`, a blinded raw training-case coronal sheet, and a blank
`manual_review_template.csv`. The raw sheet uses union labels only to find the full
hippocampal extent; it shows no A/P labels and asserts no anatomically verified apex.
MRI-derived figures remain in the ignored experiment directory.

Validation completed: four new tests passed (cut inclusion/orientation, foreground
preservation, swap-versus-miss accounting, padding invariance and physical spacing);
the full 260-label/104-prediction audit completed; both figures were rendered and
visually inspected. This verifies the analysis mechanics, not anatomical landmark
accuracy.

Existing experiments were inspected and reused, not rerun or claimed as new work:
[foldedness](../uncal_foldedness_audit_20260921/README.md),
[refined descriptors](../uncal_foldedness_refined_20260921/README.md),
[predicted foreground](../uncal_foldedness_predicted_foreground_early_stopping_20260921/README.md),
[frozen feature probe](../uncal_feature_probe_20260921/README.md).
