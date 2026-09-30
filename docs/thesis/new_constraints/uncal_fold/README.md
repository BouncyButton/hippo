# Uncal interval constraint (experimental)

`UncalIntervalLogLTNLoss` translates the landmark hypothesis into an executable
conditional rule. `FoldIntervalTeacher` grounds it using the existing compact
union-shape + T1 + position proxy. Neither component establishes expert-verified
uncal detection or a segmentation generalization gain.

## Rule and anatomical scope

The uncal apex is the last visible folded/double-level head structure when moving
posteriorly. Its slice belongs to the anterior hippocampus. See the
[literature review](../../../experiments/uncal_literature_application_20260921/README.md)
for sources and MRI identification details.

The MRI proxy uses cross-sectional expansion, superior contour/notch change,
multiple foreground runs in vertical columns, superior protrusion change, and
intensity/gradient features above the hippocampus. A soft training-derived
positional prior also enters. These are suggestive descriptors, not an explicit
recognizer of the anatomical uncus. Coefficients are learned from label-derived
A/P cuts; this is a weakly supervised prior, not additional expert annotation.

Let `[L,U]` be an interval containing the separating plane, in millimetres. Let
`y(v)` increase anteriorly, `H(v)` denote fixed foreground, and `Near(v)` restrict
the rule to a predeclared 6-mm radius around the interval centre:

```text
∀v: H(v) ∧ Near(v) ∧ y(v) > U + margin  → Anterior(v)
∀v: H(v) ∧ Near(v) ∧ y(v) < L − margin  → Posterior(v)
```

Voxels inside the interval remain unconstrained by this auxiliary. Original dense
segmentation supervision continues to apply there. Multiple-label slices are
preserved, not overwritten by a plane. A wide interval may leave no active voxels;
that is abstention, not successful anatomical detection.

Groundings are Boolean; A/P consequents are fuzzy conditional probabilities
`softmax(logits[:,1:3])`. With product conjunction, Reichenbach implication and
geometric-mean universal aggregation over active groundings, negative log
satisfaction equals mean conditional cross-entropy on those voxels. This identity
is intentional and should not be presented as a new logical operator. The potential
added value is in the anatomical grounding and abstention, not in relabeling CE.
Average equally across active cases. Optional external reliability multiplies each
case's loss; it is not a calibrated probability supplied by this prototype.

## Avoiding trivial satisfaction and numerical failures

- The teacher, interval, coordinate map, support and reliability are detached from
  the auxiliary gradient. Keep the teacher frozen; otherwise it can move its own
  landmark to evade a penalty. Detached predicted support still changes between
  model updates if recomputed; use a frozen support teacher/cache to avoid that
  indirect escape route.
- Background receives no direct gradient. Shared model weights can still change
  the outer boundary after training, so monitor union Dice and false positives.
- Log-space computation retains corrective gradients for saturated wrong A/P
  logits and computes in FP32 for half-precision input.
- Inactive cases are reported with `valid=False` and NaN diagnostic truth. Loss and
  backward are finite zero for an all-inactive batch. Filter diagnostics by `valid`;
  do not log inactive cases as perfectly satisfied.
- Report active-voxel coverage and contradictions alongside satisfaction. High
  agreement after excluding all difficult voxels is not evidence of utility.

## Training-only calibration and evaluation

`audit_interval_constraint.py` deterministically divides the 208 fold-0 training
crop IDs into 156 fitting and 52 calibration cases (seed 20260921). It fits the
fixed compact ranker on the 156 cases and never refits it after calibration.

For reference labels, the transition interval is bounded by the earliest anterior
voxel plane and latest posterior voxel plane (`min_A_y−0.5`, `max_P_y+0.5`, ordered).
This interval leaves inconsistent/mixed slices free. For each calibration case,
the residual is the distance from the predicted plane to the farther reference
endpoint. The radius uses order statistic `ceil((n+1)*0.9)` without interpolation.
The same procedure calibrates a position-only control. The fixed radius is then
evaluated on the reused 52 validation cases with GT and both predicted supports.

The order statistic is the split-conformal construction, but a 90% guarantee is
**not asserted** here: participant independence is unverified and calibration uses
GT unions whereas deployment uses predicted unions. The existing descriptor family
was also studied on this validation fold before this experiment. Interpret this
as a development audit, not a fresh confirmatory test.

Using label-derived reference intervals, the rule has zero contradictions across
all 260 local labels and covers 95.1% of GT foreground in the ±2-mm reference band
on average. This is a check of the interval construction, which uses the labels;
it does not validate the image-derived teacher.

### Measured result and decision

The audit completed on 21 September 2026. Both teachers required a **3-mm radius**
(a 6-mm-wide uncertainty interval). The local rule extends only to 6 mm either
side of the predicted centre, so this leaves two strips of constrained tissue.

| Validation foreground | Teacher | Reference interval covered | Agreement on active GT | GT near-cut band active |
|---|---|---:|---:|---:|
| GT union | Shape + MRI | 92.3% | 95.70% | 6.7% |
| Unaugmented prediction | Shape + MRI | 86.5% | 97.36% | 10.2% |
| Augmented prediction | Shape + MRI | 92.3% | 95.26% | 4.7% |
| Augmented prediction | Position only | 98.1% | 99.59% | 8.1% |

Agreement is a case mean over active GT foreground; near-cut activity is a case
mean against GT foreground in the fixed ±2-mm reference band. The image-based
augmented rule has any near-cut activity in only 7/52 cases. Rare large landmark
errors account for substantial wrong-side supervision despite acceptable marginal
interval coverage. See `experiments/uncal_interval_constraint_20260921/RESULTS.md`,
`summary.json` and `per_case.csv` for all variants, contradictions and coverage.

**Decision: do not enable this grounding as a trusted anatomical training rule.**
The mathematical rule handles every annotation, but the current image proxy does
not satisfy the combined requirement of reliability and useful boundary coverage.
It also does not outperform the position control. Widening its interval would
largely remove the very supervision we want at the A/P cut. This experiment did
not retrain the segmenter and does not show a training/generalization benefit.

The implemented loss remains suitable for testing a better, independently
annotated landmark/visibility teacher. It must be compared with its equivalent
masked conditional CE control to establish that any gain comes from grounding.

## Use

Run the calibration audit first. It writes a portable `teacher.json` with training,
calibration and validation IDs, feature schema, standardization, coefficients,
radius and provenance warnings into the ignored experiment directory.

```bash
rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_interval_constraint
rtk proxy .venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_interval_constraint.py -q
```

Example for a single native image (the caller loads NumPy image, predicted Boolean
foreground and NIfTI affine):

```python
import json
import torch
from pathlib import Path
from thesis.new_constraints.uncal_fold.interval_teacher import FoldIntervalTeacher
from thesis.new_constraints.uncal_fold.interval_constraint import UncalIntervalLogLTNLoss

saved = json.loads(Path("experiments/uncal_interval_constraint_20260921/teacher.json").read_text())
teacher = FoldIntervalTeacher.from_dict(saved["teacher"])
anchors = teacher.ground_native(image, predicted_foreground, affine,
                               radius_mm=saved["radius_mm"])
# image/logits and all spatial anchors must now be transformed together.
# This example assumes logits already correspond exactly to the native image.
kwargs = {name: torch.from_numpy(value.copy()).unsqueeze(0).to(logits.device)
          for name, value in anchors.items()}
result = UncalIntervalLogLTNLoss(local_radius_mm=6.0)(logits, **kwargs)
loss = segmentation_loss + lambda_uncal * result.loss
```

`lambda_uncal` must be selected/calibrated on training development data; no weight
is endorsed by the consistency audit. The native grounder currently rejects
non-axis-aligned or non-1-mm RAS images. Its coordinate field is in crop-local mm,
not MNI coordinates. To pad, translate, flip or rotate the image, transform the
coordinate field and support with it; exclude padding from support. Do not reuse
a cached scalar y index after spatial augmentation. Geometric scaling also requires
an explicit physical-coordinate policy.

No GT A/P label is accepted by `ground_native`. The `LoadedCase` interface is used
internally with a dummy target that is never read during prediction. Empty or
invalid foreground raises a clear error; the calling pipeline must explicitly
skip and count such cases.

The loss is exposed as a standalone module and **not enabled in the shared trainer**.
Trainer integration requires carrying these spatial anchors through the actual
MONAI transformations and freezing a support teacher. Silently deriving anchors
from augmented A/P labels would test a different, supervised objective.

Validation: 12 interval/grounding tests passed, plus four existing boundary-audit
tests. Coverage includes mixed labels, uncertainty abstention, zero background and
outside-support gradients, corrective saturated-logit gradients in FP16/32/64,
detached anchors, coordinate transforms, padding, invalid geometry and calibration
quantiles. The saved teacher also reproduced the audited interval `[22.5,28.5]` mm
for augmented `hippocampus_017` through `ground_native`, using only its image,
predicted binary foreground and affine. This is an implementation check, not an
independent test of anatomical accuracy.

## What would establish anatomical learning?

Compare matched training runs with baseline, the same masked conditional CE using
training labels, the position-only interval, and the shape+MRI interval. Independently
annotated uncal visibility/intervals would be the stronger teacher. Report boundary
Dice, A/P swaps, union Dice, active coverage and rare large errors on fresh,
participant-grouped evaluation cases. The current code enables this experiment;
consistency alone cannot demonstrate better generalization.
