# SwinUNETR constraint experiments

The training pipeline exposes four mutually exclusive experiment presets:

- `none`: the unchanged MONAI three-class Dice baseline;
- `equivariance`: consistency under exact signed 3-D translations;
- `bands`: balanced foreground/background supervision immediately around the
  ground-truth outer contour;
- `onecut`: exact tolerance-aware one-crossing logic on spacing-aware
  ground-truth surface-normal rays.

`translation` remains accepted as a deprecated alias for `equivariance`, so
previous commands remain reproducible.

## Layout

```text
new_constraints/
├── bands/
│   ├── outer_boundary.py
│   ├── calibrate_weight.py
│   ├── test_outer_boundary.py
│   └── README.md
├── equivariance/
│   ├── translation_equivariance.py
│   ├── test_equivariance.py
│   ├── MOTIVAZIONE_EQUIVARIANZA.md
│   ├── results/
│   └── README.md
├── onecut/
│   ├── outer_onecut.py
│   ├── calibrate_weight.py
│   ├── test_outer_onecut.py
│   └── README.md
├── constraint_result.py
├── objective.py
├── train_swinunetr_constraints.py
└── run_new_constraints_cluster.sh
```

Translation equivariance is owned by `equivariance/`. The root-level
`translation_equivariance.py` compatibility shim was removed after a repo-wide
scan found no remaining importer. Existing equivariance reports and plots are
stored under `equivariance/results/`.

## Common objective

The supervised baseline is preserved exactly:

```python
DiceLoss(to_onehot_y=True, softmax=True)
```

The selected auxiliary term is added through `NewConstraintObjective` and
receives the same linear ramp used by the existing equivariance pipeline:

```text
total = supervised_dice + warmup_scale * weighted_constraint
```

With the default five-epoch setting, the scales are `0.2, 0.4, 0.6, 0.8,
1.0` during epochs 1–5. These are not five Dice-only epochs. The separate
epoch-5 `none` run required for gradient calibration is Dice-only.

The constraints are mutually exclusive in one run. This prevents a run label
from silently representing a mixture of two scientific interventions.

## Running the four arms

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set none

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set equivariance

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight CALIBRATED_WEIGHT \
  --bands-calibration-json /path/to/bands_calibration.json

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set onecut \
  --onecut-weight CALIBRATED_WEIGHT \
  --onecut-calibration-json /path/to/onecut_calibration.json
```

The bands launcher deliberately requires both the explicit calibrated weight and
the exact completed calibration report. The report is snapshotted and checked
against the dataset, fold, input hashes, source/runtime identity, class grouping,
two-step morphology, and weight before training starts. See
`bands/README.md` for the training-only gradient-calibration command.

All arms retain the same model constructor, dataset split, optimiser,
scheduler, checkpoint selection, segmentation evaluation, RNG checkpointing,
and output structure. Equivariance uses an independent RNG for translation
sampling. Bands need no additional model forward pass but require the final
transformed labels. `none` performs no auxiliary training or constraint-
evaluation forward passes. One-cut also reuses the segmentation logits; its
ground-truth distance transform only constructs physical normal rays and is not
a predicted auxiliary head.

## Outer one-cut constraint

For an ordered outward ray with grouped log-odds values
`r[0], ..., r[S-1]`, each cut candidate asserts foreground on the inner prefix
and background on the outer suffix. The exact probability mass of all cut
assignments within ±1 mm of the annotation is marginalized with `logsumexp`.
The negative log mass is normalized by the number of ray samples and then
averaged first over rays and then over patients. The canonical pilot uses a
3-mm radius, 0.5-mm samples, no margin, temperature 1, and at most 4096
deterministically selected surface faces per case.

As with bands, training refuses an arbitrary weight. `onecut/calibrate_weight.py`
measures Dice and one-cut gradients on the same deterministic training-only
cases from a fresh epoch-5 Dice-only checkpoint. The selected weight targets
10% of the median Dice logit-gradient RMS, with a 50% high-gradient safety cap.
The completed report is validated against source, runtime, GPU, data, split,
checkpoint, geometry, and case-level recomputation before training begins.

`completion_manifest.json` is published last and binds the final model, metrics,
and validation-detail artifacts by SHA-256.

## Band constraint

For foreground classes `A` and `P`, define `H = A or P`. Two iterations of
6-connected erosion and dilation give:

\[
B_{in}=H\setminus E_2(H),\qquad B_{out}=D_2(H)\setminus H.
\]

The grouped foreground logit is:

\[
r_H=\operatorname{logsumexp}(z_A,z_P)-z_0.
\]

The loss averages BCE separately on the two sides and then gives them equal
weight. Patients without both sides are excluded; an entirely invalid batch
returns an AMP-safe differentiably connected float32 zero. Training logs the
two losses, voxel counts, valid/skipped fractions, and edge-touching fraction.
The internal `ConstraintResult.truth` compatibility value is `exp(-BCE)` and is
not reported as a scientific fuzzy-truth or adherence metric; band evaluation
uses the directly interpretable raw/inside/outside BCE diagnostics instead.

## Follow-up: supervised baseline and new candidates (2026-09-05)

The original Dice-only objective remains the default. Use
`--supervised-loss dice_ce --ce-weight 1 --constraint-set none` for the explicit
Dice+CE control, and `--calibration-diagnostics` for pooled validation NLL,
Brier, ECE, entropy and saturation by foreground/boundary/error stratum.
The chosen loss and effective CE weight are bound to run and resume provenance.
Adding CE does not guarantee softer predictions or a Dice gain.

Two experimental, mutually exclusive presets are available:

- `teacher --teacher-weight WEIGHT`: detached, inverse-mapped translation
  teacher with a stable logit KL objective. See [teacher/README.md](teacher/README.md).
- `ap_cut --ap-cut-weight WEIGHT --ap-axis 1 --ap-anterior-side high`:
  training-label-anchored cut posterior. See [ap_cut/README.md](ap_cut/README.md).
  The strict all-case planar assumption fails, and subsequent official-logit
  F1–F7 audits close the tolerance-aware aggregate family too. Keep this preset
  for negative-result reproduction; do not launch it as a training candidate.

Weights require a training-only audit under the selected supervised loss.
`audit_followup.py` is a diagnostic frozen-logit tool, not a training-calibration
authorization. Legacy bands/onecut calibrations remain Dice-only and reject
Dice+CE source checkpoints. Completed protocols and active training source
directories must not be modified.

The [follow-up protocol](../../experiments/loss_constraint_followup_20260905/PROTOCOL.md)
contains the comparison matrix, endpoints, CPU falsifiers and proposed control
commands. No long training job has been launched by this follow-up.
The [A/P closure record](../../../experiments/loss_constraint_followup_20260905/A_P_BRANCH_CLOSURE_20260905.md)
supersedes earlier suggestions to develop a tolerant planar constraint.

## Tests

```bash
PYTHONPATH=. python -m pytest \
  thesis/new_constraints/equivariance/test_equivariance.py \
  thesis/new_constraints/bands/test_outer_boundary.py \
  thesis/new_constraints/onecut/test_outer_onecut.py
```

Follow-up tests:

```bash
PYTHONPATH=. python -m pytest \
  thesis/new_constraints/test_supervised.py \
  thesis/new_constraints/teacher/test_translation_teacher.py \
  thesis/new_constraints/ap_cut/test_posterior.py \
  thesis/new_constraints/test_followup_training.py \
  thesis/new_constraints/test_audit_followup.py
```

The tests cover exact translation behavior, RNG restoration, exact 6-connected
band geometry, grouped log-odds, balanced patient/side reductions, correct
gradient directions, zero auxiliary gradients away from the band, AP-swap
invariance, invalid-patient handling, and FP16-safe connected zeros.
