# SwinUNETR constraint experiments

The training pipeline exposes three mutually exclusive experiment presets:

- `none`: the unchanged MONAI three-class Dice baseline;
- `equivariance`: consistency under exact signed 3-D translations;
- `bands`: balanced foreground/background supervision immediately around the
  ground-truth outer contour.

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
├── constraint_result.py
├── objective.py
├── train_swinunetr_constraints.py
├── run_new_constraints_cluster.sh
└── translation_equivariance.py  # compatibility import only
```

The root `translation_equivariance.py` is only a compatibility import; the
implementation is owned by `equivariance/`. Existing equivariance reports and
plots are stored under `equivariance/results/`.

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

## Running the three arms

```bash
sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set none

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set equivariance

sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight CALIBRATED_WEIGHT
```

The bands launcher deliberately requires an explicit calibrated weight. See
`bands/README.md` for the training-only gradient-calibration command.

All arms retain the same model constructor, dataset split, optimiser,
scheduler, checkpoint selection, segmentation evaluation, RNG checkpointing,
and output structure. Equivariance uses an independent RNG for translation
sampling. Bands need no additional model forward pass but require the final
transformed labels. `none` performs no auxiliary training or constraint-
evaluation forward passes.

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

## Tests

```bash
PYTHONPATH=. python -m pytest \
  thesis/new_constraints/equivariance/test_equivariance.py \
  thesis/new_constraints/bands/test_outer_boundary.py
```

The tests cover exact translation behavior, RNG restoration, exact 6-connected
band geometry, grouped log-odds, balanced patient/side reductions, correct
gradient directions, zero auxiliary gradients away from the band, AP-swap
invariance, invalid-patient handling, and FP16-safe connected zeros.
