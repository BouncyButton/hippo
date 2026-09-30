# Sagittal step-function fit — 22 September 2026

## Interpretation of the sketch

The unit of supervision is a **whole contour function within a sagittal
section**, not an independently corrected voxel. At fixed left-right position
`x`, the upper and lower foreground boundaries give two height functions of
posterior-to-anterior position `y`. Both the prediction and this patient's
reference annotation are fitted with decreasing isotonic step functions. The
loss compares the fitted curves across each occupied run of `y` positions.

The foreground probability is the sum of anterior and posterior class
probabilities. On each `(x, y)` ray through superior-inferior `z`, the soft
first and last foreground positions are computed from the full probability
ray. For example, the probability of a first hit at `z` is

`p(z) × product_{t<z}(1 − p(t))`.

Its conditional expected position is the predicted lower contour height; the
reverse-ray equivalent gives the upper height. Let `I↓` be decreasing isotonic
regression, and `g_x^lower(y), g_x^upper(y)` the annotated extrema. The default
geometry term is the average of

`|I↓(f̂_x^lower)(y) − I↓(g_x^lower)(y)| / Z`

and the corresponding upper-contour difference. Each contiguous annotated
`y` run is fitted independently, since some sections have real gaps. A small
ray-presence term handles missing and extra sections. A transition-location
term comparing first differences of the fitted functions is implemented but
off by default.

The projection's plateau assignments are chosen during the forward pass. For
backpropagation they remain fixed; each plateau height is a differentiable
mean of its input contour values. Gradients flow through that mean, the soft
ray endpoint, and the model's voxel logits. The construction therefore gives
a function-level correction while retaining the existing segmentation loss
for detailed anatomy and anterior/posterior class identity.

## Empirical checks

On the 52 MSD fold-0 validation reference volumes, 1,126 sagittal sections
have at least three annotated `y` positions (2,252 upper/lower functions).
The median step-fit RMS residual is 0.239 voxel; 98.6% of functions are below
one voxel RMS. About 29.9% of sections have at least one unoccupied `y` gap,
which motivated fitting each run separately. These numbers describe the
reference annotations, not generalization.

In a frozen-logit check, one equal-RMS update using Dice plus the fitted-height
loss was compared with a Dice-only update on those same 52 patients. Mean
foreground-union Dice changed by **+0.000114** relative to Dice-only; 26 cases
improved, 4 worsened, and 22 tied at this update size. Adding the separate
transition term yielded **+0.000108** with 24 improved, 4 worsened, and 24
tied. The fitted-height term is the default. These tiny one-step changes do
not establish a benefit after network retraining or on an independent cohort.

An illustrative [sagittal contour plot](../../../evaluation_output/step_contour_20260922/sagittal_step_example_338.png)
shows raw and fitted upper/lower functions for `hippocampus_338`, `x=30`.

## Implementation and reproduction

- Loss: [`sagittal_step_fit.py`](../../../thesis/new_constraints/sagittal_step_fit.py)
- Training preset: `--constraint-set sagittal_step --sagittal-step-weight <calibrated weight>`
- Reference audit: [`audit_sagittal_reference_steps.py`](../../../scripts/audit_sagittal_reference_steps.py)
- Frozen update audit: [`audit_sagittal_step_fit.py`](../../../scripts/audit_sagittal_step_fit.py)

```bash
rtk proxy .venv/bin/python scripts/audit_sagittal_reference_steps.py \
  --output evaluation_output/step_contour_20260922/sagittal_reference_fold0.json
rtk proxy .venv/bin/python scripts/audit_sagittal_step_fit.py \
  --output evaluation_output/step_contour_20260922/sagittal_step_frozen_fold0.json
rtk proxy .venv/bin/python -m pytest thesis/new_constraints/test_sagittal_step_fit.py -q
```

The next decisive experiment is a matched training run with the sagittal-step
weight calibrated on training cases only, followed by held-out Dice, contour
distance, and slice-wise plateau/transition errors. The frozen update is a
screen for gradient direction, not a substitute for that run.

## Critical review of the first loss

The current fit term sees only the projected functions. A wrong local dip and
rise can cancel when isotonic regression pools them into one plateau; the
predicted and annotated fitted functions can then agree exactly while the raw
predicted contour is wrong. On these 52 saved predictions, 21.7% of predicted
upper/lower sagittal curves are more than one voxel RMS from their own
decreasing fit, compared with 1.4% of annotation curves. The optional
`residual_weight` term compares what each projection removed from the
predicted and annotated curves; it is off by default.

A frozen-gradient ablation with `residual_weight=0.5` yielded a mean union-Dice
change of +0.000104 versus Dice-only (21 better, 3 worse, 28 tied). The current
height-only loss yielded +0.000114 (26 better, 4 worse, 22 tied). Thus the
residual term fixes a formal blind spot but has not improved this immediate
Dice screen. It needs a real training test and contour-specific evaluation.

Section weighting is another open choice. Sections with 3–9 occupied `y`
positions are 10.3% of valid sections but account for about 26.9% of the sum
of equally weighted fitted contour gaps. Those thin sections may contain
valuable tail/apex information, so any downweighting should be tested against
tail/apex errors explicitly.
