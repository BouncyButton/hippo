# Edge-consistency train/validation audit

Job **667813 completed successfully** in 3m52s, exit 0, no stderr. All three
selected checkpoints evaluated on 208 unaugmented training cases and 52
unaugmented validation cases. Training job 667767 selected edge epoch 26 and
stopped at 32; existing Dice and bands selected epochs 24 and 21 respectively.

**The edge model fits training cases better, but its overall improvement does
not carry over to validation in this experiment.**

| Model | Training macro Dice | Validation macro Dice | Training boundary FN+FP | Validation boundary FN+FP |
|---|---:|---:|---:|---:|
| Dice | 0.914831 | 0.887664 | 106691 | 31321 |
| Dice + bands | 0.912926 | 0.887689 | 108900 | 31713 |
| Dice + bands + edge | 0.920056 | 0.887571 | 103275 | 31608 |

Boundary counts include foreground/background errors within the two-voxel
bands, excluding AP swaps. Compare counts between models within a split;
training has four times as many cases as validation.

Against the direct ordinary-bands control, edge increases training Dice by
0.713 percentage points (paired case bootstrap 95% interval +0.583 to +0.845),
improving Dice in 167/208 training cases. It corrects 16519 boundary errors
while introducing 10894: **5625 net fewer errors (5.17%)**.

On validation, the Dice change is -0.0118 percentage points (95% interval
-0.2458 to +0.2257); 25 cases improve and 27 worsen. It corrects 3588 boundary
errors but introduces 3483: **105 net fewer errors (0.33%)**. This small pooled
count decrease does not establish a validation improvement.

| Edge minus bands | Training | Validation |
|---|---:|---:|
| Inner false negatives | +1895 | +1239 |
| Outer false positives | -7520 | -1344 |
| Boundary AP swaps | -1454 | -85 |
| Mean inner FN rate, percentage points | +0.420 | +1.018 |
| Mean outer FP rate, percentage points | -1.113 | -0.784 |
| Balanced boundary error, percentage points | -0.347 | +0.117 |

Balanced boundary error averages the inner-FN and outer-FP rates per case.
It improves on training, but its validation point estimate worsens: the
false-positive reduction is offset by greater missed-foreground error.
Its validation difference interval crosses zero. The validation-minus-training
effect gap is +0.464 percentage points (95% interval +0.304 to +0.616).

Against Dice alone, edge has 3416 fewer training boundary errors (3.20%), but
287 more validation errors (0.92%). Training Dice gains 0.522 percentage
points; validation Dice changes by -0.0093. Inner FN decreases on both splits,
but outer FP increases on both, with a larger rate increase on validation.

The train-minus-validation Dice gap grows from 2.524 percentage points for
bands (2.717 for Dice) to 3.248 for edge. The findings support better fitting
with limited transfer on this development split, rather than failure to fit
the training cases. Some component improvements do transfer (outer FP versus
bands, inner FN versus Dice), but there is no demonstrated overall benefit.

These are single-seed, validation-selected checkpoints, at different selected
epochs. This audit does not isolate loss effects from differing optimization
durations or prove a causal overfitting mechanism. Validation was already used
for development; bootstrap intervals omit selection and seed uncertainty.
Small AMP inference variation explains the 1–2 voxel differences from the
initial validation audit. All comparisons above use this common audit run.

Aggregate results and checkpoint bindings: `GENERALIZATION_RESULTS.json`.
Patient-level counts remain on the cluster under
`/mnt/beegfsstudents/home/3160552/edge_generalization_audit_20260924_01/results`.
