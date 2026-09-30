# Edge consistency at 5% training data

User clarification after submission: **bands-only versus bands+edge is the
primary comparison**, paired within seeds 0/1/2. Dice-only is a secondary
reference. Report selected validation Dice, boundary errors and learning speed
separately; faster threshold attainment does not automatically mean higher
final performance. This clarification changes reporting priority only; the
submitted job already includes both comparisons and no source was changed.

User-authorized experiment: fold 0, model seeds 0/1/2, maximum 75 epochs and
early stopping. Three new runs; reuse the six existing seed-matched Dice and
ordinary-bands controls from low_data_noaug_seed{0,1,2}_20260917_01.

Use the exact historical ten training case files (approximately 5% of 208),
the same 52 validation cases, no augmentation, padded 64^3 inputs, batch one,
AMP, no dropout or activation checkpointing. AdamW lr 1e-4, weight decay 1e-5,
StepLR factor 0.5 every 20 epochs. Keep the historical stopping rule: patience
8, min delta 0.0005, minimum 60 epochs, maximum 75. Best checkpoint is exact
maximum validation macro hard Dice, independent of the min-delta stop reference.
Split SHA256: 7b69e54dd720b38f58f11e73302899e626ab3985d5c15d5c72c5daeb420ce8be.

Loss: Dice + min(epoch/5,1) * (historical ordinary-bands loss + lambda_edge *
signed-face consistency). Same whole-foreground signed probability differences
and uniform face averaging in both-endpoint two-band support as the completed
full-data experiment. No degree weighting and no additional head.

Preserve each seed's ordinary-bands coefficient. Recalibrate lambda_edge
using ONLY its ten training cases and retained seed-matched epoch-five Dice
checkpoint: median weighted edge/base logit-gradient RMS ratio 0.1, capped at
p95 ratio 0.5. Base is Dice plus its ordinary bands. No full-data checkpoint,
case, coefficient or validation outcome enters coefficient calibration.
Reinitialize model and RNG streams after calibration; training starts from
scratch. No historical baseline or calibration-source training is repeated.

For speed, record first crossing and first three-consecutive-epoch crossing
of each seed's historical best Dice and best bands Dice, plus fixed Dice
levels 0.70/0.74/0.76/0.78/0.80. Preserve unreached thresholds as null. Report
epochs and optimizer updates (10 per epoch), selected/final Dice, fixed-epoch
scores and mean curve score over the common observed horizon. Do not fill or
extrapolate stopped histories. Compare each seed with its matching controls
before averaging; report every seed, including failures or unfavorable results.

GPU request is 3g.40gb; historical controls used 4g.40gb. Therefore epochs and
updates can support learning-speed comparisons, while observed runtimes are
descriptive and cannot establish a hardware-controlled wall-clock speedup.
Record calibration and training-including-I/O duration separately; calibration
reuses an existing checkpoint, whose original training cost is not included
in the new job duration.

After each run, audit selected Dice/bands/edge checkpoints on all ten training
and 52 validation cases without augmentation: Dice, inner FN, outer FP, AP
swaps, corrected/introduced errors, degree strata, and surface metrics. Also
report normalized train/validation effects as in the previous full-data audit.
Three seeds share the same subset and validation cohort; they are not three
independent datasets. Validation has already been used for development and
checkpoint selection. Do not claim independent test performance.

One sequential GPU job executes all three seeds. Reserve 3.2 GiB actual quota
before starting, retaining best/latest checkpoints plus selected exports.
No deletion is required if the observed 4.14 GiB available remains available.
