# PCGrad pilot: fold 0, three seeds

User-authorized follow-up to the completed separated-edge pilot. Exactly fold 0,
seeds 0/1/2, two matched arms per seed: sum and PCGrad. Six new models. No other
folds, weighting variants or algorithms launch automatically.

## Hypothesis and confirmation before fitting

At each seed's retained selected separated-control checkpoint, evaluate all ten
training cases in FP32 eval mode. Record parameter-gradient norms and pairwise
cosines for Dice, weighted bands, raw inner, outer and crossing edge rules.
The prespecified descriptive confirmation gate is inner-versus-crossing conflict
in a majority (at least 6/10) of cases in EACH seed. All three audits must finish
and pass before any new training starts. This is a confirmation of the sampled
pattern, not a significance test or a claim that gradient interference causes
validation errors. No validation cases enter this gate or any recalibration.

## Primary matched comparison

The prior separated arm supplies unchanged coefficients and reference data/source
hashes. Keep the same ten training/52 validation case files, no augmentation,
random initialization seeds, model, Dice supervision, unary bands, edge geometry,
equal one-third edge rule weights, AdamW 1e-4 / weight decay 1e-5, StepLR every
20 epochs / gamma .5, AMP, batch 1 and five-epoch auxiliary warmup. Maximum 75
epochs, minimum 60, patience 8, minimum delta .0005; select greatest validation
hard macro Dice. The original stopped/selected epochs need not equal 75.

For each batch define five weighted objectives:
1. Dice.
2. Scheduled existing weighted bands (inner and outer BCE remain combined).
3. Scheduled existing edge coefficient times inner-edge mean / 3.
4. Scheduled existing edge coefficient times outer-edge mean / 3.
5. Scheduled existing edge coefficient times crossing-edge mean / 3.

Both arms calculate the same five separate parameter gradients on the SAME AMP
forward graph. The sum control adds them directly. PCGrad symmetrically projects
each task gradient against the ORIGINAL other task gradients in an independently
randomized order, then sums the modified gradients. Zero gradients are skipped.
No mean reduction, adaptive loss weights, component normalization, protected task,
post-projection norm restoration, or clipping is added. Sequential projections
can reintroduce conflicts; do not claim all final pairwise cosines must be positive.

This follows Algorithm 1 of Yu et al., Gradient Surgery for Multi-Task Learning:
https://arxiv.org/html/2001.06782v4 . We implement random order explicitly as in
the paper, rather than copying the reference TensorFlow code's unused shuffle
return value. Weighted losses and sum reduction preserve the prior objective
when projection is disabled. A dedicated CPU RNG seeded 20260930+training_seed
chooses projection orders; model/data RNG streams are untouched. Its state is
saved for exact resumption along with optimizer, scheduler, AMP scaler and all
other RNG state.

Separate-gradient evaluation can introduce AMP rounding differences versus one
backward call. Therefore the primary control uses the same separate-gradient path;
the completed separated runs remain a SECONDARY historical reference, and their
learning-curve drift is reported. Test summed gradients against joint derivatives.

## Mixed precision and diagnostics

Scale each task loss with the current common GradScaler, obtain parameter
derivatives, divide by that scale in FP32, project, and restore scale before one
scaler.step/update. Any nonfinite task gradient causes the WHOLE optimizer update
to be skipped through GradScaler's overflow path. Never silently discard a bad
task. Keep grad=None for parameters unused by every task. GPU preflight tests
exercise both ordinary unscaling and overflow/skip behavior before fitting.

Projection dot products use FP64 Gram accumulation in bounded chunks. An equivalent
coefficient basis avoids repeatedly projecting large full parameter vectors.
Independent explicit-vector tests verify sequential projection and sum reduction.

Every training update records weighted pre/post task Gram matrices, projection
orders/counts, coefficients, original/projected norms and angle, AMP scale and
skipped-update status, plus case identity. This covers all ten cases every epoch.
These are actual AMP-forward parameter derivatives before AdamW, not the actual
momentum/preconditioned parameter displacement. Retain the earlier two-case FP32
probes for direct comparison as well. No online coefficient changes occur.

## Outcomes

Audit selected checkpoints on all training and validation cases. Audit validation
geometry at the same monitoring epochs (5,15,30,45,60,75 and final stop). Primary
endpoints are inner hard/soft disagreement and FN, interpreted alongside FP,
correct boundary transitions and macro/union Dice. Lower disagreement alone can
reflect uniformly wrong predictions. Report selected-checkpoint and shared epoch-60
comparisons, per-seed effects, learning curves on common epochs, conflicts and
projection magnitudes, AMP skips and compute time. Do not infer faster learning
from wall time: five backward passes have a different cost.

Paired bootstrap over 52 validation case files averages three seeds first; 10,000
resamples. This repeatedly used development fold is exploratory and intervals
condition on fitted models, excluding model-selection/new-training-set/new-fold
uncertainty and unverified patient-level dependence.

## Resources and integrity

One sequential stud GPU allocation: 3g.40gb, 4 CPUs, 24 GiB RAM, three-hour limit.
Require 1.6 GiB free at pre-submit and allocation start, .8 GiB before each arm.
Retain six selected FP32 checkpoints and metrics. Keep a full latest resumable
checkpoint for the active arm; retire its own intermediate only after selected
checkpoint and audit are durable. Preserve previous experiments and calibrations.
Hash the full isolated payload, verify prior source/checkpoint/data provenance,
and stop on mismatches. No runtime package installation or environmental change.
