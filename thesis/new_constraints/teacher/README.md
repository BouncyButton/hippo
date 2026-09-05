# Stop-gradient translation teacher

This candidate trains the unshifted prediction toward the model's own averaged,
inverse-mapped translated predictions. It uses exact integer translations and
keeps the existing supervised objective. Its implementation and numerical checks
do **not** establish a Dice improvement.

For logits `z`, temperature `T > 0`, and a selected translation `s`, define:

```text
p(v)   = softmax(z(x, v) / T)
q_s(v) = inverse_s(softmax(z_eval(translate_s(x)) / T))(v)
m_s(v) = 1 if translating and restoring retains voxel v, otherwise 0
n(v)   = sum_s m_s(v)
q(v)   = stop_gradient(sum_s m_s(v) q_s(v) / n(v)), for n(v) > 0

case_KL = T² / number_of_valid_voxels * sum_{v: n(v)>0} sum_c q_c(v) log(q_c(v)/p_c(v))
loss    = mean_cases case_KL
truth   = exp(-case_KL)
```

The teacher averages **probabilities**, after inverse mapping. At each voxel it
averages only the available views. Voxels absent from every view contribute
neither loss nor denominator. The final reduction gives every valid voxel equal
weight inside its case and every case equal weight in the batch. KL includes all
classes, including background; large background regions can dilute a localized
foreground gradient, so gradient calibration and stratum reporting matter.

The default sampling set contains the 12 shifts of ±1 and ±2 voxels along each
of the three axes. Training samples `num_views=2` distinct shifts without
replacement from an explicitly supplied independent CPU `torch.Generator`. The
identity prediction is absent. `shifts=` selects deterministic views explicitly
and may use all 12 for evaluation. Explicit identity is supported for a numerical
diagnostic, but the configured sampling set rejects identity.

## API

```python
import torch
from thesis.new_constraints.teacher import TranslationTeacherKLLoss

constraint = TranslationTeacherKLLoss(num_views=2, temperature=1.0)
generator = torch.Generator(device="cpu").manual_seed(109)
student_logits = model(images)
result = constraint(model, images, student_logits, generator=generator)
total_loss = supervised_loss + weight * warmup_scale * result.loss
total_loss.backward()

# Fixed evaluation views; no sampling-generator consumption.
result = constraint(model, images, student_logits, shifts=constraint.shifts)
```

`result.value` contains each case's temperature-compensated KL, and `result.truth`
contains `exp(-result.value)`. The optimization loss is the mean KL itself, **not**
`1 - truth`; optimizing the latter would reintroduce an exponential gradient
attenuation when KL is large. Student `log_softmax`, teacher `softmax`, and loss
reductions use float32, including when supplied student logits are float16.
Teacher probabilities are detached even when supplied through the cache API.

`details["metrics"]` exposes the per-case `TEACHER_METRICS`: `raw_loss` (the
temperature-compensated KL), `kl_unscaled`, `valid_fraction`, and
`mean_valid_views`. The latter averages view counts over the whole spatial grid,
including uncovered voxels. `confidence_weighted_agreement` and
`confidence_adherent` preserve the trainer's diagnostic interface. Agreement is
the mean linear soft Dice over foreground classes, evaluated at temperature `T`;
its default adherence threshold is 0.90. It is confidence-sensitive and differs
from the KL truth. Neither measures agreement with ground truth.

Teacher forwards run under `torch.no_grad()` with every module temporarily in
evaluation mode. The context restores every module's original training flag,
including mixed train/eval submodules, and registered buffers. It preserves the
PyTorch CPU RNG plus participating CUDA/MPS RNG states. This isolates the extra
teacher forwards from ordinary student dropout, BatchNorm, and stochastic
training streams. The independent shift generator alone advances when sampling.
If `base_logits` is omitted, the ordinary student forward runs first in the
model's original mode and has the normal training side effects.

## Frozen-logit audit

Cache unshifted logits and separately computed shifted logits from the same
checkpoint, cases, precision, and padding convention. Keep the teacher fixed
throughout each repair trajectory:

```python
teacher = constraint.build_from_cached(shifted_logits, shifts)
student = cached_base_logits.float().detach().clone().requires_grad_()
result = constraint.loss_from_teacher(student, teacher)
gradient, = torch.autograd.grad(result.loss, student)
```

`build_from_aligned(probability_views, valid_masks, shifts=...)` also accepts
already inverse-mapped, temperature-softened probabilities. It validates binary
coverage masks and normalized class distributions on their valid support.

Before a long training run:

1. On deterministic **training** cases, compare the translated teacher's decoded
   Dice/error strata with the student's using identical valid masks, precision,
   and pooling. Report FP, FN, and A/P swaps separately. An oracle teacher is only
   a numerical upper-bound control.
2. Report teacher KL, coverage, and student logit-gradient RMS by case and error
   stratum. Check the gradient on confidently wrong student voxels; a nonzero KL
   gradient is not evidence that the teacher knows the correct class.
3. Apply a preregistered grid of frozen student-logit updates, retaining the same
   cached teacher. Compare matched update magnitudes with the supervised loss and
   existing constraint. Report per-case and pooled results separately. This audit
   probes local reachability, not training efficacy.
4. Calibrate an auxiliary weight against the **actual** supervised objective for
   that run. Dice-only and Dice+CE controls require separate calibrations and
   matched comparisons. Record the source, checkpoint, case list, shifts, `K`,
   temperature, denominator, and precision before evaluation.

The implementation supplies these audit primitives; it does not run an empirical
checkpoint audit or choose a calibrated weight by itself.

## What this does and does not imply

For a fixed teacher, the logit derivative is proportional to `T * (p - q)` rather
than an additional `p * (1 - p)` factor. A student with logits ±1000 still gets a
finite corrective gradient if its teacher disagrees. This is a concrete
counterexample to the claim that every smooth loss is inert on saturated logits.
It does **not** imply that Dice+CE necessarily softens probabilities.

The teacher can nevertheless become confidently wrong and translation-invariant.
Then `p == q`, KL and its gradient are zero, and the error persists. There is no
guarantee of non-saturation, teacher improvement, reproduction of an inference-time
TTA gain, or complementarity with TTA. A two-view stochastic teacher is also
different from a 13-view inference ensemble that includes identity. Those are
empirical questions for matched Family B runs, not consequences of the formula.

CPU checks:

```bash
rtk proxy /Users/filippofocaccia/anaconda3/bin/python3 -m pytest \
  thesis/new_constraints/teacher/test_translation_teacher.py -q
```

Tests cover inverse translations, view-count normalization, excluded borders,
case-balanced denominators, no teacher gradient, exact-equivariance consistency,
corrective saturated-logit gradients, temperature scaling, a wrong fixed point,
float16 inputs with float32 evaluation, deterministic distinct sampling, and
mode/buffer/RNG restoration on success and failure.
# Follow-up: fixed overlap

`--teacher-support common` fixes valid support to the intersection across all
configured translations. Uniform two-view KL then has the full teacher's
expected student-logit gradient on that support; it still has sampling variance.
The default remains `union` to reproduce the earlier audit. The CUDA-only
`python -m thesis.new_constraints.teacher.calibrate_training` calibrates its scale
on 32 training cases from the matched five-epoch Dice control, with no optimizer
or validation evaluation. See
[`NEXT_PROTOCOL.md`](../../../experiments/equivariance_family_b_20260905/NEXT_PROTOCOL.md)
for prepared launch commands, controls and limitations.
