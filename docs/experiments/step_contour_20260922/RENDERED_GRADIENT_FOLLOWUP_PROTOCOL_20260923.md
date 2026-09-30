# Separating rendered-loss training from head-gradient feedback

Status: proposed next training experiment; the gradient-routing implementation
is tested, but these training arms have **not** been launched. This protocol
follows the exploratory fitting-batch findings and is internal development.

## Question and matched interventions

Does the rendered segmentation loss help the voxel network while its gradient
through shared function-head parameters compromises thin-versus-empty
discrimination?

Start every arm from the same saved E checkpoint in each fold. Use the same
fitting cases, batches, augmentation draws, optimizer reset, learning rate,
six-epoch duration, original auxiliary losses, and raw/rendered DiceCE mixture.
Freeze renderer strength beta identically across all arms at its saved value.
Use three continuation seeds per fold; these do not replace independent
backbone-training seeds. The same four development folds remain contaminated
by prior method design and are not a new confirmation cohort.

| Arm | Segmentation training | Renderer gradient routing |
|---|---|---|
| J | Existing normalized rendered + raw-auxiliary loss | Full joint feedback |
| H | Same loss and forward values | Stop only function-head parameter gradients from rendered loss; preserve direct and indirect input/backbone gradients |
| V | Same loss and forward values | Stop all geometry/presence/strength routes from rendered loss; retain its direct voxel-logit gradient |
| R | Raw DiceCE, same aggregate segmentation coefficient | No rendered-loss gradient; head still receives original explicit supervision |

All heads still receive the existing presence, conditional-edge, position,
ordering, and responsibility objectives where those objectives have a path.
The explicit head losses must be computed through the ordinary, trainable head
even for H. For H, a separate functional call with detached parameter values
implements the rendered-loss branch. Detaching geometry tensors would instead
implement V and remove indirect backbone learning too.

`experiments/presence_decisive_20260923/renderer_gradient_routes.py` implements
J/H/V. Its synthetic test checks identical forward values, identical raw and
decoder-feature gradients between J and H, zero rendered-loss head-parameter
gradients for H, and preservation of only the direct raw gradient for V. This
verifies routing; it is not evidence of a trained-model advantage.

## Predictions and falsifiers

- H versus J: if rendered head feedback causes the observed tradeoff, H should
  improve thin-voxel/overlap recall at matched false-positive burden and improve
  the actual post-update separation of true-thin and hard-empty presence.
  Consistent failure across continuation seeds would reject this remedy at the
  tested checkpoint and training horizon.
- H versus V distinguishes indirect backbone learning through the function
  from head-parameter feedback. A difference cannot be attributed to rendered
  forward values, which are matched before learning.
- V versus R tests whether the rendered loss supplies useful direct voxel
  training, independently of gradients through the function head.
- Record one actual AdamW step on fixed diagnostic batches, including before/
  after mean logits, per-ray signs, clipping and optimizer state. The current
  plain-SGD projections are not a substitute for this check.

## Selection, false positives, and evaluation

Freeze all selection rules before training. Select on the existing inner sets
only; keep original E as an explicit fallback. Primary endpoint: correctly
overlapping thin-ray recall subject to no increase in mean empty-ray FP count
or FP voxel count, no loss in thin-voxel recall, and ASSD no worse than E by
more than the previously specified 0.002 mm development margin. This margin is
an operational screen, not a clinical noninferiority margin.

Report raw and final ASSD, directional distances, Dice for both anatomical
classes, thin-voxel recall, missed thicker rays, calibration, and case-level
harms alongside the primary endpoint. Lock checkpoints before outer
evaluation. Do not choose a successful fold or seed for the headline result.

A separate calibration control should use a prespecified monotone search on
the inner set to match both FP burdens, including original E as a bias-only
control. The preceding capacity experiment's limited offset range cannot
exclude a benefit under a more conservative offset. Label any such follow-up
as exploratory rather than silently replacing the original selection rule.

If H only moves true and empty presence together, the next architecture change
should decouple existence classification from correction strength and include
hard-empty fitting supervision. A width increase alone does not test that
mechanism. A genuinely new labeled cohort must remain sealed until the final
architecture, calibration, preprocessing, seeds/ensemble, selection rule,
primary metric, FP guard, acceptable harm margin, and exclusion/empty-mask
handling are frozen.
