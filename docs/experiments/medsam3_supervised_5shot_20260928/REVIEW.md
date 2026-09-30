# Scoped review

Reviewed the new supervised loss bridge, matched training runner and single-job
launcher, including an independent adversarial code review using the review skill.

Fixed before submission:

- Cyclic native slice pairing preserves uniform expected slice exposure while
  anatomical edge faces explicitly exclude wraparound.
- Removed probability clipping that suppressed corrective bands gradients at low
  confidence. Stable foreground/complement log fields preserve finite gradients
  at logits of both -100 and +100.

Validation before submission: four new estimator/projection tests pass (including
full-volume value and gradient equivalence across four label configurations),
plus thirteen retained training/inference helper tests. Source compiles.
The independent reviewer reports no unresolved scoped findings. GPU calibration
is the integration check for simultaneous slice graphs and native backward calls.

Checked that all arms reset the adapter, RNGs and optimizer, replay the same
schedule, use training-only coefficient calibration, and finish all predictions
before evaluation labels are opened. No inference refinement is used.

Post-run: job 673386 completed all three arms. GPU calibration and the independent saved-prediction/training audit passed. No unresolved findings remain.
