# Residual translation sensitivity after augmentation

Use the four frozen pilot selected models, starting with augmented Dice. Identity plus the existing12 integer shifts(±1,±2 along each axis), inverse-mapped with valid-view probability averaging. Three-class argmax throughout. No target, predicted component selection, or validation-tuned parameter enters prediction. Score all50training/52validation cases; retain paired corrections/regressions and per-view disagreement. Record any cropped nonzero input or foreground and distinguish complete-support cases from those with clipping.

This exact input-coordinate intervention tests residual translation dependence under a fixed checkpoint. TTA improvement is an inference remedy, not evidence that a new constraint has been learned. Positive results require additional checkpoint/seed confirmation, starting with existing augmented reference models if compatible. A negative result argues against this specific residual-translation remedy; it cannot establish a noise floor.

Original inference runtime, frozen source, data/split/checkpoint hashes verified. Identity repeat measured against the original audit. No new training or persistent dense prediction caches. This is an evidence-driven follow-up to the historical unaugmented13-viewTTA benefit; that earlier result cannot establish a benefit after mild augmentation.
