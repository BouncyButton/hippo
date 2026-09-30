# Bands + edge: additional folds 1 and 2

Continue the unweighted bands+edge experiment completed on fold 0. Six new models: folds 1/2, seeds 0/1/2. Reuse completed fold/seed-matched Dice and bands controls and dedicated epoch-five calibration checkpoints; do not repeat their training.

Use the historical fixed ten-case subsets and 52 validation cases for each fold, with no augmentation. Preserve maximum 75 epochs, minimum 60, patience 8, minimum delta 0.0005; AdamW 1e-4, weight decay 1e-5, StepLR 20/gamma 0.5, batch 1, AMP, 64-cube padding, no resize, and five-epoch auxiliary ramp. Models train from scratch. The historical model implementation is frozen, SHA a95420e28c84fe682d737eee59aeecd9e61e4eb3efdda6893ecf7a3936ddd7fc.

Keep each fold/seed's existing bands coefficient; calibrate the uniform signed-face consistency coefficient from its own retained epoch-five Dice model on its ten training cases using the same median 0.1 / p95 cap 0.5 gradient-RMS rule as fold 0. No degree weighting, no auxiliary head, no validation calibration. Data-loader fold, calibration fold, reference export fold and split hashes must agree. Fold 1 split SHA d4c2a53da8b0dd60839cd9afc6320a298c46bce22dd6a4b7907b95f04dace086; fold 2 b3d55f8970bdaa93f123a7003a862fe7708f664642b7db823ee3c5bf974faccf.

One sequential GPU job on 3g.40gb, 4 CPUs/24 GB, 90-minute limit. Compact best/latest inference checkpoints omit optimizer state and redundant exports. They cannot resume training exactly; this does not change optimization. Check quota before submission for the full six-run allocation and before every seed for atomic checkpoint writes. Preserve all existing unweighted models and baselines.

Automatically audit every selected model on training and validation: Dice, boundary FN/FP, anterior/posterior swaps, degree-stratified errors, corrected/introduced errors and surfaces. Compare learning curves in epochs and optimizer updates, including fixed thresholds and first/three-consecutive crossings of reference best scores. Hardware/checkpoint-I/O differences preclude a total-runtime speed claim. Keep fold-specific summaries; repeated seeds are not independent validation cohorts. These development folds are not an independent test set.

Cleanup authorized by the user removes the just-completed degree-weighted A experiment and its failed precursor, including local result reports/payload and their remote run directories. Record paths and byte counts in CLEANUP.json; do not delete unrelated experiments or references. Implementation files remain reusable.
