# Prospective fourth-fold follow-up

The user requested another data fold with the same optimization seeds after seeing mixed results on folds 0–2. This follow-up adds fold 3, the next unused existing dataset fold, without selecting based on outcomes or replacing fold 2. It is an exploratory extension triggered by prior results, not an independent preregistered confirmation.

## Fixed data and training

Seeds: 17, 83, 191. Arms: native MedSAM3 loss; native + supervised bands; native + supervised bands + edge. Keep the pretrained base frozen and fine-tune the same medical LoRA adapter from its original initialization. Preserve 200 optimizer updates, two adjacent-slice pairs per update, learning rates, warmup, objective definitions and inference settings.

Five training volumes, chosen deterministically with selection seed 20260932: hippocampus_040, hippocampus_160, hippocampus_178, hippocampus_216, hippocampus_222. The eligible training pool contains 46 volumes after excluding every prior training, evaluation and pilot case; this prevents introducing training/evaluation overlap in the four-fold analysis. No image content, label shape or segmentation score selected these cases.

Evaluate all 44 eligible volumes in source fold 3 after excluding prior support and pilot volumes. Exact case lists and binary data hashes are frozen in protocols/study.json. Combined totals: 20 distinct support volumes and 189 distinct evaluation volumes. Patient identities remain unavailable, so no patient-independent generalization claim is possible.

## Exact shared calibration for the new fold

The training-only calibration algorithm, calibration seed 20260929, slab selection and gradient-ratio targets are unchanged. The operational correction is to calculate calibration once in fold3_seed17, save it, and reuse the exact report and coefficients in seeds 83 and 191. This avoids independently recomputing coefficients that drifted numerically in the original study. It is a disclosed implementation difference from the previous three folds.

The saved calibration binds the five training volumes and file hashes, calibration settings, base-checkpoint hash, and exact initial adapter/frozen-parameter hashes. Reuse requires a completed independently audited source comparison; the calibration file must equal its audited report and match its SHA256. Validation scores never choose or modify coefficients. All three new seeds must have exactly equal calibration reports; even the option for reporting earlier calibration deviations cannot waive this requirement.

## Analysis and retention

Complete all nine new fine-tuning runs regardless of intermediate results. Retain and report fold 2 and every other prior comparison unchanged. Show the new fold separately and all four folds together. The combined estimator remains equal-fold/equal-seed, using the existing paired crossed-bootstrap procedure and 20,000 draws. The exact four-fold sign-flip test has minimum two-sided p=0.125. Repeated predictions are not independent volumes; inference is exploratory, and the old calibration deviation remains disclosed.

The prior nine comparison directories are linked read-only in practice: this worker verifies and skips them; it writes only new results and the combined report. Original protocols, checkpoints, predictions and audits are unchanged. The combined report retains the failed historical calibration gate rather than loosening its threshold. A failed exact calibration check on the new fold stops aggregation.

One GPU allocation runs the three new matched comparisons, expected approximately four hours, with a six-hour limit. The base checkpoint is delivered to private node memory. Budget: 0.859 GiB of logical new artifacts, charged as 1.717 GiB under two-copy BeeGFS mirroring, plus a 1 GiB reserve. Minimum free quota before launch: 2.7173 GiB. Every new final adapter, mask, probability volume, training trace and audit is retained.

Monitoring remains manual as requested; no scheduled checks or wake-up automation are created.
