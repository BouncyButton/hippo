# nnU-Net seed 2 investigation

Investigation date: 2026-10-04. Campaign: Slurm job 681342.

## Conclusion

The saved seed 2 models really fail to segment the anterior hippocampus at their selected checkpoints in all three folds. The audit found no anterior-label removal, disconnected loss gradient, wrong deep-supervision output selection, or foreground-Dice mapping error. The evidence supports a seed-dependent optimization failure under this particular 327-update training recipe. It does **not** yet establish which optimization setting caused the failure.

The same seed resets initialization and the data-order/augmentation generators in each fold. Repetition across folds is therefore consistent with a shared seed effect; these are not three independent initializations. This alone does not isolate initialization from the other seed-controlled randomness.

No completed checkpoint, result, training protocol, optimizer, or learning-rate schedule was changed. No optimizer updates or new GPU training jobs were performed during this audit.

## What was checked

- Loaded the campaign's frozen trainer and verified the dataset SHA-256, checkpoint hashes, and stored configurations.
- Checked all 30 training-case entries (10 per fold, with overlap between folds). Preprocessing preserves the exact voxel counts of both foreground labels in every entry.
- Checked both foregrounds at all three active deep-supervision resolutions: 64³, 32³, and 16³. Minimum anterior/posterior voxel counts are respectively 1282/1246, 154/156, and 22/15. The 8³ head is deliberately assigned zero loss weight.
- Replayed the first epoch's shuffle and augmentation for all nine fold/seed combinations without training. All 90 augmented samples contained labels 0, 1, and 2; minimum anterior/posterior counts were 997/947. This check covers the first epoch, not every augmented sample in all 327 updates.
- Tested the actual native Dice+CE loss on a real training target. Perfect predictions scored -0.999555 versus 0.172387 with the two foreground classes swapped. Both correct-class logits receive negative loss gradients at every active scale, so gradient descent encourages both labels. A negative total Dice+CE value is expected for this implementation's negative-Dice term.
- Tested the evaluation path with perfect predictions: anterior and posterior hard Dice both equal 1. Added regression tests for the loss semantics, deep supervision, and perfect/swapped evaluation oracles.
- Audited best and latest checkpoints for seeds 0 and 2 in all folds: 12 checkpoint entries. CPU FP32 predictions on the first two validation patients in each fold independently confirm the anterior failure for seed 2. Seed 0 segments both classes on those same patients. The anterior output-head weight gradient is nonzero in every seed 2 checkpoint: norms range from 0.003309 to 0.016620. Both correct-class logit gradients also have the expected direction on the probed training patient.
- Read all 20 recorded validation checkpoints for all nine nnU-Net runs. Each recorded validation summary covers the full 52-patient validation fold. Independent FP32 reevaluation was sampled as described above, not a new full-fold evaluation.

## What the curves show

Seed 2 selected checkpoints:

| Fold | Selected update | Anterior hard Dice | Posterior hard Dice | Macro hard Dice | Anterior soft Dice at update 327 |
|---|---:|---:|---:|---:|---:|
| 0 | 213 | 0.000000 | 0.772341 | 0.386171 | 0.345634 |
| 1 | 229 | 0.000000 | 0.830654 | 0.415327 | 0.164515 |
| 2 | 213 | 0.000000 | 0.822649 | 0.411325 | 0.245163 |

The anterior probability signal increases late in training, but usually remains below background or posterior probabilities, so argmax produces no anterior region. For example, fold 0's anterior soft Dice increases from 0.009912 at update 17 to 0.345634 at update 327; hard Dice remains zero. Independent FP32 confusion matrices show anterior ground-truth voxels being assigned to background and posterior, rather than an evaluator simply omitting class 1.

Fold 2's latest checkpoint has a small nonzero anterior hard Dice of 0.001127 on the full validation fold. Thus the precise finding is zero anterior Dice in **all three selected checkpoints**, with negligible recovery in one final checkpoint; it is not literally zero for every evaluation in all three trajectories.

Checkpoint selection behaves as specified: it selects the highest validation mean of anterior and posterior hard Dice. It therefore selects updates 213/229/213 even though anterior soft Dice improves later, because those earlier checkpoints have a better hard-Dice average.

![Validation learning curves](learning_curves.png)

## Interpretation and limits

The run has 327 optimizer updates, equivalent to 32.7 passes through 10 training patients at batch size 1. Its nnU-Net-style optimizer is SGD with initial learning rate 0.01, momentum 0.99, Nesterov acceleration, weight decay 3e-5, gradient clipping at 12, and polynomial learning-rate decay compressed into those 327 updates. The rate used by the final update is approximately 0.00005456. Seeds 0 and 1 succeed under this same recipe, while seed 2's class learning is much slower.

This is a compute-matched, fixed-architecture **nnU-Net-derived baseline**, not the unrestricted native nnU-Net pipeline. Its fixed input/batch size, shared preprocessing/augmentation, short schedule, and BF16 training should remain explicit in the paper. Equal FLOPs do not themselves guarantee that an optimizer recipe is well calibrated to that horizon.

These observations make delayed or unstable class learning under the short schedule a plausible explanation. They do not prove that more updates would fix it, that the initial learning rate is the cause, or that BF16 training has no effect. FP32 reevaluation establishes that the saved weights' failure is real; it cannot rule out a training-precision effect. Initialization, data ordering, and augmentation also share a seed, so their effects have not been separated.

Before treating these as strong architecture baselines, use a controlled diagnostic within the same 116259-GFLOP cap: hold the dataset and other settings fixed, compare a predeclared learning-rate schedule and/or FP32 training against the current recipe, and separate initialization seeds from data/augmentation seeds. Account for the extra tuning runs separately and apply any selected recipe consistently across all evaluation seeds and folds. Do not replace only the failed seed or choose a successful seed retrospectively.

## Artifacts and verification

- [Raw checkpoint, data, loss, and gradient audit](audit.json)
- [All nine training curves](curves.json)
- [Curve figure as PDF](learning_curves.pdf) and [SVG](learning_curves.svg)
- Audit program: `scripts/audit_compute_matched_nnunet.py`
- Regression tests: `scripts/test_compute_matched_baselines.py`
- Full regression suite: **8 tests passed**, 45.974 seconds. Tests include actual forward/backward smoke checks for all three models, strict budget schedules, interrupted-work accounting, resume recovery, completed-campaign verification, and the new loss/evaluation oracles.
- Audit dataset SHA-256: `3311d2233b99e09080c40ca80f1035781d9c1dc9c5b109dc266c73eb52a3df01`
- Frozen training source SHA-256: `84c5b66f5a3fec10c8806aedd85296189dda7a3e73c9ef6651cdf930da94bf23`

Status: investigation complete for implementation checks; a unique causal optimization explanation remains unconfirmed. Production training code and completed campaign results are unchanged.
