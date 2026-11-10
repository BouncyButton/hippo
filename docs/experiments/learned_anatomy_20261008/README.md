# Learned spatial anatomy loss: one 5% SwinUNETR experiment

The user authorized implementing an anatomy encoder, verifying that its loss detects and corrects missing tissue, extra tissue and anterior/posterior swaps, then training one 5% SwinUNETR. The native architecture remains unchanged. The objective is Dice plus one frozen learned feature distance. No depth balancing, SDF, focal weighting, profile loss, topology, component count or auxiliary MRI branch is added.

## Matched segmentation protocol

MSD fold 0, seed 0; the same ten training patients and 52 validation patients as the completed Dice baseline (74.166740% validation macro Dice). Full SwinUNETR trains from scratch: feature size 24, one MRI input, padded 64 cubed, batch size one, no augmentation or pretraining, CUDA bf16, AdamW 1e-4 / weight decay 1e-5, StepLR 20 / 0.5. Maximum 75 epochs; early stopping minimum 25, patience eight, delta 0.0005. Dice includes background. Select the highest validation anterior/posterior macro Dice. The completed baseline is reused, never rerun.

Epochs 1-5 are Dice only. At the end of epoch 5, calibrate the learned term using logit-gradient RMS on the ten training cases: median ratio target 0.1, p95 cap 0.5. Ramp the fixed coefficient over epochs 6-10. This follows the preceding profile experiments. Optimizer, scheduler, ordering and random states resume exactly; completed runs are verified and skipped.

## The learned representation

A compact three-scale 3D encoder maps candidate probabilities (background/anterior/posterior), guide ROI and original-field-of-view support to 8, 12 and 16 spatial feature channels at full, half and quarter resolution. Each level uses a learned 3 cubed convolution and LeakyReLU; no global pooling, normalization across the entire volume, raw-label identity feature or pretrained MRI encoder is used. Guide ROI is context, not a hard restriction on where tissue may be predicted. Probabilities in artificial padding are replaced by background, with zero prediction gradient there.

The encoder is trained only on the same ten training annotations. The existing frozen out-of-fold guide ROI plus 8 mm is extracted into ten tiny files; no validation labels or additional training patients are used. Prior fitting does not access MRI intensity. A decoder has three heads: reconstruct the candidate labels, reconstruct the clean labels, and detect the location/type of corruption (none, missing, extra, A/P swap). Candidate reconstruction and typed error detection preserve sensitivity to damage that a pure denoiser could learn to ignore. Each prior task uses the mean CE over classes present inside the original FOV; tasks have equal weight. These CE objectives train the prior, not SwinUNETR.

Use 1,000 fixed prior optimizer steps, seed 0, AdamW 1e-3 / weight decay 1e-5. Sample from ten patients, both tissue classes and three error types; patch radii 1.5, 3 and 4.5 mm, candidate confidence uniform 0.6-0.99, and 25% clean examples. Missing patches start at tissue surfaces, added patches at nearby background, and swaps within either class. Use the final checkpoint without validation selection. Save resumable prior state every 50 updates. Prior pretraining is additional compute compared with the baseline and does not add patient diversity.

Freeze the encoder. The segmentation auxiliary is the equal mean, over its three layers, of spatial squared feature differences between the current soft prediction and the corresponding one-hot target. Each layer is averaged over original-FOV feature sites and channels and scaled by its clean-training feature second moment. Target features are detached and cached. Frozen encoder parameters receive no gradients, but autograd traverses the encoder to the SwinUNETR logits. Decoder heads are discarded. At inference only the original SwinUNETR remains; neither encoder nor guide is needed.

## Mandatory predeclared gradient gate

After fitting, use a separate random generator (seed 20261008) for new corruptions of the ten training anatomies. This is a corruption audit on training anatomies, not an independent patient test. Radii are 1, 2.5 and 4 mm, with confidence 0.8 and 0.95. The corrected audit retains the original 360 uncertain three-class probes and adds 360 isolated two-class probes, for 720 total. Isolated probes transfer probability only between the true and wrong class, leaving the third fixed; an isolated A/P swap therefore preserves foreground probability. The original uncertain probes also change the third-class probability and remain a mandatory stress test. Other voxels are almost one-hot correct, avoiding widespread artificial soft background errors.

Audit **the learned term alone**. Dice cannot hide an incorrect auxiliary gradient. For each probe:

- The corrupted prediction must have greater learned loss than the nearly correct prediction. Exact annotation-versus-itself loss must be zero.
- Compare annotation, half-corrected and corrupted probabilities; at least 95% of probes per error type must have ordered increasing loss.
- Differentiate with respect to logits. At least 90% of corrupted voxels per error type must receive a helpful true-minus-wrong class gradient, and no probe may fall below 75%.
- Take an actual unmasked gradient descent step, with maximum absolute logit change 0.25. Every probe must lower the learned loss and increase the mean correct-label probability on its damaged patch.
- In at least 95% of probes per type, net foreground probability mass must increase for missing tissue and decrease for extra tissue. For swaps, the absolute net foreground change must remain below half the correctly relabelled probability mass. Record changes to retained tissue separately.
- Check finite values and gradients, frozen encoder weights, exact encoder hash, all ten training IDs and zero validation usage.

Each of the six error-type/construction groups must pass the fixed thresholds. The run aborts before SwinUNETR training on failure. Passing provides evidence about these audited perturbations, not universal anatomical validity. The thresholds are not relaxed after seeing results.

## Initial audit and encoder refinement

Job 689618 fitted all 1,000 prior steps, then stopped at the gradient gate before any SwinUNETR training. Missing and extra tissue passed. All A/P probes increased correct-label probability, but 66 of 120 removed too much foreground under simultaneous label/background uncertainty. Original source, protocol, audit and job records are preserved under `attempts/689618`.

The initial three-class probability construction was inappropriate as a pure A/P swap because it also reduced foreground probability. Fixing that construction alone would conceal a useful stress-test failure. The revised audit therefore includes both isolated swaps and all original uncertain probes, with unchanged thresholds. It is a development audit revisited after an observed failure, not an untouched held-out evaluation.

Reuse the fitted encoder and decoder, without repeating the 1,000 completed steps. Perform 300 additional encoder-only updates (seed 17, AdamW 3e-4, weight decay 1e-5) using newly generated training corruptions, alternating isolated and uncertain constructions, confidence 0.75-0.98. Differentiate the feature loss with respect to candidate logits, then train the encoder so this gradient aligns with the supervised cross-entropy correction direction on the damaged voxels. The objective is mean cosine mismatch plus 0.1 times the original reconstruction/error-detection objectives; decoder heads stay frozen. This is second-order encoder training. Cross-entropy is the correction teacher during encoder fitting, not an additional SwinUNETR loss.

Retain the initial training-only feature scale constants used by this gradient objective. Preserve the initial frozen checkpoint, the refinement history and a resumable refinement checkpoint. Freeze the refined encoder and repeat the full 720-probe audit. This correction explicitly addresses the observed background leakage; no successful segmentation run is repeated and the single 5% run remains gated.

After the audit, a production CUDA bf16 SwinUNETR backward/optimizer check verifies that the learned loss reaches the native network while the encoder stays frozen. Then start the one authorized segmentation run. Local unit tests cover padding exclusion, class-specific corruption semantics, gradient contracts, failure gating and exact restart before/after coefficient calibration. A real local 64 cubed SwinUNETR backward check also runs before submission.

## Reporting and resources

Each Slurm allocation requests an A100 MIG 4g.40gb, four CPUs, 24 GiB RAM and a four-hour limit. The replacement resumes after completed prior fitting: refine encoder, repeat audit, production preflight, one segmentation run. Check BeeGFS quota before submission; require at least 2 GiB free with an estimated output budget below 1 GiB. Preserve best and resumable checkpoints, per-case metrics, predictions, all original and revised audit records, coefficients, histories, source hashes and protocol. No recurring automation is created.

Compare Dice, missing/extra foreground, A/P swaps, surfaces and depth bands against the saved Dice baseline. Pay particular attention to whether a Dice gain again comes from excess-tissue suppression with worse missing tissue. Report prior-fitting time separately. This is exploratory fold-0/seed-0 evidence under a fixed 75-epoch schedule; a positive result would not yet distinguish learned anatomy from the regularization induced by an extra spatial feature metric, nor establish generalization across patients/seeds.

## Probability-step refinement after the second failed gate

Job 689653 completed its 300 updates and all 720 probes. All isolated-error groups and uncertain missing/extra groups passed. Uncertain A/P mass preservation still failed in 68/120 probes. No SwinUNETR training started. Preserve this attempt, including its source, separately. The initial 1,000 updates and completed 300 updates are reused, never repeated.

A local diagnostic using one training anatomy tested square-root probability geometry, and its average over all six class permutations, with the existing encoder. Only 9/12 and 11/12 uncertain-swap probes passed the mass threshold, respectively. These transformations are not adopted: the production learned feature distance remains unchanged. A reference CE gradient passes the two single-voxel uncertain-swap tests (foreground-change/correct-label-gain ratios 0.255 and 0.398), demonstrating that the threshold is attainable. Matching an approximate logit-gradient direction was insufficient for the learned encoder.

The next encoder-only refinement uses 600 fixed additional updates, seed 29, AdamW 3e-4, frozen decoder heads and the same corruption distribution as the preceding refinement. Differentiate through a real unmasked logit step (maximum absolute change 0.25) of the learned feature loss. Train the encoder to align the resulting probability change with the correct probability repair, plus weight 2 on an excess mass-change penalty and weight 0.1 on the retained representation objectives. For swaps, the penalty activates above an absolute whole-FOV foreground-change/correct-label-gain ratio of 0.1; for missing/extra patches it penalizes an incorrect mass direction. The scale denominator is detached. These objectives train the encoder; they are not additional SwinUNETR losses. Preserve all earlier checkpoints and save resumable state every 50 updates.

Before any segmentation training, repeat the complete 720-probe audit with fresh corruption seed 20261009 and the same six groups, radii, confidence values and acceptance thresholds. The training margin of 0.1 is stricter than the unchanged audit threshold of 0.5. This remains a development audit on the same ten training anatomies; it is not independent anatomical validation. The frozen encoder's learned feature distance alone must pass. No other loss, projection or gradient manipulation is applied during the audit or SwinUNETR training.
