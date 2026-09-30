# Signed gradients on real E fitting batches — 23 September 2026

**Finding:** E already receives a preservation gradient on erased thin anatomy. On these batches, the total objective increases presence in 430/448 erased thin-ray observations (95.98%) and increases the raw foreground-logit direction at all 734 true voxels within those rays. A missing output-level preservation signal is not supported. Shared-parameter interference and whether training can realize that signal remain separate questions.

## Scope and verification

- Saved E checkpoints from folds 1–4; four fitting batches of two cases per fold, sampler seed 1000. This gives 32 case exposures, 26 distinct cases, 981 retained thin-ray observations, 448 erased thin-ray observations, 63 already-missed thin-ray observations, and 2,273 hard-empty-ray observations. Folds overlap in fitting data; these are not independent replications.
- Thin means true span 1–2 without border truncation. Erased means raw foreground present but final foreground absent. Hard empty means truly empty and raw foreground present or head presence at least 0.5.
- Training-mode backbone forward with AMP; head and losses in float32. The diagnostic output matches the reference renderer within 1e-5. Loss decomposition is asserted against the existing implementation. Eleven local synthetic/integration tests pass.
- All original coefficients and the normalized rendered/raw DiceCE mixture are retained. The raw-foreground direction raises anterior and posterior logits equally, preserving their odds; it includes direct and indirect head routes. Decoder-feature gradients are separate. This is not a full backbone-parameter update.
- Results and per-batch case IDs: `experiments/presence_decisive_20260923/results_signed_gradients/`. Source protocol: `experiments/presence_decisive_20260923/SIGNED_GRADIENT_PROTOCOL.md`.

## Loss-separated signed derivatives

Values below are mean derivatives multiplied by 10⁶. **Negative means ordinary gradient descent pushes the measured logit UP.** Absolute scale depends on loss reduction; it does not by itself establish a weak or strong effective optimizer update.

| Weighted loss | Erased thin: presence | Erased thin: true raw foreground | Retained thin: presence | Hard empty: presence |
|---|---:|---:|---:|---:|
| seg_rendered | -2.4957 | -6.3971 | +0.4328 | +3.1708 |
| seg_raw_aux | +0.0000 | -1.7962 | +0.0000 | +0.0000 |
| curve_presence | -1.8889 | -1.3999 | -0.6804 | +0.1057 |
| curve_conditional_edges | +0.0000 | -0.2830 | +0.0000 | +0.0000 |
| curve_positions | +0.0562 | -0.0512 | -0.0092 | -0.0025 |
| responsibility | +0.0000 | -0.3346 | +0.0000 | +0.0000 |
| ordering | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| total_existing_E | -4.3284 | -10.2621 | -0.2568 | +3.2740 |
| thin_focal_batch_calibrated_0p3 | -8.4155 | -17.9780 | -0.8704 | +0.0007 |
| total_plus_batch_calibrated_focal | -12.7439 | -28.2401 | -1.1273 | +3.2746 |

The optional focal row uses a **new coefficient recalibrated to a 0.3 raw-logit gradient-RMS ratio separately on each diagnostic batch** (range 0.00161635–0.00475415). It is not the coefficient or checkpoint from the historical focal screen. Those rows are a hypothetical local intervention.

## Direction and fold consistency

| Stratum / output | Fraction pushed UP by E | Interpretation |
|---|---:|---|
| erased_thin / presence_logit | 95.98% | Preservation is requested on almost all erased rays. |
| erased_thin / foreground_delta_true_voxels | 100.00% | All sampled erased true voxels receive a positive foreground instruction. |
| already_missed_thin / presence_logit | 100.00% | The signal also exists when raw output already misses the ray. |
| retained_thin / presence_logit | 30.17% | Most retained rays are instead locally pushed toward lower presence. |
| retained_thin / foreground_delta_true_voxels | 56.66% | Some true raw-voxel gradients oppose preservation through coupled paths. |
| hard_empty / presence_logit | 0.00% | The total objective consistently suppresses hard-empty presence. |

For erased-ray presence, the UP fractions are 95.79%, 95.70%, 96.90%, and 95.42% in folds 1–4. For erased true-voxel logits they are 100% in every fold.

Lowering presence on a currently retained ray is not automatically an error: it can reduce excessive foreground within that ray. However, it demonstrates that presence serves both existence and voxel correction, so its segmentation gradient need not behave as an existence-classifier gradient.

## Renderer versus fit routes

On erased rays, rendered segmentation contributes −2.4781×10⁻⁶ through presence’s direct renderer route and −0.0176×10⁻⁶ through its fit route. Including the position loss, the fit route contributes +0.0385×10⁻⁶. Thus the fitted-position route mildly opposes rescue on average, but does not dominate the existing output-level signal.

## Why simply increasing focal weight is not a clean fix

For background voxels within erased true-thin rays, E’s total raw-foreground derivative averages +0.00682×10⁻⁶; 40.15% are pushed upward. Adding the diagnostic focal term changes the mean to −0.45292×10⁻⁶ and pushes **95.41%** upward. The focal term alone pushes every sampled background voxel in those rays upward through the complete raw-to-head-to-rendered path, despite having a direct negative-voxel term. These are local derivatives, not observed new false positives after training.

Focal supervision on true thin rays has exactly zero direct presence-renderer gradient on hard-empty rays. Small indirect effects remain through spatial fitting and shared representations. Existing E losses still suppress hard-empty rays, but focal alone does not directly teach thin-versus-empty discrimination. This is a plausible explanation for a rescue/FP tradeoff; it does not establish why historical fold 4 failed its guard.

## Shared representation: what the output signs do not settle

Direct presence supervision has negative cosine with the existing total head-stem parameter gradient in 9/16 batches (mean cosine −0.0892). The calibrated focal head-stem gradient has mean L2 norm 0.00518 versus 0.00440 for the existing total, and opposes that total in 9/16 batches. Thus a nominal 0.3 calibration at raw logits does not imply a small change to head learning. These are whole-batch parameter gradients, not isolated thin-ray updates.

The available output-level signal cannot establish whether AdamW, shared features, competing examples, saturation, or finite training time prevents its realization. The exploratory exact first-order shared-head parameter-response diagnostic below tests this distinction on the same batches.

## Current inference

1. **Missing preservation supervision:** contradicted at the sampled saved-E outputs.
2. **Another loss reverses most erased-thin output gradients:** contradicted here; 95.98% of presence and 100% of true-voxel directions remain favorable.
3. **Shared-parameter interference:** directly demonstrated for the tested plain-SGD head direction on these batches; this is not yet an AdamW or full-backbone training result.
4. **Presence doubles as renderer correction strength:** supported by opposite incentives on retained and erased rays and by background-gradient spillover.
5. **Bigger head solves it:** unresolved by these derivatives. The separate frozen-representation width/pooling study tests that claim under fixed false-positive guards.

No outer labels were used to fit or select anything in this diagnostic. No optimizer step or checkpoint modification was performed. These are local fitting-batch findings, not external-generalization or anatomical-validity claims.

## Exploratory shared-head parameter response

The second pass computes `−∇θ(mean output) · ∇θ(loss)` across **all function-head parameters**, holding backbone outputs/features fixed. This is the exact first-order output response per unit plain-SGD learning rate. Positive means an increase. No parameter update was applied. Its original signed-gradient summaries match the first pass exactly. The initially interrupted allocation produced no batch results; the completed pass used allocation 666669.

| Loss direction | Erased thin: mean presence response | Batches decreasing erased presence | Hard empty: mean presence response | Batches increasing empty presence |
|---|---:|---:|---:|---:|
| seg_rendered | -1.091879 | 12/16 | -1.097906 | 4/16 |
| curve_presence | +1.012989 | 0/16 | +1.014084 | 16/16 |
| curve_conditional_edges | -0.024115 | 16/16 | -0.024977 | 0/16 |
| curve_positions | -0.044343 | 16/16 | -0.044873 | 0/16 |
| total_existing_E | -0.147436 | 8/16 | -0.153758 | 8/16 |
| thin_focal_batch_calibrated_0p3 | +1.949513 | 0/16 | +1.951551 | 16/16 |
| total_plus_batch_calibrated_focal | +1.802137 | 2/16 | +1.797852 | 14/16 |

This resolves an important ambiguity: a favorable derivative with respect to each erased ray's output does **not** imply a favorable change through shared parameters. E's total head-gradient direction lowers erased-thin mean presence in 8/16 batches, while the rendered segmentation term alone does so in 12/16. The corresponding rendered foreground margin at true erased-thin voxels also falls in 8/16 under the total direction.

The diagnostic focal direction increases erased-thin mean presence in 16/16 batches, but also increases hard-empty mean presence in 16/16, with almost identical mean responses (+1.949513 and +1.951551). Its direction increases mean rendered foreground margin on hard-empty voxels in 16/16 batches. Adding it to E increases that empty-voxel margin in 13/16 batches. This is measured local coupling, not evidence that every such change crosses an argmax boundary or becomes a false positive.

Even direct presence supervision increases mean hard-empty presence under the shared-head direction in 16/16 batches despite its per-empty-ray derivative pointing downward. Other rays' demands propagate through shared parameters. The next discriminating comparison is therefore a mechanism that improves thin-versus-empty separation, or decouples existence from renderer strength, against a matched scalar-bias/calibration control. Increasing a focal coefficient is not a selective rescue mechanism here.

Limits: this uses a Euclidean SGD direction, without AdamW moments, adaptive scaling, clipping, weight decay, or backbone updates. It measures stratum means; some individual rays can move differently. It is a local derivative at selected saved checkpoints, not evidence of the whole training trajectory. Three new full-training seeds and a matched continuation experiment would be needed before attributing historical focal failure to this mechanism.

Complete second-pass artifacts: `experiments/presence_decisive_20260923/results_signed_gradients_v2_completed/`.

## Check using the actual historical 0.3-screen coefficients

The saved per-fold focal coefficients were retrieved with SHA-256 provenance: 0.0041105424, 0.0038782548, 0.0028469007, and 0.0036222434. Applying these constants to the saved unit-focal parameter responses gives the same qualitative result: focal alone increases erased-thin mean presence in 16/16 batches (mean response +2.038763), and hard-empty mean presence in 16/16 (+2.037817). E plus this focal direction increases both stratum means in 14/16 batches and increases mean foreground margin on hard-empty voxels in 13/16.

This rescaling is valid because directional responses are linear in the loss coefficient. It still evaluates saved E on the diagnostic batches, not the subsequent focal checkpoints or historical AdamW state. Combined per-voxel sign fractions cannot be reconstructed from aggregate moments and are not claimed for these coefficients. Reproduction script: `rescale_historical_focal.py`; result: `results_signed_gradients_v2_completed/historical_focal_responses.json`.
