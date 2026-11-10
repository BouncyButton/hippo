# Tissue profiles and auxiliary-view controls: final results

Saved 8 October 2026. Slurm job **688313** completed successfully, exit 0, on 8 October at **01:53:06 Europe/Rome**, after **7 h 14 min 03 s**. All **36 new runs** and their preflight checks completed; six previously completed Dice controls give **42 comparison cells**. No new experiments were submitted for this report.

## Main findings

- **A separate, focused processing/fusion path helps in this experiment. Unwrapping has not been shown to be necessary.** With Dice alone, the native-view MRI branch achieved **81.9904% versus 74.1667%** at 5% data (+7.8236 percentage points), and **88.3102% versus 87.8514%** at full data (+0.4588 points). It sees the same MRI samples in their original voxel orientation, within the frozen guide-defined ROI; it is a small CNN, not a second SwinUNETR.
- **The alternative views do not clearly beat this native-view control under Dice alone.** MRI-only plane/unwrapped branches score 81.9168%/81.8895% at 5%, and 88.2627%/88.2786% at full data. Their paired case-bootstrap intervals versus the native-view branch include zero. This is absence of clear evidence of superiority, not a formal equivalence finding.
- **Explicit position/depth channels are not necessary for the large branch gain.** Directly adding anatomical channels to SwinUNETR improves Dice by 1.0800 points at 5% and 0.1121 at full data, much less than the native-view auxiliary. However, the MRI-only branches still use guide-derived ROI support and chart membership for pooling, mapping and fusion; anatomy has not been removed.
- **The profile constraints have architecture-dependent effects.** On ordinary SwinUNETR they improve 5% Dice to 75.5069% (pooled) and 75.5302% (multiscale), but only 87.8748% and 87.9159% with full data. They do not consistently improve the stronger branch models. On the native-view branch, pooled profiles reduce 5% Dice by 0.5343 points, while multiscale profiles add only 0.0625 points.
- **A higher Dice can conceal additional missing tissue.** At 5%, the native-view branch's Dice gain accompanies 48,674 fewer extra voxels, but 8,175 more missed voxels, aggregated across 52 validation cases. At full data, the same Dice-only comparison has 2,429 fewer missed voxels and 1,693 more extra voxels. These are net count differences, not exact recovered-voxel identities.

## Complete Dice matrix

All values are best-checkpoint validation anterior/posterior macro Dice, in percent. Pooled and multiscale columns mean Dice plus that profile term. At 5% the auxiliary has context fusion; at full data it additionally has attention. Cross-regime comparisons therefore also change branch architecture.

| Architecture | 5% Dice | 5% pooled | 5% multiscale | Full Dice | Full pooled | Full multiscale |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary SwinUNETR | 74.1667 | 75.5069 | 75.5302 | 87.8514 | 87.8748 | 87.9159 |
| Direct MRI + position/depth/ROI inputs | 75.2468 | 75.9163 | 75.9834 | 87.9635 | 88.0055 | 87.8946 |
| Plane branch + position channels | 81.9997 | 81.8476 | 81.7432 | 88.2854 | 88.2400 | 88.2020 |
| Unwrapped branch + position channels | 81.6822 | 81.9099 | 81.8726 | 88.2273 | 88.3218 | 88.3044 |
| Plane branch, MRI-only encoder | 81.9168 | 81.8508 | 81.7517 | 88.2627 | 88.1769 | 88.1705 |
| Unwrapped branch, MRI-only encoder | 81.8895 | 81.9428 | 82.0792 | 88.2786 | 88.3549 | 88.3171 |
| Native-view branch, MRI-only encoder | 81.9904 | 81.4561 | 82.0528 | 88.3102 | 88.3224 | 88.3539 |

The numerically best 5% cell is MRI-only unwrapped + multiscale, **82.0792%**, just **0.0263 points** above native-view + multiscale (**82.0528%**). At full data the best cell is MRI-only unwrapped + pooled (**88.3549%**), essentially tied numerically with native-view + multiscale (**88.3539%**); these two use different losses and are not an isolated view comparison. Selecting the best of many validation cells creates selection bias.

## Matched Dice-only evidence

Intervals below are 95% percentile intervals from 10,000 paired resamples of the 52 validation cases. They condition on the trained, validation-selected checkpoints; they do not measure training-seed uncertainty or adjust for multiple comparisons.

| Comparison | 5% delta in percentage points [95% interval] | Full delta in percentage points [95% interval] |
|---|---:|---:|
| Native-view MRI branch minus ordinary SwinUNETR | +7.8236 [+7.1653, +8.4748] | +0.4588 [+0.0889, +0.8345] |
| MRI-only plane minus native-view branch | -0.0736 [-0.3088, +0.1566] | -0.0475 [-0.3992, +0.3047] |
| MRI-only unwrapped minus native-view branch | -0.1009 [-0.3891, +0.1920] | -0.0316 [-0.3257, +0.2719] |
| Direct anatomical inputs minus ordinary SwinUNETR | +1.0800 [+0.8091, +1.3282] | +0.1121 [-0.1512, +0.3766] |

There is a loss interaction: at 5% with pooled profiles, MRI-only unwrapped beats the corresponding native-view branch by **0.4868 points** (case-bootstrap interval **[0.2503, 0.7155]**). This partly reflects the native-view branch's deterioration under pooled profiles; it does not establish that unwrapping is generally superior. All matched comparisons are retained in `matched_comparisons.json` and the original case differences in `evidence/comparison.json`.

## What could become a learnable constraint?

The current evidence does **not** prove that the auxiliary MRI encoder is unnecessary. A previous, different checkpoint (5% unwrapped plus depth loss) retained 81.6623% Dice versus 81.8925% normally when mapped auxiliary features were zeroed at evaluation. This kept the ROI mask and learned fusion and changed inputs only after training. The corresponding full-data audit dropped from 88.3790% to 85.2813%, so the low-data result cannot be generalized to every setting. Neither audit replaces training a matched model without the auxiliary encoder.

The direct next hypothesis is a small **ROI-conditioned residual correction module**: reuse native SwinUNETR decoder features F and guide ROI mask m, learn a correction R(F,m), then classify F + R(F,m). It can use the MRI evidence already represented in F without reading the scan through another encoder. Train the whole network jointly with Dice first. This is a learned spatial prior/refinement architecture; it does not guarantee anatomical validity and is not automatically a new constraint loss.

To identify its cause, compare ordinary SwinUNETR, native-feature refinement without the ROI mask, the same refinement with the ROI mask, and the existing native-view MRI branch. Match training settings and initialization and repeat both fractions. A gain from refinement alone would support added decoder processing; a gain specifically from the mask would support spatial conditioning. These are proposals only, not submitted runs.

A training-only learned constraint is a further, separate option: train a compact anatomy encoder on **training labels only** to reconstruct clean masks from masks corrupted by additions, omissions and anterior/posterior swaps. Once this representation is learned, freeze it and penalize the difference between its representations of the model's soft probabilities and the target labels, alongside Dice. Frozen parameters still permit gradients back to the segmentation probabilities. A free penalty trained only to minimize its own value could collapse to zero, so it needs independent reconstruction/denoising supervision and freezing, or a carefully designed alternative training objective. At inference such a loss encoder can be discarded.

That learned representation could retain local arrangements that pooled profiles erase, but it can also miss small errors or overfit the ten training cases in the 5% condition. It must use the same allowed training subset; learning its prior from the full dataset would invalidate a 5% comparison. Validation labels remain evaluation targets and do not train the prior. Use actual MRI evidence and inspect FN/FP, A/P swaps and surfaces: a prior that merely shrinks predictions would repeat the existing low-data trade-off. No hard clipping to the guide ROI is proposed, because the guide itself can omit tissue.

Learned anatomy representations used as supervision have precedent in [Oktay et al., Anatomically Constrained Neural Networks](https://arxiv.org/abs/1705.08302). [Larrazabal et al., Post-DAE](https://arxiv.org/abs/2006.13791) provides a related denoising shape-prior precedent, but applies a post-processing reconstruction; it is not evidence that a training-only penalty or our proposed refinement will improve hippocampus segmentation.

## Protocol and limitations

- MSD Task04 hippocampus / Dataset101_MSD, fold 0, seed 0. Training: 10 of 208 cases (labelled 5%) or all 208; validation: 52 shared cases.
- Entire SwinUNETR and any auxiliary/fusion components train jointly from scratch. Feature size 24, padded native 64³ MRI, batch size 1, no augmentation or pretraining, CUDA bf16. Inputs are hippocampus crops, not whole brains.
- AdamW, learning rate 1e-4, weight decay 1e-5; StepLR every 20 epochs by 0.5. Maximum 75 epochs; early stopping at minimum 25 epochs, patience 8, minimum improvement 0.0005. The highest validation macro Dice checkpoint is reported.
- Dice includes background. No depth balancing, SDF, focal weighting, entropy gating, topology or component-count penalty is used in this campaign. Profile coefficients use training-only gradient calibration after five Dice epochs and ramp during epochs 6–10.
- Guide-derived frames and ROI support are frozen. Training guides were out-of-fold. MRI-only auxiliary branches still depend on the guide ROI plus 8 mm and fixed chart/context groups. These guide models add cost beyond the final segmentor.
- The native-view MRI auxiliary adds 20,984 parameters at 5% (about 0.134% of the 15,703,029-parameter baseline). Parameter count alone therefore does not establish the cause; fusion placement, convolutional processing, ROI conditioning and normalization/context are possible explanations.
- Alternative-view controls match trainable weights and parameter counts, but resampling, grid shape, effective physical convolution scale, padding and support layout differ. The comparison is between processing pipelines, not an isolated mathematical coordinate change.
- This is one fold and one seed, with fixed guides and a 75-epoch schedule. The earlier three-seed plane/unwrapped study did not replicate this new native-view control. Case bootstrap intervals are not seed replication. The 5% baseline reached the maximum 75 epochs while branch models often stopped earlier; conclusions apply to this schedule and do not establish converged-model superiority.
- All comparisons use validation data already used for checkpoint/model selection. Generalization, anatomical guarantees and clinical usefulness are not established. Aggregate error differences cannot identify exact corrected voxels. Original runtime counters differ for reused controls, so speed claims need separate benchmarking.

## Full metric appendix

FN = missed foreground; FP = extra foreground; A/P = anterior/posterior label swaps. Error counts are pooled over 52 validation cases. MSD and HD95 are mean case-level surface metrics in millimetres. Epochs are best/last. Full precision, class Dice, foreground Dice, parameter counts and runtimes are preserved in `all_results.csv` and `all_results.json`.

| Data | Arm | Loss | Dice % | FN | FP | A/P | MSD mm | HD95 mm | Epochs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5pct | native | baseline | 74.1667 | 15,386 | 80,273 | 5,794 | 1.0295 | 2.3150 | 73/75 |
| 5pct | native | profile | 75.5069 | 17,014 | 70,617 | 5,410 | 0.9336 | 2.1265 | 73/75 |
| 5pct | native | multiscale | 75.5302 | 16,791 | 70,768 | 5,415 | 0.9341 | 2.1325 | 75/75 |
| 5pct | informed | baseline | 75.2468 | 19,953 | 67,161 | 5,468 | 0.9554 | 2.1407 | 74/75 |
| 5pct | informed | profile | 75.9163 | 21,004 | 61,711 | 5,406 | 0.9116 | 2.0923 | 74/75 |
| 5pct | informed | multiscale | 75.9834 | 21,345 | 60,862 | 5,437 | 0.9034 | 2.0939 | 73/75 |
| 5pct | plane_position | baseline | 81.9997 | 23,486 | 31,554 | 4,447 | 0.6561 | 1.5758 | 39/47 |
| 5pct | plane_position | profile | 81.8476 | 24,918 | 30,375 | 4,393 | 0.6638 | 1.5635 | 39/47 |
| 5pct | plane_position | multiscale | 81.7432 | 22,817 | 33,339 | 4,559 | 0.6702 | 1.6231 | 39/47 |
| 5pct | unfold_position | baseline | 81.6822 | 26,649 | 28,237 | 4,597 | 0.6628 | 1.6059 | 45/52 |
| 5pct | unfold_position | profile | 81.9099 | 26,446 | 27,674 | 4,575 | 0.6523 | 1.5929 | 42/50 |
| 5pct | unfold_position | multiscale | 81.8726 | 25,238 | 29,418 | 4,611 | 0.6579 | 1.6060 | 42/50 |
| 5pct | plane_mri_context | baseline | 81.9168 | 24,376 | 31,095 | 4,313 | 0.6642 | 1.5779 | 45/51 |
| 5pct | plane_mri_context | profile | 81.8508 | 24,350 | 31,240 | 4,392 | 0.6652 | 1.5868 | 40/48 |
| 5pct | plane_mri_context | multiscale | 81.7517 | 24,969 | 31,336 | 4,149 | 0.6734 | 1.5919 | 35/41 |
| 5pct | unfold_mri_context | baseline | 81.8895 | 23,584 | 31,424 | 4,659 | 0.6578 | 1.5629 | 38/46 |
| 5pct | unfold_mri_context | profile | 81.9428 | 24,627 | 30,230 | 4,464 | 0.6583 | 1.5852 | 41/49 |
| 5pct | unfold_mri_context | multiscale | 82.0792 | 26,569 | 27,563 | 4,208 | 0.6531 | 1.5782 | 42/50 |
| 5pct | native_mri_context | baseline | 81.9904 | 23,561 | 31,599 | 4,435 | 0.6589 | 1.5727 | 40/48 |
| 5pct | native_mri_context | profile | 81.4561 | 19,363 | 39,506 | 4,598 | 0.6938 | 1.6550 | 27/35 |
| 5pct | native_mri_context | multiscale | 82.0528 | 25,971 | 28,277 | 4,321 | 0.6563 | 1.5706 | 42/50 |
| full | native | baseline | 87.8514 | 17,533 | 17,177 | 3,635 | 0.4253 | 1.1415 | 23/25 |
| full | native | profile | 87.8748 | 17,926 | 16,474 | 3,644 | 0.4225 | 1.1415 | 21/25 |
| full | native | multiscale | 87.9159 | 17,638 | 16,792 | 3,644 | 0.4196 | 1.1176 | 14/25 |
| full | informed | baseline | 87.9635 | 16,541 | 17,990 | 3,623 | 0.4216 | 1.1237 | 16/25 |
| full | informed | profile | 88.0055 | 17,639 | 17,042 | 3,379 | 0.4234 | 1.1256 | 23/31 |
| full | informed | multiscale | 87.8946 | 17,627 | 16,853 | 3,648 | 0.4210 | 1.1415 | 23/25 |
| full | plane_attention | baseline | 88.2854 | 15,234 | 18,643 | 3,485 | 0.4149 | 1.1176 | 11/25 |
| full | plane_attention | profile | 88.2400 | 14,512 | 19,327 | 3,668 | 0.4146 | 1.1017 | 9/25 |
| full | plane_attention | multiscale | 88.2020 | 15,138 | 19,073 | 3,521 | 0.4190 | 1.1336 | 12/25 |
| full | unfold_attention | baseline | 88.2273 | 17,311 | 16,131 | 3,563 | 0.4117 | 1.1176 | 16/25 |
| full | unfold_attention | profile | 88.3218 | 16,361 | 17,421 | 3,364 | 0.4138 | 1.1261 | 25/33 |
| full | unfold_attention | multiscale | 88.3044 | 17,743 | 15,734 | 3,398 | 0.4109 | 1.1069 | 15/25 |
| full | plane_mri_attention | baseline | 88.2627 | 16,368 | 17,054 | 3,619 | 0.4116 | 1.1317 | 9/25 |
| full | plane_mri_attention | profile | 88.1769 | 17,962 | 15,427 | 3,627 | 0.4100 | 1.1017 | 15/25 |
| full | plane_mri_attention | multiscale | 88.1705 | 16,686 | 17,695 | 3,356 | 0.4219 | 1.1434 | 11/25 |
| full | unfold_mri_attention | baseline | 88.2786 | 16,064 | 17,377 | 3,568 | 0.4097 | 1.1237 | 11/25 |
| full | unfold_mri_attention | profile | 88.3549 | 16,049 | 17,389 | 3,468 | 0.4127 | 1.1377 | 13/25 |
| full | unfold_mri_attention | multiscale | 88.3171 | 17,480 | 15,783 | 3,427 | 0.4081 | 1.1237 | 21/25 |
| full | native_mri_attention | baseline | 88.3102 | 15,104 | 18,870 | 3,413 | 0.4179 | 1.1265 | 11/25 |
| full | native_mri_attention | profile | 88.3224 | 16,990 | 16,177 | 3,519 | 0.4081 | 1.1237 | 21/29 |
| full | native_mri_attention | multiscale | 88.3539 | 15,909 | 17,569 | 3,434 | 0.4114 | 1.1158 | 23/29 |

## Preservation and cleanup

All 180 recorded artifact hashes (five artifacts for each of the 36 new runs, including checkpoints and prediction arrays) were checked before cleanup. The local evidence archive retains the small per-case results, training histories, configurations, completion records, diagnostics, logs, source snapshots and deployment/protocol metadata. It excludes model weights and prediction/geometry arrays, which the user authorized deleting. Reused controls and their per-case comparison vectors are included in the comparison evidence; their original records also remain in the previously saved unfolding-study archive.

`evidence_manifest.json` lists every archived file and SHA-256. `collection_receipt.json`, `verification.json`, `preservation_receipt.json`, `cleanup_plan.json` and `cleanup_receipt.json` record verification and cleanup. The exact deletion scope is generated outputs inside `/mnt/beegfsstudents/home/3160552/tissue_profiles_fold0_seed0_20261007_01` only. Source, deployment records, the shared atlas target, raw data, unrelated campaigns and the local report/evidence are preserved. Model checkpoints deleted from this job cannot be used for further inference audits without retraining or another independent copy.

Cleanup completed successfully on 8 October 2026 after local preservation and a fresh empty scheduler check. Deleted **420 files**, including **36 best checkpoints**, prediction arrays, generated diagnostic geometry and logs: **2,594,293,533 logical bytes (2.4161 GiB)**; allocated bytes in the manifest: **2,594,388,480**. Every selected path was confirmed absent and the shared atlas target remained accessible.

The refreshed BeeGFS account quota was **77.29 / 93.13 GiB used**, approximately **15.84 GiB free** (previous reported usage: 82.12 GiB). Quota accounting changed by 4.83 GiB; the deletion manifest's directly measured file volume is 2.4161 GiB, and those figures are reported separately. Exact records are in `cleanup_receipt.json` and `final_quota.json`.

The preservation receipt hashes the report version reviewed before deletion. `final_manifest.json` hashes this final report, which adds the cleanup outcome.
