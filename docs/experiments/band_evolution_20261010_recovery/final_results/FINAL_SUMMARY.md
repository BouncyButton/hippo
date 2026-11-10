# Final results: 189-arm band-evolution screen

Job **692240 completed all 189/189 arms**, exit code 0, on **11 October 2026 at 01:18 Europe/Rome**. Slurm elapsed time was **14:16:07**. No stages failed and no experiments remain queued in this campaign.

Ordinary image-only SwinUNETR, no auxiliary branch, 5% training data (10 cases), fold 0, seed 0, and 52 validation cases. All arms share the verified five-epoch Dice-only prefix, optimizer/scheduler settings and data order; training stops by the frozen early-stopping rule or the 75-epoch cap. Results use each arm's selected validation checkpoint and raw three-class argmax predictions. A/P Dice is the mean of anterior and posterior Dice, averaged over validation cases.

## Main results

| Variant | A/P Dice % | Foreground Dice % | Extra voxels | Missed voxels |
|---|---:|---:|---:|---:|
| Dice + canonical bands | 76.7272 | 79.3186 | 64,806 | 16,923 |
| Previous best: independent 1+3 mm targets | 78.1316 | 80.7367 | 56,037 | 18,157 |
| **3 mm regional band, worst-quarter emphasis + independent targets** | **78.9369** | **81.6338** | **50,548** | **19,087** |
| Stronger independent targets, 60/40 inside/outside | 78.7425 | 81.3906 | 49,779 | 20,365 |
| Stronger independent targets, 50/50 inside/outside | 78.5507 | 81.2270 | 52,729 | 18,839 |
| 3 mm regional band with 1 mm exponential decay + targets | 78.4775 | 81.1750 | 55,198 | 17,280 |
| 1 mm cross-entropy regional band + targets | 78.3556 | 80.9774 | 53,551 | 19,020 |

The winner is `band_focus_top25_in50`. Compared with canonical Dice + bands it gains **2.2096 percentage points**, winning on **52/52** validation cases. Compared with the previous independent-target winner it gains **0.8052 points**, winning on **48/52** cases, with median gain **0.8343 points**.

Most of the improvement over the previous winner is posterior: anterior Dice rises from 79.3479% to 79.4427%, and posterior Dice from 76.9154% to 78.4311%.

## What the winning loss does

The original bands term is replaced by a regional squared probability-error term within 3 mm of the annotated outer boundary, using the union of anterior and posterior tissue as foreground. Native-field masking excludes padding. On each side independently:

`side_loss = 0.5 * mean(all squared errors) + 0.5 * mean(largest 25% of squared errors)`

The band loss then averages the inside and outside side losses with equal 50/50 weight. The hardest quarter is selected from the current predictions on each forward pass; the remaining voxels still receive supervision through the full mean. This is voxel-error emphasis, not connected-component or coherent-region selection.

Dice and the independent inside/outside targets at 1 and 3 mm remain active. Pair scales are equally weighted, as are the two sides. Pair distance means the total inside-to-outside voxel-centre separation, whereas regional-band width is distance from voxel centres to the annotated boundary-face midpoint set.

After the scheduled constraint ramp, the calibrated loss is:

`Dice + 0.0157389242217977 * regional_band + 0.00851541901513619 * independent_pairs`

The matching plain 3 mm regional band with independent targets scores **77.9481%**, versus **78.9369%** with worst-quarter emphasis: **+0.9888 points**. Each formulation was calibrated under the same gradient-budget procedure, so this compares complete calibrated objectives rather than holding their raw numeric coefficients fixed.

## Other findings and limitations

- Stronger pair supervision helped the equal-distance profiles: the target median gradient ratio increased from 0.10 to 0.15, retaining the same 0.5 p95 cap. With 60/40 inside/outside weighting, Dice increased from 77.8218% at standard strength to 78.7425%. With 50/50 weighting it increased from 78.1316% to 78.5507%. This was not uniformly beneficial: the stronger far-weighted profile scored only 77.7845%.
- Chamfer combinations scored **77.5404%** with 50/50 side weighting and **77.7817%** with 60/40 weighting. Neither beat the 78.1316% independent-target reference. The 60/40 Chamfer arm did improve A/P interface Chamfer from 1.3685 to 1.3412 mm, despite its lower overall Dice.
- Local foreground-volume matching did not produce a meaningful overall gain: its best arm scored 78.1530%.
- The winning band objective removes **5,489 extra voxels** but adds **930 missed voxels** and **75 A/P swaps** relative to the previous winner, summed across the validation cohort. Mean outer-surface distance improves from 0.8393 to 0.8186 mm and HD95 from 1.9387 to 1.8975 mm.
- The winner does **not** improve A/P cut placement relative to the previous winner: best-plane error is 1.3077 versus 1.2308 mm, and A/P interface Chamfer is 1.3860 versus 1.3685 mm. Its benefit is primarily foreground segmentation and posterior overlap.
- This is an exploratory screen of 189 variants on one reused validation fold and one seed. The ranking and effect sizes need confirmation on additional seeds/folds or an untouched evaluation set. These results do not establish that the loss reproduces the auxiliary branch's mechanism.

## Verification and retained records

All 189 run identities and completion records were verified, including **953 artifact hashes** and **181 checkpoint-retirement receipts**. All packaged source hashes passed. The three controls exactly reproduced the previous 64-arm campaign's A/P and foreground Dice and aggregate error counts.

Aggregate exports were checked against their remote SHA-256 hashes. This local export includes no case identifiers, per-case medical records, images, model tensors or prediction arrays.

- `results.csv`: all 189 aggregate result rows.
- `RESULTS.md`: all arms versus Dice + bands in queue order.
- `aggregate_summary.json`: ranked results, aggregate paired comparisons, top configurations/coefficients, verification and retention metadata.
- `status.json`: final completion status and stage timings.
- `checkpoint_retention.json`: three controls plus the five best models retained on the cluster.
- `local_verification.json`: verified aggregate-file hashes and remote audit summary.

Remote metrics, configuration, logs and predictions remain preserved. Redundant best checkpoints were automatically retired by the declared retention policy. The completed job has no active resume checkpoint. BeeGFS quota at the final status check was **90.62/93.13 GiB**, leaving approximately **2.51 GiB** free.

Protocol identity: `8132b9215d1bbb6ad0ead278fcb3cf40c6e9fd05a2db0cdc7c54502eb08099d5`.
