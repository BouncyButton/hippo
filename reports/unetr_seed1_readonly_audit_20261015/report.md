# UNETR seed-1 read-only audit

The checked archives show real poor learning for seed 1, with severe foreground overprediction. No discrepancy was found in initialization, cohort selection, scientific settings, optimization accounting, learning-rate values or saved metric calculations. This is not a proof that every possible implementation bug is absent.

All work used downloaded completed artifacts and local CPU analysis. No running Colab process was stopped, changed or asked to perform additional training.

## Final baseline Dice (%)

| Fold | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| 0 | 49.87 | 15.64 | 57.60 |
| 1 | 52.17 | 16.16 | 56.14 |
| 2 | 50.36 | 15.04 | 55.93 |

| Seed | Mean training Dice | Mean validation Dice |
|---|---:|---:|
| 0 | 59.57 | 50.80 |
| 1 | 17.97 | 15.61 |
| 2 | 64.73 | 56.56 |

## What was checked

- Downloaded metadata hashes and archive acknowledgment/source binding.
- All nine baseline runs and two fold-0 bands runs; 245 updates, none discarded.
- Exact seed-specific initialization hashes, identical across folds and matched arms.
- Exact frozen cohort IDs and per-case label voxel counts across seeds.
- Identical scientific source, data, precision, loss, learning-rate trajectory and FLOP schedules.
- All 20 curve milestones finite; pre-constraint baseline/bands prefix matches exactly.
- Bands calibration uses exactly the ten training cases, achieves10% median and respects50% p95 ceiling.
- Independently recomputed hard Dice and foreground FN/FP from every saved confusion matrix.
- Verified cleanup receipts; completed checkpoints already retired, no model replay performed.

150 scientific/diagnostic files matched their archive manifest hashes. The 22 small performance-invocation logs and empty lock files were not downloaded; this is a targeted scientific audit, not a fresh verification of every archived file. Checkpoints had already been deleted after the original verified archival/cleanup, so trained-model inference could not be independently rerun.

## Failure mode

| Fold, seed | Predicted foreground / true foreground | Foreground false positives | Foreground false negatives |
|---|---:|---:|---:|
| 0, 0 | 3.08× | 371,689 | 9,040 |
| 0, 1 | 11.24× | 1,785,552 | 453 |
| 0, 2 | 2.31× | 234,357 | 6,289 |
| 1, 0 | 2.84× | 322,075 | 4,438 |
| 1, 1 | 10.77× | 1,687,911 | 302 |
| 1, 2 | 2.34× | 235,102 | 3,572 |
| 2, 0 | 2.95× | 335,398 | 7,817 |
| 2, 1 | 11.72× | 1,798,442 | 568 |
| 2, 2 | 2.34× | 230,992 | 6,136 |

The curves are smooth and finite; the poor result is present before constraints activate. The seed-1 fold-0 baseline was archived at 2026-10-15T13:14:30.550093+00:00, before the parallel scheduler was introduced that afternoon. Concurrent workers do not explain that first failure.

On fold 0, bands improve seed 0 from 49.87% to 52.31%, but reduce seed 1 from 15.64% to 12.61%. The first four pre-constraint milestones match their respective baselines exactly. Bands achieve the requested 10% median gradient ratio in both cases; p95 is about 10.9%, below the 50% ceiling. The bad seed is not explained by an accidentally excessive calibrated bands weight.

## Interpretation and limits

UNETR has 92,685,555 randomly initialized parameters here and only 245 optimization updates. Seed changes both initialization and the deterministic training-case order. The evidence supports sensitivity to that combination under the established short budget; it does not separate initialization from sample-order effects, nor establish that seed 1 would remain poor after longer training.

The authorized seed-3 baseline/bands checks on folds 0/1/2 are supplementary. Keep all original seed-1 results and report seed 3 separately; do not selectively replace the poor seed in the original three-seed estimate. The cluster GPU differs from the Colab full A100, which must remain explicit in cross-environment comparisons.

![Validation curves](unetr_baseline_validation_curves.png)

Machine-readable evidence: audit_checks.json; reproducible checks: audit.py; original download hashes: download_receipts.json.
