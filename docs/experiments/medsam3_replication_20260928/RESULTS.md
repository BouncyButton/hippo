# Completed MedSAM3 five-shot replication: observed results

All 27 fine-tuning runs and nine individual saved-mask/training audits completed. Original job 673549 ran 1:23:46 before the documented initial-loss precision audit correction; resumed job 673784 ran 10:58:24. The resumed Slurm job has FAILED status because the final calibration-consistency gate failed after all training/evaluation had finished. No training runs are missing.

The results show an average benefit for bands+edge, but do not establish a reliable improvement across data splits. Bands alone did not improve the overall mean. Preserve this as mixed evidence rather than confirmation of the original pilot.

| Model | Equal-fold/equal-seed mean Dice | Difference from baseline |
|---|---:|---:|
| MedSAM3 baseline | 82.91% | — |
| + Bands training loss | 82.87% | -0.04 percentage points |
| + Bands + edge training losses | 83.99% | +1.08 percentage points |

![Paired effects by fold](fold_effects.png)

## Variation across folds

Each entry averages the same three training seeds. Training uses five volumes per fold; evaluation uses 46, 52 and 47 volumes respectively (145 distinct evaluation volumes in total). Repeated predictions across seeds are not independent new volumes.

| Fold | Baseline | Bands | Bands + edge | Combined minus baseline |
|---|---:|---:|---:|---:|
| 0 | 82.68% | 83.47% | 85.41% | +2.73 pp |
| 1 | 83.06% | 83.60% | 84.01% | +0.95 pp |
| 2 | 82.98% | 81.53% | 82.55% | -0.43 pp |

The primary combined-versus-baseline effect is positive in 6/9 matched comparisons and 2/3 folds. Its exploratory paired crossed-bootstrap 95% interval is **[-0.61, +2.76] percentage points**, including zero. The exact two-sided fold-level sign-flip sensitivity test gives **p=0.50**. With only three fold blocks, inference is imprecise; this does not prove an absence of benefit either.

Bands alone improves 5/9 comparisons; its exploratory interval is [-1.58, +1.12] points. Adding edge to bands improves 8/9 comparisons and all three fold averages (+1.12 points overall; exploratory interval [+0.07, +2.08]). That incremental contrast was designated descriptive; its fold-level sign-flip p=0.25 and should not be promoted to a confirmed primary finding.

## Calibration deviation and integrity

The original cross-seed coefficient threshold (rtol=1e-5, atol=1e-8) is retained and remains failed. Observed maximum coefficient deviation relative to seed 17 is **0.1054%**, for the fold-1 bands coefficient. Initial adapter/frozen-parameter hashes and calibration cases/anchors match across seeds within each fold; maximum observed initial calibration loss difference is 1.53e-5. These observations are compatible with numerical gradient variation, but do not prove its cause or irrelevance to training.

This report uses the explicit `--report-calibration-deviation` analysis option. It retains all nine comparisons and their original protocols/results, records the failed gate in machine-readable output, and leaves the original cluster source, failed job status and logs intact. No coefficient, training procedure, raw prediction, score, split or statistical setting was changed to produce this report. The flag produces an observed-data report, not a passing strict-study audit. Thirteen local analysis/audit tests passed, including preservation of the strict failure, explicit deviation reporting, and rejection of unaudited or altered results.

Patient identity mapping is unavailable: this is volume-level evidence and cannot establish patient-independent generalization. Training/evaluation volumes and pilot cases are separated by the frozen protocol. Differences across folds combine training-set and evaluation-cohort changes. The pretraining provenance of the released medical adapter also remains unverified.

For a future stricter replication, compute and save each fold's training-only calibration once and reuse exactly the same coefficients across its seeds; keep all other settings fixed and use prospectively chosen evaluation splits. This is a follow-up recommendation; no new training jobs have been submitted.

## Detailed observed report

### All nine comparisons

All 27 fine-tuning arms completed and each run passed the independent saved-mask/training audit.

**The strict cross-seed calibration gate did not pass.** This is an explicitly requested descriptive report of all observed runs, with the original threshold retained. No coefficients, predictions, protocols or runs were changed or excluded. The cause and effect of the calibration differences remain unresolved; this is not a fully compliant replication claim.

| Fold | Constraint | Maximum relative coefficient difference | Original check |
|---|---|---:|---|
| 0 | bands | 0.002946% | failed |
| 0 | edge | 0.024488% | failed |
| 1 | bands | 0.105425% | failed |
| 1 | edge | 0.030773% | failed |
| 2 | bands | 0.004593% | failed |
| 2 | edge | 0.009522% | failed |

| Fold | Seed | Validation volumes | Baseline Dice | Bands Dice | Bands + edge Dice |
|---|---:|---:|---:|---:|---:|
| 0 | 17 | 46 | 0.835453 | 0.849107 | 0.865451 |
| 0 | 83 | 46 | 0.823506 | 0.834957 | 0.851188 |
| 0 | 191 | 46 | 0.821586 | 0.820149 | 0.845669 |
| 1 | 17 | 52 | 0.835619 | 0.838681 | 0.844303 |
| 1 | 83 | 52 | 0.821328 | 0.832596 | 0.850066 |
| 1 | 191 | 52 | 0.834704 | 0.836706 | 0.825847 |
| 2 | 17 | 47 | 0.812242 | 0.803372 | 0.806707 |
| 2 | 83 | 47 | 0.839457 | 0.833626 | 0.840399 |
| 2 | 191 | 47 | 0.837791 | 0.808829 | 0.829483 |

| Contrast | Mean change (Dice points) | Exploratory 95% interval | Positive runs |
|---|---:|---|---:|
| bands_edge_minus_baseline | +1.083 | [-0.610, +2.757] | 6/9 |
| bands_minus_baseline | -0.041 | [-1.575, +1.121] | 5/9 |
| bands_edge_minus_bands | +1.123 | [+0.074, +2.078] | 8/9 |

Repeated case/seed predictions are not independent. With three fold blocks, the minimum two-sided exact sign-flip p-value is 0.25. Bootstrap intervals are exploratory, not a significance declaration.

Only three fold blocks; bootstrap tails are approximate and no patient-independent claim is possible.

The original pilot is excluded. All training volumes are excluded from every evaluation set. Training support sets differ across folds; each is held fixed across three new optimization seeds. Coefficients are calibrated from training data only with a fixed calibration seed per fold. No validation-based tuning or early stopping is used.

Final post-run integrity verification passed for all 27 saved adapters, all saved prediction hashes, and all 320 frozen input files; see FINAL_INTEGRITY.json.
