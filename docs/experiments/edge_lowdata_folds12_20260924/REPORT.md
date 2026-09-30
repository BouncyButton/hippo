# Completed additional-fold comparison

Job 668085 completed successfully, exit 0, elapsed 33m42s. All six requested models and train/validation audits finished. Fold 0 below is the retained previous experiment.

Validation macro Dice, mean across initialization seeds 0/1/2; differences in percentage points.

| Fold | Bands (%) | Bands + edge (%) | Gain (pp) |
|---|---:|---:|---:|
| 0 | 77.081 | 77.775 | +0.694 |
| 1 | 75.535 | 76.303 | +0.767 |
| 2 | 76.771 | 77.354 | +0.584 |
| Mean | 76.462 | 77.144 | +0.682 |

Bands+edge improves selected-checkpoint validation Dice in all six new runs (and all three retained fold-0 runs). Boundary foreground FN+FP decreases in every new run. The corresponding new-fold means are:

| Fold | Bands boundary FN+FP | Edge boundary FN+FP | Relative reduction |
|---|---:|---:|---:|
| 1 | 76310.33 | 72493.67 | 5.00% |
| 2 | 67283.33 | 64670.33 | 3.88% |

Same fixed 10 training cases and 52 validation cases per fold across the three seeds. This is a development-fold result with validation-selected checkpoints, not an independent test result. Seeds reuse validation cases and do not add independent cohorts. No existing control training was repeated. No degree weighting is used.

Full per-seed metrics, train/validation boundary counts, checkpoint selection and learning-curve summaries are in SUMMARY.json. Existing fold-0 details are in ../edge_lowdata_20260924/REPORT.md.
