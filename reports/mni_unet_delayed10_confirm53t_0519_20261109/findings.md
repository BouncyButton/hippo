# UNet delayed 10% bands confirmation

Original union bands, Dice-only AdamW LR 0.00016, no augmentation. Activation at update 470, ramp 76 updates, calibrated gradient target 10%, p95 guard 75%. Optimization cap 53,305,109,592,450 FLOPs; calibration accounted separately under the established convention. All constrained runs completed 938 updates; matched archived baselines completed 939.

| Seed | Fold | Baseline Dice (%) | Delayed bands Dice (%) | Gain (pp) | Evidence |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 56.53 | 61.25 | +4.72 | New confirmation |
| 0 | 1 | 57.71 | 66.87 | +9.16 | Reused pilot |
| 0 | 2 | 56.96 | 66.50 | +9.54 | New confirmation |
| 5 | 0 | 67.77 | 70.03 | +2.26 | New confirmation |
| 5 | 1 | 62.62 | 64.31 | +1.70 | Reused pilot |
| 5 | 2 | 62.99 | 65.81 | +2.82 | New confirmation |
| 19 | 0 | 65.78 | 71.33 | +5.55 | New confirmation |
| 19 | 1 | 64.67 | 69.92 | +5.26 | New confirmation |
| 19 | 2 | 64.28 | 70.25 | +5.98 | New confirmation |

All nine pairs: 9/9 positive; mean gain +5.221 percentage points; minimum gain +1.698 points.

Seven new pairs: 7/7 positive; mean gain +5.161 points.

Consistency criterion (all nine paired gains positive): PASSED.

Two development pairs were reused exactly, with no additional training. Folds overlap and all results use development validation, so this is not an independent-test or significance claim. No recipe or seed was changed during confirmation; no full-matrix promotion was made.

All seven new runs have verified durable archives, complete Dice/FN/FP/internal-swap diagnostics, and completed checkpoint cleanup. Frozen inputs and source, calibration records, curves, configurations, logs, and archive SHA-256 receipts are preserved in this report directory.
