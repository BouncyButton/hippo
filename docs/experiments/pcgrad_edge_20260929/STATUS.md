# PCGrad edge pilot: completed

Job **676122** completed successfully on 29 September 2026 at 17:05:59 Europe/Rome, exit 0, elapsed 50 minutes 30 seconds. All six runs finished. Six GPU preflight checks passed; all-ten-case gradient-conflict confirmation passed for all three seeds. No optimizer updates were skipped.

Primary matched-control versus PCGrad validation macro Dice (%):

| Seed | Control | PCGrad | Control selected/stopped epoch | PCGrad selected/stopped epoch |
|---|---:|---:|---|---|
| 0 | 77.4542 | 52.6761 | 64 / 67 | 61 / 69 |
| 1 | 79.3703 | 66.3309 | 63 / 71 | 58 / 66 |
| 2 | 76.2885 | 42.2383 | 73 / 75 | 62 / 70 |
| Mean | 77.7043 | 53.7484 | — | — |

PCGrad improves inner pair correctness and reduces false negatives, but increases false positives and disconnected components while worsening outer correctness, correct boundary transitions, and Dice in every seed. This budget does not demonstrate better anatomical generalization. Longer-budget behavior remains untested; no follow-up training was launched.

Six selected checkpoint hashes were rechecked remotely against completion records. All downloaded selected-case files match their completion hashes. Full per-case and per-update diagnostics are retained in reports/pcgrad_edge_20260929/results/. Interpretation and anatomical comparison are in reports/pcgrad_edge_20260929/INTERPRETATION.md and ANATOMY_SUMMARY.json.

The separate inference-only job 676160 recomputes bands metrics for the prior pooled/separated experiment, without changing its source/results or checkpoints. The user will review both analyses before deciding future experiments.
