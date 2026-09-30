# Coherence audit completed

Job **673837 completed successfully** in **4 minutes 36 seconds** (exit 0).
All 27 original checkpoints were evaluated. Results were retrieved on 29 September
2026 into `reports/edge_coherence_20260928/results/`.

**Finding:** signed boundary contrast improves in all nine fold–seed comparisons.
Outer hard-label disagreement is approximately unchanged versus baseline, while
inner disagreement worsens in all nine comparisons. There is no uniform
improvement in voxel coherence or global connectedness.

Read [the interpretation](../../../reports/edge_coherence_20260928/INTERPRETATION.md)
and [the full numerical report](../../../reports/edge_coherence_20260928/results/REPORT.md).

## Verified existing evidence

Equal means over the nine fold–seed comparisons. FN and FP are mean counts per 52-case validation split, not independent nine-cohort totals.

| Measurement | Baseline | Bands | Bands + edge |
|---|---:|---:|---:|
| Macro Dice (%) | 72.633 | 76.462 | 77.144 |
| Foreground FN | 9855.333 | 12590.889 | 13548.111 |
| Foreground FP | 97389.889 | 70599.778 | 65326.889 |
| Mean surface distance (mm) | 1.129 | 0.942 | 0.909 |
| Surface Dice at 1 mm (%) | 65.082 | 75.717 | 77.624 |

False negatives increased by 37.5% versus baseline and 7.6% versus bands. False positives decreased by 32.9% versus baseline and 7.5% versus bands. Thus the improvement is predominantly reduced over-segmentation, with a recall tradeoff.

Surface distance and 1 mm surface Dice improve in all nine comparisons against both baseline and bands. This supports boundary placement, but does not establish neighbor coherence. Dice depends on overlap counts and does not encode the adjacency arrangement of errors.

## Completed analysis

The audit measures probability contrast error, hard disagreement separately in inner/outer same-side pairs, correct true boundary transitions, and foreground fragmentation. A baseline volume-matched control tests how much can be reproduced by changing foreground extent. Case effects are paired across models, and seeds are averaged before bootstrap resampling.

The completed outputs are `REPORT.md`, `SUMMARY.json`, `cases.json`,
`bindings.json`, `completion.json`, and `coherence.png`/`coherence.pdf` in the
local results directory. Cluster originals remain under
`/mnt/beegfsstudents/home/3160552/edge_coherence_20260928_01/results`.

Four synthetic metric tests and an aggregation/report smoke check passed. Original model weights and training runs are unchanged. Local inference was benchmarked at approximately 2.7 seconds per case; the user chose to retain the queued GPU audit.

## Historical pause assessment on 28 September

The user subsequently authorized pausing MedSAM3 only if it could be resumed
after this audit. Read-only checks found that account 3160552 has AdminLevel
None and no coordinator role. Slurm's suspend/resume commands require a
privileged user or account coordinator. The deployed trainer and worker hashes
match the inspected local files. They save final adapter weights, but not the
optimizer and RNG state required to resume an interrupted training arm. The
worker can skip completed audited comparisons; this is not mid-arm resumption.

No pause, cancellation, requeue, or process signal was issued. MedSAM3 job
673784 remains running; fold0_seed17 and fold0_seed83 are complete, with
fold0_seed191 in progress at this check. Audit 673837 remains queued under
QOSMaxJobsPerUserLimit. This preserves the user's requirement not to interrupt
work without a safe resume path.
