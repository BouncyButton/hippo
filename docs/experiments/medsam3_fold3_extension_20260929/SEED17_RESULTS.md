# Fold 3, seed 17: verified result

Recorded 2026-09-29. Slurm job 675308; all three models completed 200 updates and prediction/scoring on the same 44 validation volumes. The job continued to seed 83 without intervention.

| Model | Mean 3D whole-hippocampus Dice | Difference from baseline (percentage points) |
|---|---:|---:|
| baseline | 0.85241004 | 0.00000 |
| bands | 0.85579544 | 0.33854 |
| bands_edge | 0.85635468 | 0.39446 |

Bands+edge minus bands: **0.05592 percentage points**.

The audit passed: 264 NIfTI outputs checked, matched initialization/sample schedule/learning rates, unchanged frozen backbone, and active supervised training constraints. All three initial loss components matched exactly. Remote results SHA-256: `51313dc25c7efc7a5984461b91820e91a3dd97c8d588806abcaaf23729266e8f`. Retrieved metadata are preserved in seed17_metadata.json; these are reserialized JSON, not byte-identical copies of the remote files. Mean Dice was independently recomputed from all 44 per-case scores and agreed within 1e-12.

## Interpretation

Both constrained arms improve the mean on this seed, but the gain is modest; edge adds only 0.056 points beyond bands. This is promising evidence for continuing the already-submitted study, not proof of a general improvement. Seed 83 and seed 191 remain to be assessed. The earlier negative fold 2 remains part of the overall evidence. This fold was added after inspecting earlier outcomes, and any outcome-dependent stopping makes the extension exploratory. Patient grouping is unknown, so volume-level splits do not establish patient-independent generalization.

The user retains the decision to stop or delete anything. No job or artifact was changed; the seed-17 notification heartbeat is paused after reporting this result.

