# Geometry repair and continuation of job 686405

Job 686405 was cancelled on the user's instruction after the 5% unfolding stage failed. Completed results are preserved in the original campaign root. The continuation reuses three completed main runs (5% Dice, 5% depth and full-data Dice) and two completed 5% guide fits. The interrupted full-data depth run resumes from epoch 12, update 2496, with model, optimizer, scheduler and random state restored. No completed training is repeated.

## Root causes and fixes

1. A low-data guide produced a flat, closed, two-sided surface component. Cutting it with a parallel plane could retain the same component twice without opening it. The generic libigl disk-cut routine leaves sphere-topology components untouched. Closed components now receive a topological edge seam that duplicates seam vertices and preserves every original triangle. The [libigl API documentation](https://libigl.github.io/libigl-python-bindings/api/igl/#cut_to_disk) describes the closed-sphere exception.
2. Long, complicated disk charts can collapse numerically under a single convex-boundary harmonic parameterization. Such charts are now partitioned along their principal face-centre direction into connected subcharts and parameterized separately. Every source triangle must occur exactly once; hole filling is disabled. The small-triangle check scales with average source triangle area. Plane controls use the same resulting chart membership and tensor sizes.
3. The worker previously skipped branch experiments after a failed preparation and continued with the next data regime. It now records the failed stage and stops immediately.

These are geometry and execution repairs. Models, training objectives, data splits, loss coefficients, epoch limits and stopping settings are unchanged. There are no added segmentation losses or shape constraints. The original two cut sides remain distinct; additional charts are geometric patches, not additional anatomical bodies.

## Verification

- The flat-component regression fails against the original implementation (surface area is duplicated), then passes with the seam fix.
- A synthetic long capped tube reproduces numerical chart collapse; adaptive partitioning preserves its complete triangle multiset and yields valid charts.
- Seven focused tests pass, including fail-fast execution and preservation of native training identities.
- All 114 available guide predictions pass: 62 low-data inputs (10 out-of-fold training cases plus 52 validation cases), and 52 full-data validation inputs. Future full-data out-of-fold guide predictions still require the mandatory in-job preflight.
- Full SwinUNETR forward/backward passes on the repaired hippocampus_093 input with 80 charts and nonzero auxiliary/fusion gradients.
- A real optimizer smoke run resumes across the geometry repair, recovers interrupted best-checkpoint publication and verifies that completed runs return without training again.

## Provenance and continuation

The repaired branch experiments have a new protocol identity. Native and guide training retain their verified original identity because their training configuration and model/loss implementation are unchanged. Both protocols and imported artifact hashes are checked before submission and execution.

The continuation root is `/mnt/beegfsstudents/home/3160552/unwrapped_swin_fold0_seed0_20261006_02`. Imported checkpoints are hard-linked; subsequent checkpoint saves use atomic replacement, preserving the original files. The old source snapshot and original job receipt remain intact. The continuation schedules 17 unfinished main runs (16 new branch runs plus one interrupted native run) and two unfinished full-data guide fits in one allocation.
