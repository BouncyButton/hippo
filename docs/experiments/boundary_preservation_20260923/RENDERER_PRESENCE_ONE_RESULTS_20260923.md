# Presence-one renderer test: thin anatomy returns, background protection collapses

Job **666877** completed the single predefined intervention on **208 cases** in
**2m06s**, exit code zero. Presence was set to exactly one only in the renderer;
raw logits, fitted boundaries, smoothing results, beta and renderer shape were
unchanged. No training or selection occurred. See the
[frozen protocol](RENDERER_PRESENCE_ONE_PROTOCOL_20260923.md).

**The explicit presence multiplier is a major source of thin-ray suppression in
saved E, but simply removing it makes this renderer unusable.** This tests the
frozen trained architecture, not the feasibility of training a boundary-only
correction architecture without a presence head.

## Full-case results

Entries are case means. The raw column is E's own voxel output, not model A.

| Metric | E raw | E rendered | Renderer presence = 1 |
|---|---:|---:|---:|
| ASSD, mm | 0.523650 | **0.483148** | 7.350005 |
| Union Dice | 0.877790 | **0.886133** | 0.131370 |
| HD95, mm | 1.3829 | **1.2395** | 26.6388 |
| Thin-ray presence recall | 85.97% | 57.13% | 97.21% |
| Thin-ray true-overlap recall | 83.12% | 55.55% | 92.05% |
| Thin-voxel recall | 78.79% | 53.84% | 86.47% |
| FP rays/case | 80.75 | **28.94** | 3,480.99 |
| FP voxels/case | 605.09 | **393.16** | 45,037.95 |
| FN voxels/case | **236.93** | 355.71 | 263.40 |

ASSD, union Dice and both FP burdens worsened in **all 208 cases**. Thin-overlap
recall increased in 207 cases and tied in one. Thin-voxel recall increased in all
208 cases. No existing E foreground voxel was deleted or relabeled, as expected
from the monotonicity of increasing presence with everything else fixed.

The intervention recovered **19,200 true foreground voxels** and added
**9,286,116 false-positive voxels** across the 208 cases. It made 724,046 of the
739,064 truly empty rays contain predicted foreground (97.97%). Thus the harm
extends far beyond restoring the raw network's false positives: the rendered
output's FP burden greatly exceeds E raw.

The paired ASSD change was **+6.866857 mm**, conditional case-bootstrap 95%
interval **[+6.783496, +6.949370] mm**. This interval conditions on the saved
models and reused development cases; it does not establish external uncertainty.

## Exact recovery of previously deleted thin rays

The historical definition is reproduced: nonborder true rays with foreground
span at most two voxels, **9,031 rays** total. Pooled ray counts below differ
from the case-mean percentages above.

| Endpoint | E raw | E rendered | Presence = 1 |
|---|---:|---:|---:|
| Any predicted foreground on a true thin ray | 7,760 | 5,105 | 8,774 |
| Predicted foreground overlaps a true voxel | 7,487 | 4,953 | 8,308 |

- All **2,655** raw-present thin rays erased by E became present again.
- Of those 2,655 rays, **2,476** had true overlap afterward; 179 still contained
  only misplaced foreground. Restored presence alone overstates anatomical rescue.
- Restricting the denominator to rays where raw E actually overlapped anatomy,
  E erased overlap on **2,534** rays. Presence one restored true overlap on
  **2,502 (98.74%)** of them.
- Of all **4,078** thin rays lacking true overlap under E, **3,355 (82.27%)**
  recovered overlap; **723** remained missed. These counts reproduce the earlier
  label-guided maximum-presence feasibility finding, but this intervention also
  exposes the cost on empty rays because it is applied everywhere.

An additional count-based definition, one or two true voxels irrespective of
border/span, includes 9,051 rays and 2,662 raw-present/E-erased rays. All 2,662
became present; 2,483 acquired true overlap. These alternative denominators are
reported in the aggregate and are not substituted for the historical definition.

## What still prevents recovery at presence one?

The 723 remaining thin rays without true overlap partition as follows:

| Endpoint diagnosis | Rays |
|---|---:|
| Raw E overlapped truth, but frozen geometric rendering still suppresses it | 32 |
| Raw E had no true overlap; geometry adds foreground support on at least one true voxel, but not enough | 385 |
| Raw E had no true overlap and geometry adds no positive support on any true voxel | 306 |

The last two groups can have joint raw-logit and geometric deficits. These are
descriptions of the fixed endpoint, not independently estimated causal shares.
They do not attribute total ASSD to thin rays. In particular, boundary coordinates
were not replaced with ground truth or refitted without presence weighting.

## Fold consistency

| Fold | E ASSD | Presence-one ASSD | E thin overlap | Presence-one thin overlap | E FP rays/case | Presence-one FP rays/case |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.481385 | 7.408532 | 56.73% | 94.71% | 30.38 | 3,534.87 |
| 2 | 0.492962 | 8.064645 | 53.49% | 95.96% | 31.98 | 3,547.54 |
| 3 | 0.473597 | 7.102874 | 54.18% | 86.07% | 23.62 | 3,427.17 |
| 4 | 0.484647 | 6.823969 | 57.81% | 91.44% | 29.79 | 3,414.38 |

## Architectural implication

The test supports the user's concern about the presence gate's authority:
removing this one multiplier, conditional on saved E, restores nearly all
raw-correct thin rays whose overlap was deleted. It also shows that the current
geometry renderer relies on the same multiplier for background protection.

Setting presence to one does **not** delegate foreground detection to the voxel
network. It tells the renderer to express its fitted interval on every ray,
including empty ones, and lets the resulting logit correction override the raw
network. Empty rays still have fitted coordinates, although those coordinates
do not necessarily describe an anatomical structure. This explains why the
intervention is not equivalent to the proposed division of responsibilities.

Consequently, a boundary-only redesign would need the correction to become
inactive away from voxel-supported boundary regions, rather than globally
asserting presence. Its authority to create or remove foreground would need
explicit bounds. How to define support without excluding real raw-missed thin
rays remains an unresolved design question. This experiment does not test such
a redesign, and does not justify removing presence from the smoothing fit.

**Retain E unchanged.** The inference toggle fails decisively; the architecture
question remains open with a more clearly identified mechanism.

## Verification and artifacts

Three tests passed: exact renderer reconstruction and monotone intervention with
unchanged inputs; one-voxel rescue versus residual geometric suppression; and
distinguishing restored ray presence from correct overlap. Launcher syntax and
uploaded manifests passed. Every case reproduced historical E ASSD and Dice
within 1e-4; all source/data hashes, checkpoint hashes, no-gradient assertions,
and nonduplicated 52-case-per-fold coverage checks passed.

Only the non-identifying summary was downloaded under the user's standing
authorization. Images and individual records remain on the cluster. This is a
single intervention on previously reused internal cases with one original model
seed, not an independent confirmation cohort.

- [Aggregate metrics, counts and conditional intervals](RENDERER_PRESENCE_ONE_SUMMARY_666877.json)
- Implementation: `experiments/presence_decisive_20260923/audit_renderer_presence_one.py`
- Tests: `experiments/presence_decisive_20260923/test_renderer_presence_one.py`
- Remote results: `/home/3160552/renderer_presence_one_20260923_01a0cd/results_666877`
