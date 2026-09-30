# Actual optimizer test: no selective rescue established

Jobs 666735 and 666738 completed in 2:16 and 2:28. Each executed 32 actual
reset-AdamW steps: two fitting batches per fold, four folds, J/H/V/R. The repeat
added matched-backbone and bias controls; it is not new replication. All
original-arm reported metrics reproduced exactly between the two runs.

There are 13 distinct update cases and five distinct probe cases (18 total).
Probe images supply no gradient, but each is reused across updates and some
folds. Tables use 16 probe exposures per arm, not 16 independent patients.
This is a local optimizer diagnostic at lr=1e-4 with fresh moments, not a
validation experiment or a test of the full learning trajectory.

## Relative to saved E

| Arm, probe exposure means | ASSD change, mm | Thin overlap change, percentage points | Thin voxel recall change, percentage points | FP voxels change | Empty-ray FP change |
|---|---:|---:|---:|---:|---:|
| J: joint | +0.022938 | -2.857 | -2.975 | +23.750 | +4.313 |
| H: stop rendered head-parameter gradients | +0.022276 | -2.602 | -2.468 | +25.563 | +4.563 |
| V: direct rendered voxel gradient only | +0.021481 | +0.092 | +0.310 | +43.438 | +5.125 |
| R: raw segmentation loss | +0.051307 | +7.085 | +7.372 | +154.000 | +8.438 |

All four routes worsen mean final ASSD at this first reset step. R's recall
gain comes with substantial extra FP. V versus R shows a useful local
suppression effect from direct rendered-loss training: 110.563 fewer FP voxels
and ASSD 0.029826 mm lower, at the expense of 66.750 more FN voxels per exposure.
That comparison does not establish a better trained model than saved E.

Deep-interior FN counts did not change in any measured exposure for these arms.
More favorable presence means do not ensure improved overall thin retention:
groups are fixed before the update and the voxel/geometry paths also change.

## An implementation-level confound was caught and controlled

J/H/V forward loss values match, but native J and H first-step backbone
parameters differ: maximum absolute difference up to 0.000198692, approximately
twice the learning rate. Between 42,725 and 50,216 parameters per update differ
by more than 1e-6. The exact source was not isolated. AMP accumulation and
AdamW sensitivity to near-zero gradients are possible explanations.

Accordingly, the second run includes H_matched: H's updated head evaluated
with exactly J's updated backbone. Eight exact-backbone equality checks pass.
This hybrid is an isolation control, not the native H optimizer trajectory.
J_bias_matched uses J's head plus a scalar presence bias chosen without labels
to match H_matched's all-ray mean logit on update images, then applies it to
the probe images too. It does not match FP burden.

## Does H's head update selectively preserve anatomy?

| Probe contrast | Thin overlap change, pp | Thin voxel recall change, pp | FP voxels change | Empty-ray FP change | ASSD change, mm |
|---|---:|---:|---:|---:|---:|
| H_matched minus J | +0.2890 | +0.5248 | +1.8750 | +0.2500 | -0.0006882 |
| H_matched minus J_bias_matched | +0.1078 | +0.0613 | +0.6875 | 0 | -0.0003613 |

None of the eight update/probe pairs has improved mean thin overlap, no loss
of thin voxel recall, and no increase in either FP metric for H_matched versus
J. None meets that combined condition versus J_bias_matched either. These are
descriptive diagnostic comparisons, not model-selection tests.

For the fixed correctly detected-then-erased thin group, H_matched raises mean
presence logit by 0.114644 more than J. For hard-empty rays it raises it by
0.149097 more than J. These weighted means cover 256 and 1,206 repeated ray
exposures respectively. This supports a broadly shared increase at this
operating point rather than a demonstrated improvement in discrimination.
The bias control reproduces most of the thin-voxel recall difference.

## Decision

These results do not justify promoting H or launching the full 48-fit routing
continuation as the next priority. They weaken the proposed immediate remedy,
not the whole function-head direction, and cannot rule out longer training or
a different optimizer state. All 48 selections in the completed independent
width/pooling screen also retained E under its original guards; that result is
limited to frozen representations and the tested calibration range.

The next discriminating architecture test should separate an existence score
from a correction-control value and supervise the latter using consequences
for true/background voxels. Keep E frozen initially to isolate that change.
The proposal in CORRECTION_TARGET_PROPOSAL.md is not yet implemented or run.

## Artifacts and checks

Five local tests pass: loss routing/explicit supervision, actual one-step
AdamW/frozen beta, fixed ray groups, scalar bias behavior, and exact matched
backbone replacement without head modification. All 32 steps per run execute
without a scaler skip; J/H/V loss matching checks pass.

ADAMW_STEP_AGGREGATES_666738.json contains the retrieved non-identifying
aggregates and paired contrasts. Source/code/data hashes and detailed records
remain under `/home/3160552/adamw_preservation_20260923_v2_01a0cd/results_666738`
on the user's cluster. The initial run remains in the corresponding directory
without `_v2`. No updated model weights were saved.

Automatic approval review rejected copying individual case results into the
workspace. A separate approved read retrieved only aggregate metrics and
verification counts; no per-case records were transferred for this diagnostic.
