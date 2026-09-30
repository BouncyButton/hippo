# Decisive presence experiments — 23 September 2026

Protocol and implementation: `experiments/presence_decisive_20260923/` in the
Desktop/hippo checkout. Historical function-head sources/checkpoints are read
from the isolated prior experiment; they are not modified.

## Execution record

- Job 666639: initial component audit stopped at a deliberately strict
  historical-case reproduction check on fold 1 / hippocampus_166.
- Job 666650: repeated-inference diagnostic completed. Three repetitions have
  identical masks. Original and new ASSD evaluators agree exactly at
  0.3504276702205217 mm, matching multiple previously saved re-evaluations.
  The original record is 0.3497679963486836 mm. Raw ASSD matches exactly at
  0.4753487416852803 mm. The historical rendered discrepancy is unresolved;
  it is not introduced by the copied renderer or new evaluator.
- Job 666651: component audit completed on all 208 cases against the frozen
  pre-existing E re-evaluation reference, preserving both references in reproduction.json.
  Cluster source directory: `/home/3160552/presence_decisive_20260923_v2`.
- Job 666653: capacity run stopped at initial wide-head equivalence; GPU
  rounding differed by up to 0.00178 logit. No new-head outer evaluation took
  place. A frozen initial-branch subtraction now guarantees zero initial
  correction, and every arm is restarted. The original stopped run is retained.
- Job 666656: scalar-presence feasibility diagnostic completed on all 208 cases;
  results in `PRESENCE_FEASIBILITY_RESULTS_20260923.md`.
- Job 666660: exact-initialization sequential run superseded early by fold-group
  scheduling to reduce elapsed time. Results are not reused.
- Job array 666669: same frozen training runs as two fold groups (1,2 and 3,4),
  respecting the cluster's two-job submission limit. Cluster source directory:
  `/home/3160552/presence_decisive_20260923_v4`.

Twelve local synthetic/integration tests pass, including exact renderer parity,
surface contribution accounting, true-voxel overlap versus mere ray presence,
empty-ray supervision, identical initial predictions across head interventions,
symmetry breaking in widened channels, detached-renderer gradient isolation,
FP-constrained selection, and training/checkpoint reload.

## Interpretation boundary

Stage 1 contains explicitly non-deployable oracle interventions. Stage 2 is a
four-arm width/pooling study on frozen E representations and fitted geometry,
with three presence-training seeds per fold. These are not independent
backbone seeds. All data remain internal development data. No candidate is
promoted solely because it improves mean ASSD or ray presence.

The full component results are in `DECISIVE_COMPONENT_RESULTS_20260923.md`.
Thin rays contribute 0.0341696091 mm (7.0723%) of E's mean ASSD under the exact
surface attribution; missed thin rays contribute 0.0150457490 mm (3.1141%).
These are descriptive surface contributions, not additive causal error shares.
Oracle correction of missed thin masks reduces ASSD by 0.0236126764 mm;
correcting missed thicker rays reduces it by 0.0641740075 mm. These
counterfactuals use labels and their effects are nonadditive.

Setting presence to one on true thin rays raises correctly overlapping thin
recall from 55.55% to 92.05%, but increases FP voxels from 393.16 to 423.31/case
and leaves ASSD essentially unchanged (0.483148 to 0.483316 mm). Empty-ray FP
counts remain exactly unchanged because this intervention changes only true
thin rays. This motivates the additional minimum-presence feasibility test.

## Real fitting-batch signed-gradient audit

The requested audit completed on 16 batches across four folds (26 distinct fitting cases). Existing E pushes presence upward on 430/448 erased thin-ray observations (95.98%) and raw foreground upward on all 734 true voxels in those rays. Loss-separated results, indirect background-gradient spillover under the diagnostic focal term, and interpretation limits are in `SIGNED_GRADIENT_RESULTS_20260923.md`. An exploratory follow-up measures the exact first-order mean-output response to shared-head SGD directions; this is not an AdamW training continuation. Its first attempt was terminated when the parent capacity allocation completed and it was restarted under allocation 666669. No training state was changed.

Capacity allocation 666670 completed folds 1–2. All 24 arm/seed selections retained E. The final unselected heads raised thin recall but also false positives; the second fold group is running under allocation 666669. The prespecified offset grid was limited to −1, −0.5, 0, 0.5, 1, so this does not exclude a benefit under more conservative calibration. Ranking diagnostics and the remaining folds are needed before concluding about capacity.

The shared-head response diagnostic also completed. Despite favorable local
output derivatives, E's full function-head SGD direction lowers erased-thin
mean presence in 8/16 batches. The actual historical focal coefficients, applied
to these saved-E batches, raise erased-thin and hard-empty mean presence in
16/16 batches with almost equal mean directional responses (+2.038763 and
+2.037817). This is local shared-parameter coupling, not a reconstruction of the
historical AdamW trajectory.

A value-matched gradient-routing helper now passes a test proving that stopping
head-parameter gradients preserves both raw-logit and decoder-feature gradients.
The proposed matched continuation protocol is in
`RENDERED_GRADIENT_FOLLOWUP_PROTOCOL_20260923.md`; it has not been launched.
The final two capacity folds remain running. At the latest snapshot, 34/48
head fits had completed and all completed selections retained original E.
