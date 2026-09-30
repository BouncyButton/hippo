# Saved-head discrimination audit

Written before running this diagnostic. This is a retrospective audit of the
24 final, unselected heads from job 666769, not a new model-selection experiment.

Use all 166 fitting and 42 inner cases in each original fold. Evaluate both
objectives and all three head-fitting seeds. No outer images, weight updates,
checkpoint reselection or changes to the completed experiment are allowed.

For true thin rays, other true rays, hard-empty rays, other empty rays, and
feasible initially missed thin rays, record the distribution of learned
renderer-control corrections: count, mean, standard deviation and quantiles.
Compute within-case AUROC for thin versus hard-empty and feasible missed thin
versus hard-empty using correction alone, original anchor, and final control.
Average defined case AUCs; report distributions within each fold/seed rather
than treating overlapping fitting folds as independent patients. Constant
corrections have AUC 0.5 whenever both groups exist.

Recompute label-dependent feasible intervals on fitting and inner data solely
as retrospective diagnostics. Inner reference intervals never enter fitting,
constant selection, optimizer steps, or model inputs. Their availability at
audit time does not make them deployable targets or inference features.

Compare the learned correction with E and two constants chosen using fitting
data only: the case-averaged learned correction, and the constant in [-4,4]
minimizing the arm's original case/group-balanced fitting objective. Include
the endpoints and zero in constant minimization. Apply the same constants to
inner data, without re-estimating them there. Compare fitting objective values,
mask disagreement, true-voxel recoveries/deletions, thin overlap/voxel recall,
FP rays and FP voxels. ASSD is not recomputed for this discrimination diagnostic.

Interpretations to test:

- Near-constant corrections, chance ranking and similar constant-mask outcomes
  on fitting data undermine a pure held-out generalization explanation.
- Good fitting discrimination with poor inner discrimination suggests a
  generalization problem.
- Useful ranking but worse-than-constant voxel outcomes suggests that correction
  amplitude or renderer activation thresholds defeat the apparent ranking.
- Lower fitting loss than the best constant establishes input-dependent benefit
  for that objective; it does not establish selective anatomical rescue.

If fitting discrimination is poor, optimization and representational limits
remain unresolved until a small controlled overfit/feature-access experiment.
No architecture change is inferred automatically from a low AUC.

Record saved checkpoint/source/data/split hashes and no-gradient assertions.
Write only aggregates from this audit; individual diagnostic records and image
data must not be exported. Report all 24 heads, including null findings.
