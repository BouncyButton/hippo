# Outer-constraint combinations and replication

Use ordinary image-only SwinUNETR, 5% training data (10 patients), fold 0,
seed 0. The frozen parent is `w81_strength_qi70_qo10_mi65_mo85_b125_p75`,
with 81.3239687141% validation macro A/P Dice. Its exact outer-band and
endpoint coefficients are read from its verified completed configuration.

## Combination screen: 22 runs

First replay the parent without any additional losses. Require the same
stopping epoch, best epoch, and training/validation trajectory within 1e-6
absolute tolerance before running additions.

Test SDF, edge, depth, SDF+edge, SDF+depth, edge+depth, and all three.
Each subset receives a **joint** training-logit gradient budget of 2.5%, 5%,
or 10% relative to the complete frozen parent. Normalize each term first,
give terms equal normalized shares, then calibrate their sum. These are not
raw numerical loss percentages or per-term budgets. Calibration uses only
the ten training cases at the verified common epoch-five Dice checkpoint.

- SDF: existing full-grid anterior/posterior signed-distance loss, clipped
  at 8 mm on the verified 1-mm grid.
- Edge: existing signed foreground face-contrast error in the original
  two-step annotation bands.
- Depth: existing multiclass cross-entropy with 2-mm anatomical depth bins,
  an 8-mm foreground margin, and sparse-bin mass taper below 64 voxels.

All supervision uses training annotations. Inference takes only the MRI;
there is no auxiliary branch, guide input, plane view, or unwrapping view.
Preserve max75 epochs, StepLR20 gamma0.5, AdamW, the original early stopping,
five Dice-only epochs and five constraint-ramp epochs. No changes to the
incumbent outer/A/P experiment definitions or ordering.

## Replication: nine fresh runs after selection

After the incumbent and combination sweeps both finish, select the completed
ordinary-network max75/StepLR20 run with highest fold0/seed0 validation macro
Dice; break exact ties by objective ID. Freeze this choice before replication.
Candidate settings include the incumbent outer and targeted A/P runs as
well as the new combinations. Save the complete candidate score table and
hashes of both result tables.

Run the selected recipe at folds 0,1,2 crossed with seeds 0,1,2. Use the
existing three low-data splits: each has ten training cases and 52 validation
cases, with disjoint validation folds. Keep the training subset fixed within
each fold across seeds. Start every model afresh, including fold0/seed0.
Freeze the loss definitions, distances, focusing, relative budgets and timing.
Apply the same training-only calibration rule at each fold/seed's own
epoch-five checkpoint; never tune loss settings from replication validation.
Report every cell, fold-level mean/SD, and the overall descriptive mean/SD.
Fold0 was used for selection; these are validation replications, not test scores.

## Queue, checks and retention

Use one running incumbent job and one queued follow-up at most. Replace the
unused pending continuation with the follow-up; execute the 22 combinations
and nine replications sequentially in that single follow-up allocation.
Require the predecessor's completed status before starting.

The CPU checks cover exact original loss values and gradients, unchanged
outer coefficients, joint added strength, absent classes, depth-bin weights,
the copied training loop, deterministic checkpoint resume, fold isolation,
selection policy, and training-only replication calibration. A GPU gate in
the queued job checks all calibration terms on the actual ten training cases
and real SwinUNETR backward/optimizer updates before the control replay.

Keep the screen control and top five checkpoints. Preserve all results,
configuration, training curves and predictions; retire completed replication
checkpoints only after saving and verifying results. Never delete an active
resume checkpoint. Fail visibly if the predecessor or replay check fails.

## Submission

Submitted on 2026-10-12 as job **697891**, dependent on incumbent **697750**.
Cancelled only the unused pending continuation **697751**. The verified queue
contained exactly one running job (697750) and one pending job (697891).
The follow-up contains 31 sequential training runs: 22 combinations/control
and nine replications. All eight synthetic checks passed both locally and in
the cluster environment. Actual-model GPU preflight and the matched control
replay run when the dependency releases. Free BeeGFS quota at submission was
approximately 19.63 GiB.
