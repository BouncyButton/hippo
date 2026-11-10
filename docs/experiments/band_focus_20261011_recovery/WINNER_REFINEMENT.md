# Winner refinement: preserve the tail, prioritize the 80.725% configuration

The user authorized finishing the remaining asymmetric 3/2 mm runs, retaining useful penalty, endpoint and depth-shell comparisons, and replacing the lower-priority unfinished experiments with sensible combinations around `a32_side_qi60_qo15_mix75` (80.7251884% validation A/P macro Dice).

The experiment matrix is frozen in [winner_plan.json](winner_plan.json). The exact transition snapshot, preserved completion hashes, resume checkpoints, skipped identifiers, execution order and source hashes are recorded remotely in `winner_scope.json` at preparation. Completed outcomes are retained, including completed experiments belonging to families that are no longer scheduled.

## Fixed conditions

Ordinary image-only SwinUNETR, 5%, fold 0, seed 0; the same ten training and 52 validation cases; no auxiliary branch, augmentation or pretraining. Training remains max75, StepLR20 gamma0.5, the original AdamW settings and early stopping. The shared five-epoch Dice-only initialization, train-only calibration at epoch five, and constraint ramp over epochs six through ten are unchanged. The three previously removed longer-schedule runs remain excluded.

The starting loss uses 3 mm inside / 2 mm outside bands, Brier error, hardest 60% inside and 15% outside, 75% emphasis on each selected subset, and equal total inside/outside loss mass. Independent endpoint targets use 1+3 mm separations with equal distance weights. Selection fractions, subset emphasis, total side mass and calibrated term strengths are distinct controls.

## Execution order

1. Preserve and skip every already-completed experiment.
2. Finish all pending `a32_` variants, resuming any interrupted checkpoint exactly.
3. Check and calibrate all 287 new objectives on the ten training cases, with actual SwinUNETR parameter-update checks for representative mechanisms.
4. Run 16 prioritized new variants: fraction refinement, stronger endpoint feedback, separate side emphasis, inside-only cross-entropy mixtures and modest side-mass adjustments.
5. Run the 21 retained original comparisons: six BCE/hybrid penalties, twelve endpoint configurations and three depth-shell configurations.
6. Complete the remaining finite winner-refinement matrix.

Two sequential allocations use one 4g.40gb MIG GPU, four CPUs and 24 GiB RAM each, with 23h50 walltime. The worker stops launching arms after 22.5 hours and hands remaining work to its continuation. An unused pending continuation is cancelled after completion.

Submitted on 2026-10-12 as **697145**, followed by **697146** with `afterok:697145`. The transition preserved **217 completed runs**, retained **12 pending asymmetric variants first**, retained **21 older comparisons**, and added **287 new variants**, for **537 active experiments** including preserved results. Exactly **112 unfinished lower-priority variants** were removed from the active schedule. Three of the previously discussed last fifteen had finished while this amendment was being prepared.

The interrupted `a32_budget_b0p75_p1p0` checkpoint was verified at epoch five and remains first. The older parked `side_qi25_qo40` checkpoint remains preserved at epoch 41 but is not scheduled. Scope SHA-256: `b92c9c072804c0ab78a813433e6d9dd341d6b7c6b096c1f0e7f49f8ca9bc271a`. Submission quota check showed approximately **21.3 GiB free**. See [winner_scope.json](winner_scope.json) and [winner_submission.json](winner_submission.json).

Startup was confirmed on gnode01: job 697145 resumed the first pending asymmetric run and advanced to epoch 39 without an error. Its saved configuration confirms one MRI input channel, no auxiliary branch, max75, StepLR20 and unchanged early stopping. Slurm reports job 697146 pending on `afterok:697145`. See [winner_startup_verification.json](winner_startup_verification.json). The new-variant GPU calibration gate is pending until the retained asymmetric block finishes; the successful startup is not a claim that this later gate has already run.

## New variants

| Family | Distinct new arms |
|---|---:|
| Inside/outside selection fractions | 15 |
| Independent side emphasis × nearby fractions | 120 |
| Band/endpoint gradient strengths | 33 |
| Inside-only 10%/25% BCE mixture × side mass × endpoint strength | 48 |
| Total inside/outside loss mass × endpoint strength | 33 |
| Endpoint distance, side and eligibility variants on the winner | 12 |
| Depth-shell coverage on the winner and a nearby selection profile | 12 |
| Immediate neighboring band widths | 8 |
| Small side-specific local-ranking diagnostics | 6 |
| **Total** | **287** |

The main fraction grid is inside 50/60/70/80% × outside 5/10/15/20%, retaining the original 75% emphasis. Independent emphasis tests use eight nearby fraction profiles and inside 50/65/75/85% × outside 65/75/85/90%. Higher-order strength, penalty and side-mass interactions use a small set of explicitly specified profiles rather than an unbounded Cartesian product. Scientific aliases and already-tested identical configurations are deduplicated.

The 21 retained original comparisons retain their exact configurations, identities and frozen calibration. They mostly use the older 3/3 mm formulation; the new matrix additionally transfers endpoint and shell mechanisms directly onto the winner. Unfinished lower-priority runs are unscheduled, not presented as failed outcomes. Their prior evidence and any parked checkpoint remain available.

## Implementation and verification

New code is isolated in `winner_matrix.py`, `winner_objectives.py`, `winner_resume.py`, `winner_tests.py` and `run_winner.sh`. The frozen original and asymmetric source files and protocol are unchanged. New run identities include `objective_version=winner_v1`; a hashed amendment binds their implementation. Original run entries use the exact original objective path.

The extension independently overrides inside/outside emphasis, penalty and optional neighborhood ranking. Each side retains a contribution from every eligible voxel, with an additional contribution from its selected difficult subset. Empty sides are omitted and remaining side masses renormalized. Padding and voxels outside the specified bands receive no band-loss gradient. Inside-only BCE changes only the inside band's feedback before whole-term calibration.

Six local tests passed, covering all 287 finite objective/gradient/calibration budgets; exact values and gradients for frozen controls; explicit independent-side reductions; all-background/all-foreground/empty-valid-support behavior; confident inside errors and finite-difference gradients; preserved source identities and execution priority; and exact interrupted-versus-uninterrupted training for the two new principal mechanisms. Shell syntax also passed. See [winner_local_checks.json](winner_local_checks.json). A mandatory GPU gate precedes every new scientific run; its result is recorded separately when executed.

The existing storage policy retains fixed controls and the top five checkpoints, together with the shared warmup, active/parked resume checkpoints, and all metrics, configurations, logs and predictions. Only aggregate experiment metadata is copied to the local documentation.

These are adaptive development experiments on reused validation data. A new best score requires confirmation on an independent seed/fold before generalization claims.
