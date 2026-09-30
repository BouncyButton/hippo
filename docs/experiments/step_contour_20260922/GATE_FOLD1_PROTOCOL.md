# Fold-1 contour trust-gate follow-up — frozen before DEV evaluation

> **Canceled, 2026-09-22.** Slurm job `665958` was stopped during D training at the user's request. Its A and D arm directories were deleted. No gate was fitted and fold-1 outer DEV was never evaluated. This is a historical protocol only.

Motivation: the fold-0 D head predicted lower/upper edges nearly as accurately as its segmentation, but its hard mask added false-positive occupied columns and reduced union Dice. A trained rule may identify the subset of disagreements where the head is more reliable.

## Cohorts and model

- Use the existing MSD fold-1 patient split: 208 outer-training and 52 outer-DEV patients. Fold-1 DEV is disjoint from fold-0 DEV, although both folds have appeared in previous exploratory work.
- Deterministically select 42 of the fold-1 outer-training patients as INNER-VAL (seed 0); fit the segmentation models on the remaining 166. No fold-0 DEV outputs or fold-1 outer-DEV labels enter model or gate fitting.
- Train matched seed-0 A (Dice only) and D (Dice plus image-conditioned head and all-pairs transition loss) using the original pilot's preprocessing, optimizer, batch size, no augmentation, 50-epoch cap, patience-8 checkpoint selection, and TRAIN-only gradient-based auxiliary weight calibration. Select the checkpoint on INNER-VAL foreground-class Dice.
- Extract the selected D model's probabilities, head presence and edge distributions, segmentation mask, and head mask on INNER-VAL. Save the same predictions on outer DEV only after the model and gate are fixed.

## Trust rule

The rule makes one decision per sagittal `(x,y)` column where D's segmentation and head disagree on union foreground voxels: retain the segmentation column or substitute the head column. It uses only outputs available without labels: head presence probability, segmentation occupancy and foreground probabilities, probability on added/removed voxels, head edge confidence and entropy, interval lengths, and signed edge displacements. The label for gate fitting is whether the head makes fewer union voxel errors than segmentation in that column.

Fit a standardized L2 logistic regression with patients weighted equally. Use five-fold **patient-grouped** INNER-VAL cross-validation to select regularization `C ∈ {0.03, 0.3, 3.0}` and acceptance threshold `{0.50, 0.60, 0.70, 0.80, 0.90}` by mean patient union-Dice improvement over unmodified D. If no candidate has positive cross-validated gain, select the no-op rule. Fit the chosen classifier on all 42 INNER-VAL patients. No outer-DEV labels are used in feature scaling, fitting, model selection, or threshold choice.

## Final analysis

On the 52 outer-DEV patients, report mean union Dice for A, D, D's full head mask, and D with the selected gate. Use paired patient bootstrap 95% intervals (2,000 resamples) for D–A, gate–D, and gate–A; report patient wins/losses, total disagreements, accepted columns, accepted head/segmentation wins, and head-only versus segmentation-only acceptance. A meaningful next step requires a positive gate–D interval and no concentration of harm in a few patients. A null result remains informative: it would indicate that these output-only confidence features do not predict when to trust the head.

The new run records exact source and dataset hashes, split membership, selected checkpoints and hashes, classifier candidates, case metrics, and a final Markdown result. Fold 1 is a replication relative to the fold-0 contour-head pilot, not an untouched prospective test cohort.
