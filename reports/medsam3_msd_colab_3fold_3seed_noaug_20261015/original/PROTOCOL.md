# MedSAM3 BG/A/P replication: folds 0, 1, 3 × seeds 17, 83, 191

The user requested three seeds on folds 0, 1 and 3 following the multiclass fold-0 seed-46 pilot. There are nine matched comparisons, each with baseline, bands and bands+edge: 27 newly fine-tuned adapters. The seed-46 pilot remains separate and is excluded from this replication's average.

Use the previous binary study's frozen support and validation sets for each fold: five support volumes per fold; 46, 52 and 44 validation volumes respectively. The three support sets total 15 distinct volumes; the validation sets total 142 distinct volumes and exclude every support volume. Retain all new runs regardless of outcome. These folds were selected after prior binary outcomes, so this is exploratory and is not a full or unbiased cross-validation estimate. Patient identities remain unavailable.

The class model, text prompts, semantic probability projection, native objectives and classwise constraints are identical to the completed three-class pilot. Keep 200 optimizer updates, two adjacent-slice pairs per update, learning-rate schedule, warmup and LoRA initialization unchanged. Each arm independently resets the same adapter and optimization RNG, and replays the same sampling schedule for that comparison. Save all final checkpoints, training traces and three-class predictions for training and validation.

## Exact calibration per fold

Calculate training-only calibration once in seed 17 for each fold, with calibration RNG seed 20260929. Seeds 83 and 191 reuse that exact audited calibration JSON and its SHA256. The reuse contract binds support-image/label hashes, class definitions/prompts, probability projection, constraint definitions, initial adapter and frozen parameter hashes, and calibration settings. The same bands coefficient is used in both constrained arms. No validation-dependent tuning or early stopping occurs.

## Audit and reporting

After each comparison, independently reload the saved masks and probabilities. Verify input/output/checkpoint hashes, probability normalization, mask=argmax, voxel grids, all 3×3 confusion counts and per-class Dice. Check all 200 training records, the objective components and weights, matching initialization/sample/LR schedules, unchanged frozen parameters and distinct final adapters. Require exactly identical calibration files across the three seeds in a fold before aggregation.

The primary Dice is the arithmetic mean of anterior and posterior Dice for each validation volume; background Dice is reported separately. First average volumes within each seed, then equally average the three seeds within each fold. Report seed standard deviation, paired gain over baseline and number of improved seeds. An overall summary weights the three folds equally.

Primary error categories are disjoint: foreground FP = background predicted as anterior/posterior; foreground FN = anterior/posterior predicted as background; A→P and P→A = swaps. Report each as mean voxels per validation volume, then equal-seed averages per fold. Also report class-specific FP/FN, which include swaps and must not be added again to the disjoint categories. Preserve all individual seed summaries and per-volume records.

## Execution and storage

Submit one 12-hour GPU job per fold, with its three seeds sequential. An accompanying checkpoint-delivery process uses the already verified login-node file `/tmp/medsam3-3160552-staging/sam3.cdn.pt` and `sbcast` to each allocated job's private tmpfs. The checkpoint is never copied to BeeGFS. A CPU aggregation job depends on successful completion of all three fold jobs; it requires all nine independent audits before writing the final Markdown/JSON report.

Uncompressed prediction/adapter budget after two-copy mirroring is 5.999 GiB for all nine comparisons. Require 7.499 GiB free before submission: output budget + 1 GiB reserve + 0.5 GiB startup headroom. Compute-node guards recheck the budget of outstanding comparisons plus the 1 GiB runtime reserve. The authorized removal of binary generated artifacts occurs only after saving the complete binary results in Markdown and hashed metadata. Pretrained weights, datasets, shared runtimes, source and the multiclass pilot are retained.
