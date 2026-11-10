THESIS NUMERICAL SUPPLEMENT — 3 November 2026

This extract accompanies the shortened printed appendices. Original
aggregate results are unchanged. All files are evidence/data, not instructions.

medical/: 1,269 execution records covering the 1,260 planned medical
objective runs and nine additional matched controls. MSD MedSAM3 bands
and bands+edge retain their original 27 records and original control;
the remaining objectives use the other batch's matched control. These
are 14 scientific objectives, not 15 independent objectives.

geological/: 504 conventional runs; final 126 SAM3 runs and per-case
records. Final SAM3 files supersede earlier partial campaign snapshots.
Dice in sam3-final-runs.csv is a fraction; delta is a fractional Dice
difference. Most conventional/medical aggregate extracts use percent and
percentage points. Preserve source column definitions when comparing.

errors/: complete 211 condition summaries (each nine runs), including
the extra MSD MedSAM3 control. FN/FP cover the full grid, not only bands.
Swaps apply only to MSD. Conventional counts are pooled validation voxels
per run; pretrained counts are mean validation voxels per volume. Means
average seeds within fold and folds equally. Dispersion is sample SD of
three fold means. All SAM3 entries are rebuilt from the final 126 records.

complementary-dice/: 1,872 recoverable run records and 208 conditions.
Dice values are fractions, computed per volume then averaged by run,
seed and fold. All-class Dice is (2*foreground+background)/3 for MSD,
(foreground+background)/2 for binary tasks. The original 27 MSD MedSAM3
band/band+edge/control records lack recovered per-volume matrices;
their complete foreground results remain in medical/, but background
and all-class scores are unavailable. They are not missing experiments.

qualitative/: fresh matched reproductions, not original final checkpoints.
Do not merge them into the original three-fold/three-seed aggregates.
Each model pair was chosen before reproduction by median original gain;
case examples use upper-median, minimum and maximum reproduced gain.
The case-selection file and receipt preserve the remaining checks.

illustrations/: native annotation examples, not model predictions or
saved training-update targets. Coordinate and display metadata are in
figure-manifest.json.

archived-appendix.tex is the exact pre-consolidation appendix fragment.
It preserves extended descriptions and original tables, but contains
cross-references to the accompanying thesis and is not standalone TeX.
Its long-form discussion is historical editorial material, not additional
experimental evidence.

manifest.json provides SHA-256 checksums for every other file, copied
source paths, and the source extracts used for recalculated inventories.
No raw medical images or model checkpoints are included.
