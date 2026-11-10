# Background Dice: archive recovery

Inventory generated 2026-10-23T13:37:57.856045+00:00.

Original evaluation metadata was recovered from local archives, both connected Google Drive accounts, and the historical cluster archive. No training or inference was run. The thesis source and its reported foreground Dice were not changed.

## Coverage

| Study | Verified evaluations | Planned evaluations |
|---|---:|---:|
| MSD — four conventional models | 504 | 504 |
| ADNI — four conventional models | 504 | 504 |
| GeoFWI3D — four conventional models | 504 | 504 |
| ADNI — MedSAM3 | 126 | 126 |
| MSD — MedSAM3 extension | 108 | 108 |
| MSD — original MedSAM3 campaign | 0 | 27 |
| GeoFWI3D — SAM3 | 116 | 126 |

**1,862 run evaluations and 124,860 validation-volume records** are available. The extension includes nine matched baseline controls; these are retained for within-campaign comparisons and are not nine additional thesis objectives. The SAM3 count is an archive snapshot, not a live training-status check.

## Recovered files

- [Per-run Dice](background_dice_runs.csv): original foreground Dice, background Dice, and mean Dice including background, with evaluation source and SHA-256.
- [Per-volume Dice and confusion matrices](background_dice_cases.csv): exact class-count evidence used for the calculations.
- [Condition summaries](background_dice_conditions.csv): seed averages within folds, equally weighted fold averages, and SD across fold averages. Incomplete conditions are marked explicitly.
- [Missing evaluations](missing_runs.csv): archive gaps and pending SAM3 evaluations.
- [Verification](verification.json): coverage, hash checks, foreground-score checks, and any failed validations.
- [Reproducible calculation](analyze_recovery.py): reads the saved metadata and rebuilds these CSVs.

All Dice values in the CSVs are fractions, not percentages. `background_dice` is the score for class 0 alone. `dice_including_background` is the unweighted mean across all classes within each volume: `(background + foreground)/2` for binary tasks, or `(background + anterior + posterior)/3` for MSD. These per-volume values are then averaged. Counts are never pooled across volumes to approximate a missing average.

## Provenance and checks

The 323 downloaded MSD ZIPs and 76 SAM3 ZIPs were checked against the SHA-256 in their saved archive identity. MSD coverage was completed using the existing local recovery records and the earlier `unetr_completed_result.zip`. The nine missing GeoFWI3D baseline folders were recovered from their original seed-search and fold-0-confirmation campaigns. Each available evaluation manifest hash was checked, every saved per-class background score was checked against its confusion matrix, and all recovered run foreground averages were reconciled against the final thesis inventory or completed-run receipt.

Download receipts record the Drive file ID, account, canonical source link, byte count, and SHA-256. They omit temporary authenticated download URLs. `remote/` preserves downloaded evidence; `extracted/` contains evaluation metadata extracted from the verified ZIPs. Existing local source files are referenced in place rather than duplicated.

All 1,862 recovered evaluations match their available manifest SHA-256 and reproduce their recorded foreground run score. The [retrieval audit](retrieval_verification.json) independently rechecked 2,370 downloaded files against their saved download receipts, with no mismatches. The [download inventory](downloaded_file_inventory.json) retains their canonical source identities.

## Unresolved historical campaign

The original MSD MedSAM3 baseline, bands, and bands-plus-edge results remain completed foreground results. Their 27 per-volume evaluation records were not located during this recovery. The `existing_results.json` bundle and the October 3 retirement report preserve aggregate scores, which are insufficient to reconstruct the exact background-inclusive average.

The cluster archive `reproducibility_20261003.tar.gz` was recovered and its SHA-256 matches the October 3 preservation receipt (`ecb3ab73e9317e6a86a25338705e9091d893cdb41e3d445fe47f3f8b69152d4f`). It contains source and protocols for that replication, but no original `results/fold*_seed*/results.json` files. The original cluster result directory is absent. Project files, local metadata archives and ZIP inventories were also checked; search receipts are saved alongside this report. This establishes that those records were not found in the searched locations, not that no other backup can exist.

The newer extension baseline is kept separate and is not substituted for the original matched baseline. Missing background Dice is left missing; the existing foreground conclusions are unaffected.
