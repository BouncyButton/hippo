# Thesis background-Dice integration

Updated the existing `Template_tesi/thesis-revision.tex` in place. The new Results subsection reports foreground, background and all-class Dice for matched baselines and the selected conventional/MedSAM3 configurations. A separately labelled SAM3 snapshot includes only six complete conditions. The discussion checks all 180 complete objective comparisons and retains the missing original MSD MedSAM3 background evidence explicitly.

- `integrate.py`: the source integration and numerical assertions (one-time application; refuses duplicate insertion).
- `table_values.csv`: full-precision values used in both tables.
- `verification.json`: input/source hashes, checked claims and successful built-in compiler result.
- `change.patch`: the complete source diff.

Source evidence: `../background_dice_archive_recovery_20261023/`. No experiment was rerun. The conclusions chapter contains no repeated discussion of these results; its only adjustment updates the status of recovered geological baseline records.
