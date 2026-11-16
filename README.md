# Logic-guided spatial constraints for hippocampus segmentation

This release contains training code and final-checkpoint validation results for MSD, MNI-HISUB25, ADNI and HFH. The models are 3D U-Net, nnU-Net, UNETR, SwinUNETR and MedSAM3. Each model–dataset combination has a baseline and 13 spatial objectives, evaluated on three partitions with three seeds each.

## Reproduce the reported results

From the repository root, use Python 3.11 or newer. The reporting commands use the standard library and the included per-run scores.

```sh
python evaluation/constraint_gap.py --output outputs/tables
python evaluation/paper_tables.py --output outputs/tables/paper.csv
```

`objective_pairs.csv` contains every paired result, including decreases and ties. `constraint_gap.md` summarizes all 280 objectives by model and dataset. `paper.csv` reconstructs the selected rows of manuscript Tables 2 and 3 at full precision. Dice is expressed in percent and gains in percentage points. The SD is the sample SD of three seed-averaged fold means. The CSV retains the baseline from the same execution batch for both original MSD MedSAM3 objectives.

The auxiliary full-data columns correspond to three different training protocols. Empty cells identify comparisons for which a matching score is unavailable. The archival input filenames and original SHA-256 digests are in [evaluation/data/provenance.json](evaluation/data/provenance.json).

## Run an experiment

The model definitions and workers are in `baselines/`, `scripts/` and `thesis/new_constraints/`. Data preparation is under `datasets/`. HFH scratch and all four MedSAM3 workflows run directly from this checkout with the study data and, for MedSAM3, the released weights. The other scratch workflows use the linked frozen protocol bundles. The release includes no patient volumes or model checkpoints. Instructions, dependencies and commands for each dataset and model family are in [the workflow guide](workflows/README.md).

## Validation design

The paper tables select the objective with the highest mean on the same validation folds used to report its gain. Earlier validation outcomes also informed the study design. In the scratch experiments, MSD seed 20 replaced seed 1, and ADNI seed 4 replaced seed 2 after observing performance. MNI used revised training recipes and a larger compute budget. The MNI MedSAM3 third-seed procedure used a baseline threshold and a positive pilot-gain criterion before completing the matrix. On HFH, recipe selection included objective-specific settings. For nnU-Net, timing and coefficient multipliers were chosen by objective and seed using validation across the three folds.

For the reported HFH nnU-Net EDGE+DEPTH objective, the gain at the selected timing was 1.37 points with the unit multiplier across all seeds and 1.70 points with the selected multipliers. Both estimates use the development folds. The per-seed values are in [the weight comparison](evaluation/data/hfh_nnunet_weight_comparison.csv).

The tables describe the recorded validation cohorts. A claim about the expected gain on unseen subjects would require an independent evaluation of the fixed procedure. MSD uses crop identifiers for the split. Subject-level separation on MSD cannot be verified from those identifiers. The complete development history is retained on `Filippo_dev` at `a003d5323df99dfdd10f666d60e41dafd93f69ac`.

## Checks

```sh
python -m pytest -q evaluation workflows/test_portable.py
python workflows/smoke_objectives.py MSD
python workflows/smoke_objectives.py MNI
python workflows/smoke_objectives.py binary
```

The reporting tests compare the reconstructed gains against all 2,520 archived paired differences. The portable-workflow tests check every released cell, HFH update counts and activation times against the score tables, and the two distinct MSD adapter controls. The CPU smoke checks exercise all four spatial terms and their gradients on synthetic labels. Run each smoke command in its own process because the archived dataset workers install dataset-specific losses in module state.
