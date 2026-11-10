# No-augmentation thesis results — 21 October 2026

The thesis presents 1,260 planned objective runs in 140 conditions:

| Campaign | Runs | Conditions |
|---|---:|---:|
| MSD, four models, FLOP cap | 504 | 56 |
| ADNI, four models, FLOP cap | 504 | 56 |
| MSD MedSAM3, original cluster study | 27 | 3 |
| MSD MedSAM3, Colab extension (additional objectives) | 99 | 11 |
| ADNI MedSAM3, two support crops | 126 | 14 |

All conditions contain three folds and three seeds. The repeated Colab baseline controls are supporting records, excluded from the headline objective inventory. MSD MedSAM3 covers all 14 planned objectives across two campaigns; each campaign retains its own matched baseline. Original baseline/B/B+E: 74.93/78.76/80.12 percent A/P Dice, with gains +3.83/+5.19 percentage points. These objectives are complete, not missing.

`audit.py` regenerates `results.json`, `audit.json`, `planned_1260_runs.csv`, `planned_140_condition_summaries.csv`, and complete audit ledgers `all_1269_runs.csv` and `all_141_condition_summaries.csv` from the archived sources in this repository. Run it from the repository root with `python3 reports/thesis_noaug_update_20261021/audit.py`. The study key separates the two MSD adaptation baselines. Means give equal weight to seeds within a fold and folds; dispersion is sample standard deviation over three fold means. Conventional error counts pool validation voxels per run; MedSAM3 error counts are per-crop means.

`audit.json` records source hashes and the scope of verification. The conventional seed replacements and adaptation fold/support selection are disclosed as exploratory limitations. The original 27 MedSAM3 rows are reconciled with preserved passed audits; deleted tensors are not reloaded. All 126 ADNI MedSAM3 archives are checked against their metadata hashes and per-case confusion matrices, beyond the stale 109-row snapshot.

The manuscript is edited in place at `Template_tesi/thesis-revision.tex`. Augmented and superseded historical performance tables are excluded from this revision; their durable source archives are retained. SAM3 fine-tuning and the final nonmedical three-fold, three-seed replication await integration. Dataset reference examples (including ADNI), annotation audit, table of contents and bibliography remain.
