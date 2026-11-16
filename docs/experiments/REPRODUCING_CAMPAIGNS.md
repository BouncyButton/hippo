# Reproducing the hippocampus experiments

Run the commands below from the repository root, using the Python environment
described in the [README](../../README.md). The HFH converter uses Python 3.11
or later. A checkout is sufficient for the CPU checks. Training requires the
appropriate dataset and model weights; exact replay of a historical run also
requires its frozen input and protocol bundles.

| Question | Start here | Inputs and interpretation |
| --- | --- | --- |
| How was HFH converted and split? | [HFH dataset guide](../../datasets/Dataset104_HFH/README.md) | Extract the official release locally. The converter verifies the native grid and writes the inventory. The five validation folds use only labeled subjects. |
| How were the low-data HFH comparisons run? | [Preparation](../../scripts/colab_hfh/prepare.py), [recipe selection](../../scripts/colab_hfh/constraint_search.py), [MedSAM3 policy](../../scripts/colab_hfh/medsam_constraints_policy.py) | These experiments use crops localized with reference labels. Their Dice measures segmentation on those crops, not detection in a full scan. Preparation also reads earlier ADNI and MSD bundles. |
| When does a third MNI seed advance? | [Seed search](../../scripts/mni_medsam3_cluster/seed_search.py), [paired pilot decision](../../scripts/mni_medsam3_cluster/constraint_campaign.py) | Start from the frozen two-seed matrix. The baseline must reach 0.60 Dice in each fold. The remaining arms run only if a pilot arm has positive mean paired validation gain across all three folds. The selected seed is exploratory. |
| How were onset and strength varied for MNI nnU-Net? | [Timing pilot builder](../../scripts/colab_mni_early/build.py) | Reuses the frozen MNI recipe, training budget and split. The initial pilot is fold 0, seed 0. |
| How were larger training sets and learning rates compared? | [Full-fold manifest builder](../../reports/full_data_baselines_20261115/prepare_manifest.py), [matched controls](../../reports/full_data_baselines_20261115/build_matched_controls.py), [learning-rate protocol](../../reports/full_data_lr_20261116/README.md) | Rebuilding the comparison needs the parent splits, per-update traces, and result archives. Compare learning rates within the same data regime; the low-data contrast also changes training-set size. |
| How were spatial errors measured? | [Metric definitions](../../reports/paper_constraint_motivation/spatial_audit/spatial_metrics.py), [audit runner](../../reports/paper_constraint_motivation/spatial_audit/run.py) | The audit consumes evaluation inputs from the corresponding runs. Its boundary-error measures complement Dice. |

## Check the code without training

```bash
python -m pytest -q \
  scripts/colab_hfh/test_exploration.py::SelectionTests \
  scripts/colab_hfh/test_exploration.py::ArchiveTests \
  scripts/colab_hfh/test_exploration.py::GeometryTests::test_notebook_python_cells_parse \
  scripts/colab_hfh/test_constraint_drive.py \
  scripts/colab_hfh/test_medsam_drive.py \
  scripts/mni_medsam3_cluster/test_seed_search.py
```

The Drive checks use mocks. Other campaign tests verify real manifests, packages,
and archived controls under `reports/`; those checks require the corresponding
local archive. Generated notebooks, prediction volumes, training checkpoints,
and account credentials are not inputs to the CPU command above.

## Run on your own infrastructure

Supply `HIPPO_STORAGE_EMAIL` and `HIPPO_HFH_DRIVE_ROOT` in the HFH runtime
before importing its Drive helpers. Set `HIPPO_COMPUTE_EMAIL` for launchers
that verify the notebook account. The legacy HFH bootstrap also uses
`HIPPO_RCLONE_FILE_ID` for the pinned binary and `HIPPO_LEGACY_DRIVE_ROOT`
to check earlier results. Full-fold Colab workers use
`HIPPO_FULL_DATA_DRIVE_ROOT` instead of the HFH folder. These variables identify
accounts and folders; authenticate separately in the runtime.

Set `HIPPO_CLUSTER_HOST` to your SSH alias and `HIPPO_CLUSTER_ROOT` to your
experiment directory on the cluster. Collectors that use a remote home or
Python environment accept `HIPPO_REMOTE_HOME` and `HIPPO_REMOTE_PYTHON`.
The Slurm specifications retain the original GPU resource requests. Adapt them and the corresponding
campaign configuration before submitting on another cluster. Keep the original
manifests for verification; use a separate output directory for a new run.
