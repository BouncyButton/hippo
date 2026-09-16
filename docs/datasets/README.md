# Datasets

This folder contains scripts and metadata for building dataset folders.

The expected dataset layout is:

```text
DatasetXXX_NAME/
  dataset.json
  imagesTr/
    case_0000.nii.gz
  labelsTr/
    case.nii.gz
```

Raw image files are not included in this repository.

## Dataset Builders

| Folder | Script | Notes |
| --- | --- | --- |
| `Dataset101_MSD/` | `create_msd_dataset.py` | Uses W&B if available, otherwise can rebuild from the public MSD hippocampus tar file. |
| `Dataset102_MNI/` | `create_mni_dataset.py` | Uses W&B if available, otherwise can rebuild from the MNI HiSub25 download. |
| `Dataset103_ADNI/` | `create_adni_dataset.py` | Uses W&B if available, otherwise downloads the ADNI hippocampus protocol release files used by the script. |
| `Dataset105_COBRA/` | `create_cobra_dataset.py` | Uses W&B if available, otherwise rebuilds from COBRA resources and the atlas repository. |
| `Dataset104_HFH/` | `open_tle_dataset.py` | Helper for inspecting the HFH/TLE dataset. Data is not included. |

## Build A Dataset

From the repository root:

```bash
python datasets/Dataset102_MNI/create_mni_dataset.py --target datasets/Dataset102_MNI
```

Force a rebuild instead of using a W&B artifact:

```bash
python datasets/Dataset102_MNI/create_mni_dataset.py --target datasets/Dataset102_MNI --rebuild
```

## Create Splits

The baseline launch scripts generate splits automatically. To run split creation
by hand:

```bash
python datasets/generate_holdout_cv_splits.py \
  --dataset-dir datasets/Dataset102_MNI \
  --split-scheme nested \
  --outer-fold 0 \
  --outer-n-folds 5 \
  --n-folds 5
```

This writes a train/test split and cross-validation splits inside the dataset
folder.

## Helper Scripts

`fix_unetrpp_*` scripts convert or repair split files for UNETR++.

`create_unetrpp_dataset_json.py` writes the explicit `dataset.json` file that
UNETR++ expects.

