"""Prepare aligned image and label tensors from the archived dataframe."""
from monai.transforms import Compose, EnsureChannelFirstd, NormalizeIntensityd, Resized, EnsureTyped, Lambdad, \
    DivisiblePadd, SpatialPadd, CenterSpatialCropd

from monai.data import DataLoader, Dataset as MonaiDataset

import torch

import pandas as pd

def _case_name_from_row(row, dataset):
    dataset = dataset.upper()
    if dataset == "MSD":
        subject_id = str(row["subject_id"])
        return f"{subject_id}"
    if dataset == "MNI":
        subject_id = str(row["subject_id"])
        direction = str(row["direction"])
        return f"hippocampus_mni_{subject_id}_{direction}"
    if dataset == "ADNI":
        image_id = str(row["image_id"])
        direction = str(row["direction"])
        return f"hippocampus_adni_{image_id}_{direction}"
    if dataset == "COBRA":
        subject_id = str(row["subject_id"])
        side = str(row["side"])
        return f"hippocampus_cobra_{subject_id}_{side}"
    raise ValueError(f"Unknown dataset: {dataset}")

def _load_pkl_dataframe(pkl_path):
    df = pd.read_pickle(pkl_path, compression="gzip")
    if df.empty:
        raise ValueError(f"Empty dataframe in {pkl_path}")
    return df

def _build_monai_dataset_from_pkl(df, dataset, num_classes, spatial_size=(64, 64, 64), do_resize=False):
    dataset = dataset.upper()
    if dataset in {"MSD", "COBRA"}:
        image_key = "image_data"
        label_key = "label_data"
    elif dataset in {"MNI", "ADNI"}:
        image_key = "image_data"
        label_key = "data"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    records = []
    for _, row in df.iterrows():
        records.append({
            "case_name": _case_name_from_row(row, dataset),
            "image": row[image_key],
            "label": row[label_key],
        })

    xforms = [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # nnU-Net default for non-CT modalities is per-case z-score
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        DivisiblePadd(keys=["image", "label"], k=32, mode=("constant", "constant")),
        Lambdad(
            keys=["label"],
            func=lambda x: torch.clamp(torch.round(torch.as_tensor(x)), 0, num_classes - 1),
        ),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
    ]
    if do_resize:
        xforms.insert(
            2,
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        )
    else:
        # ensure fixed size for batching when resize is disabled
        xforms.insert(2, SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, mode=("constant", "constant")))
        xforms.insert(3, CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size))
    transforms = Compose(xforms)

    return MonaiDataset(data=records, transform=transforms)
