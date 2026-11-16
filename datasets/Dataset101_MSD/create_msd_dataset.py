"""Convert the extracted MSD Task04 paired training volumes to the study layout."""
import argparse
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pandas as pd


def prepare(source, target):
    rows = []
    for folder in ("imagesTr", "imagesTs", "labelsTr"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    for image in sorted((source / "imagesTr").glob("hippocampus_*.nii.gz")):
        case = image.name.removesuffix(".nii.gz").removesuffix("_0000")
        label = source / "labelsTr" / (case + ".nii.gz")
        source_image, mask = nib.load(image), nib.load(label)
        if source_image.shape != mask.shape or not np.allclose(source_image.affine, mask.affine):
            raise ValueError(f"Image and label grids differ for {case}")
        rows.append(dict(subject_id=case, image_data=source_image.get_fdata(),
                         label_data=mask.get_fdata(), dims=source_image.shape))
        shutil.copyfile(image, target / "imagesTr" / (case + "_0000.nii.gz"))
        shutil.copyfile(label, target / "labelsTr" / label.name)
    if not rows:
        raise ValueError("No paired MSD training volumes found")
    for image in sorted((source / "imagesTs").glob("hippocampus_*.nii.gz")):
        case = image.name.removesuffix(".nii.gz").removesuffix("_0000")
        shutil.copyfile(image, target / "imagesTs" / (case + "_0000.nii.gz"))
    pd.DataFrame(rows).to_pickle(target / "msd_hippocampus_full.pkl", compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted Task04_Hippocampus directory")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.target)
