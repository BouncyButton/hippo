"""Build the frozen HFH or MNI crop arrays from the local study data."""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from portable import PROTOCOLS, check_arrays, read, verify


def crop(image, label, spacing, value):
    points = np.argwhere(label == value)
    lower, upper = points.min(0), points.max(0)
    centre = (lower+upper)/2
    shape = np.array([64, 32, 64])
    target_spacing = np.array([1., 2., 1.])
    scale = target_spacing/spacing
    # Align slice indices to integer source slices: no interpolation through 2-mm gaps.
    origin = centre-(shape-1)*scale/2
    origin[1] = np.floor(origin[1])
    if np.any(lower < origin) or np.any(upper > origin+(shape-1)*scale):
        raise ValueError('Foreground does not fit predeclared 64 mm field of view')
    x = affine_transform(image.astype(np.float32), np.diag(scale), offset=origin,
        output_shape=tuple(shape), order=1, mode='constant', cval=0, prefilter=False)
    y = affine_transform((label == value).astype(np.uint8), np.diag(scale), offset=origin,
        output_shape=tuple(shape), order=0, mode='constant', cval=0, prefilter=False)
    if not np.isfinite(x).all() or y.sum() == 0:
        raise ValueError('Invalid resampled crop')
    return x, y, dict(source_origin_voxels=origin.tolist(), output_to_source_scale=scale.tolist(),
        source_label=value, source_bbox=[lower.tolist(), upper.tolist()], spacing_mm=target_spacing.tolist(),
        shape=shape.tolist(), native_foreground_voxels=len(points), resampled_foreground_voxels=int(y.sum()))


def prepare_hfh(source, output, inventory):
    import nibabel as nib
    for case in sorted({name.rsplit("_", 1)[0] for name in inventory}):
        index = case.rsplit("_", 1)[1]
        image = nib.load(source / f"HFH_{index}.hdr")
        label = nib.load(source / "Labels" / f"HFH_{index}_Hipp_Labels.hdr")
        x, y = np.asarray(image.dataobj).squeeze(), np.asarray(label.dataobj).squeeze()
        if x.shape != y.shape or not np.allclose(image.affine, label.affine):
            raise ValueError(f"Image and label geometry differ: {case}")
        for side in (1, 2):
            xx, yy, _ = crop(x, y, np.array(image.header.get_zooms()[:3]), side)
            np.savez_compressed(output / f"{case}_h{side}.npz", image=xx, label=yy)


def prepare_mni(source, output, inventory):
    import pandas as pd
    verify(source, read(PROTOCOLS / "medsam3_mni.json")["common"]["dataset_sha256"])
    seen = set()
    for row in pd.read_pickle(source, compression="gzip").itertuples():
        name = f"hippocampus_mni_{row.subject_id}_{row.direction}"
        if name in seen or name not in inventory:
            raise ValueError(f"Duplicate or unexpected MNI crop: {name}")
        seen.add(name)
        np.savez_compressed(output / (name + ".npz"), image=np.asarray(row.image_data),
                            label=np.asarray(row.data, dtype=np.uint8))
    if seen != set(inventory):
        raise ValueError("Incomplete MNI crop set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HFH", "MNI"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = read(PROTOCOLS / (args.dataset.lower() + "_files.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    (prepare_hfh if args.dataset == "HFH" else prepare_mni)(args.source, args.output, inventory)
    check_arrays(args.output, inventory, inventory)
    print(f"Verified {len(inventory)} reconstructed {args.dataset} crops")


if __name__ == "__main__":
    main()
