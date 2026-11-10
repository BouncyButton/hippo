"""Prepare salt-containing archive-0 volumes and fixed nested low-data splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tarfile

import nibabel as nib
import numpy as np

from download_geofwi3d import ARCHIVE_BYTES, ARCHIVE_MD5, digest
from geofwi3d_data import (SHAPE, archive_metadata, make_splits, normalized_image,
                          salt_mask, sha256, validate_splits, write_json)


def prepare(root: Path, split_seed: int = 0) -> dict:
    receipt = json.loads((root / "download_receipt.json").read_text())
    archive = root / "downloads/models_batch_0.tar.gz"
    metadata = root / "downloads/parameters.csv"
    if archive.stat().st_size != ARCHIVE_BYTES or digest(archive, "md5") != ARCHIVE_MD5:
        raise RuntimeError("Archive does not match the published checksum")
    if sha256(metadata) != receipt["metadata_sha256"]:
        raise RuntimeError("Metadata changed after acquisition")
    rows = archive_metadata(metadata)
    eligible = {f"model_{int(row['sample_index']):04d}": row for row in rows if int(float(row["yn_salt"]))}
    if len(eligible) != 479:
        raise RuntimeError("Archive-0 cohort differs from the verified 479 salt-containing scenes")
    splits = make_splits(rows, split_seed)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["splits"] != splits:
            raise RuntimeError("Refusing to change an already prepared dataset's split")
        for case in existing["cases"].values():
            for field in ["image", "label"]:
                if sha256(root / case[field]) != case[field + "_sha256"]:
                    raise RuntimeError(f"Prepared file changed: {case[field]}")
        return existing
    cases, pending, seen_members = {}, {}, set()
    train = set(splits["train_pool"])
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            match = re.fullmatch(r"(?:\./)?(?:allmodels/)?(model_\d{4})/(image3d|rgt3d|vp3d)\.bin", member.name)
            if not match or match[1] not in eligible:
                continue
            name, modality = match.groups()
            if (name, modality) in seen_members or not member.isfile() or member.size != np.prod(SHAPE) * 4:
                raise RuntimeError(f"Unexpected or duplicate archive member: {member.name}")
            seen_members.add((name, modality))
            payload = source.extractfile(member).read()
            pending.setdefault(name, {})[modality] = np.frombuffer(payload, dtype="<f4").reshape(SHAPE)
            if len(pending) > 8:
                raise RuntimeError("Archive order would require excessive buffering")
            if len(pending[name]) != 3:
                continue
            arrays, row = pending.pop(name), eligible[name]
            label, audit = salt_mask(arrays["rgt3d"], arrays["vp3d"], float(row["salt_vel"]))
            image, scale = normalized_image(arrays["image3d"])
            image_dir, label_dir = ("imagesTr", "labelsTr") if name in train else ("imagesVal", "labelsVal")
            image_path, label_path = root / image_dir / f"{name}_0000.nii.gz", root / label_dir / f"{name}.nii.gz"
            for path, array in [(image_path, image), (label_path, label)]:
                path.parent.mkdir(parents=True, exist_ok=True)
                volume = nib.Nifti1Image(array, np.eye(4))
                volume.header.set_xyzt_units("unknown")  # source has no physical voxel-spacing declaration
                nib.save(volume, path)
            cases[name] = {"image": str(image_path.relative_to(root)), "label": str(label_path.relative_to(root)),
                           "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
                           "seed": row["seed"], "fault_present": int(float(row["yn_fault"])),
                           "generated_salt_body_count": int(float(row["nsalt"])),
                           "normalization_abs_percentile_99_5": scale, "shape": list(SHAPE), **audit}
            if len(cases) % 25 == 0:
                write_json(root / "preparation_status.json", {"state": "preparing", "completed_cases": len(cases), "expected_cases": 479})
                print(json.dumps({"prepared_cases": len(cases), "expected_cases": 479}), flush=True)
    if pending or set(cases) != set(eligible):
        raise RuntimeError("Incomplete native image/label preparation")
    validate_splits(splits, cases)
    manifest = {"dataset": "GeoFWI3D", "archive_index": 0, "version": 1,
                "native_shape": list(SHAPE), "axes": "X,Y,Z; Z is depth", "spatial_units": "voxels",
                "target": "all salt material inside the observation window; other material is background",
                "label_rule": "(rgt3d == 0) AND (vp3d == float32(metadata.salt_vel))",
                "label_audit": {"removed_null_rgt_background_voxels": sum(c["removed_null_rgt_background_voxels"] for c in cases.values()),
                                "tiny_physical_salt_components_retained": True},
                "normalization": "native image-only 99.5th absolute-amplitude percentile, clipped to [-1,1]",
                "archive_md5": ARCHIVE_MD5, "archive_sha256": receipt["sha256"],
                "metadata_sha256": receipt["metadata_sha256"], "splits": splits, "cases": dict(sorted(cases.items()))}
    write_json(manifest_path, manifest)
    write_json(root / "splits.json", splits)
    write_json(root / "dataset.json", {"name": "GeoFWI3D_Salt_Archive0", "channel_names": {"0": "seismic reflectivity"},
               "labels": {"background": 0, "salt": 1}, "numTraining": len(train), "numValidation": len(splits["validation"]),
               "file_ending": ".nii.gz", "description": "Synthetic nonbiological 3D full-object segmentation; no test split"})
    write_json(root / "preparation_status.json", {"state": "complete", "case_count": len(cases),
               "training_pool": len(train), "validation": len(splits["validation"]),
               "subsets": {k: len(v) for k, v in splits["training_subsets"].items()}, "label_audit": manifest["label_audit"]})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/Dataset203_GeoFWI3D"))
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.root.resolve(), args.split_seed)
    print(json.dumps({"cases": len(result["cases"]), "validation": len(result["splits"]["validation"]),
                      "subsets": {k: len(v) for k, v in result["splits"]["training_subsets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
