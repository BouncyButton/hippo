"""Audit raw NIfTI A/P geometry without unpickling training data or running a model."""

import argparse
import ast
import gzip
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np


DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def read_label(path):
    raw = gzip.open(path, "rb").read()
    endian = "<" if struct.unpack_from("<i", raw)[0] == 348 else ">"
    if struct.unpack_from(endian + "i", raw)[0] != 348:
        raise ValueError(f"Not a NIfTI-1 file: {path}")
    dim = struct.unpack_from(endian + "8h", raw, 40)
    if dim[0] != 3:
        raise ValueError(f"Expected a 3-D NIfTI label: {path}")
    datatype = struct.unpack_from(endian + "h", raw, 70)[0]
    dtype = np.dtype(endian + DTYPES[datatype])
    offset = int(struct.unpack_from(endian + "f", raw, 108)[0])
    slope, intercept = struct.unpack_from(endian + "2f", raw, 112)
    values = np.frombuffer(raw, dtype=dtype, count=np.prod(dim[1:4]), offset=offset)
    label = values.reshape(dim[1:4], order="F").copy()
    if slope != 0:
        label = label * slope + intercept
    if not np.isin(label, [0, 1, 2]).all():
        raise ValueError(f"Unexpected labels: {path}")
    metadata = {
        "shape": list(dim[1:4]), "datatype": datatype,
        "vox_offset": offset, "scl_slope": slope, "scl_inter": intercept,
        "qform_code": struct.unpack_from(endian + "h", raw, 252)[0],
        "sform_code": struct.unpack_from(endian + "h", raw, 254)[0],
        "sform_rows": np.array(struct.unpack_from(endian + "12f", raw, 280)).reshape(3, 4).tolist(),
        "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return label, raw, dtype, metadata


def direct_scalar(raw, dtype, metadata, coordinate):
    x, y, z = coordinate
    nx, ny, _ = metadata["shape"]
    byte = metadata["vox_offset"] + dtype.itemsize * (x + nx * (y + ny * z))
    raw_value = np.frombuffer(raw[byte:byte + dtype.itemsize], dtype=dtype)[0].item()
    slope, intercept = metadata["scl_slope"], metadata["scl_inter"]
    return {"coordinate": coordinate, "byte_offset": byte, "raw_value": raw_value,
            "scaled_value": raw_value * slope + intercept if slope else raw_value}


def reference_reader(path):
    """Extract only the pre-existing parser; do not import its audit main program."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "read_nifti"
    ) or (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_DT" for t in node.targets)
    )]
    scope = {"np": np, "gzip": gzip, "struct": struct, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope["read_nifti"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/Dataset101_MSD"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gt_geometry_audit.json"))
    parser.add_argument("--reference-reader", type=Path)
    parser.add_argument("--export-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    split_path = args.dataset / "splits_final.json"
    split = json.loads(split_path.read_text())[args.fold]
    previous_reader = reference_reader(args.reference_reader) if args.reference_reader else None
    report = {
        "data_source": "raw labelsTr NIfTI-1, original stored voxel axes",
        "training_pickle_verified": False,
        "scope_caveat": "This does not establish equality to training pickle arrays or prior interface-mask analysis.",
        "dataset_path": str(args.dataset.resolve()), "fold": args.fold,
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "axis": 1, "anterior_side": "high", "cases": [], "summary": {},
        "reference_reader": str(args.reference_reader) if args.reference_reader else None,
        "export_verification": {str(path): {"available": 0, "matched": 0, "nonplanar": 0}
                                for path in args.export_dir},
    }
    if args.reference_reader:
        report["reference_reader_sha256"] = hashlib.sha256(args.reference_reader.read_bytes()).hexdigest()
    for cohort in ("train", "val"):
        summary = Counter()
        for name in split[cohort]:
            path = args.dataset / "labelsTr" / f"{name}.nii.gz"
            label, raw, dtype, metadata = read_label(path)
            n1 = (label == 1).sum(axis=(0, 2))
            n2 = (label == 2).sum(axis=(0, 2))
            mixed = np.flatnonzero((n1 > 0) & (n2 > 0))
            s1, s2 = np.flatnonzero(n1 > 0), np.flatnonzero(n2 > 0)
            orientation = len(s1) > 0 and len(s2) > 0 and s2[-1] < s1[0]
            unique_cut = orientation and s1[0] - s2[-1] == 1
            planar_errors = n1.cumsum()[:-1] + n2[::-1].cumsum()[::-1][1:]
            examples = []
            for y in mixed:
                coordinates = []
                for class_id in (1, 2):
                    x, z = np.argwhere(label[:, y, :] == class_id)[0]
                    check = direct_scalar(raw, dtype, metadata, [int(x), int(y), int(z)])
                    if check["scaled_value"] != class_id:
                        raise AssertionError("Direct byte-offset lookup disagrees with array parsing.")
                    coordinates.append(check)
                examples.append({"slice": int(y), "anterior_voxels": int(n1[y]),
                                 "posterior_voxels": int(n2[y]), "direct_byte_checks": coordinates})
            reference_match = None
            if previous_reader is not None:
                reference_match = bool(np.array_equal(previous_reader(path), label))
                if not reference_match:
                    raise AssertionError(f"Previous parser disagrees for {name}.")
            row = {
                "cohort": cohort, "case": name, "metadata": metadata,
                "exact_planar_unique_cut": bool(unique_cut),
                "gt_cut_last_low_slice": int(s2[-1]) if unique_cut else None,
                "mixed_slices": examples, "minority_voxels": int(np.minimum(n1, n2).sum()),
                "best_planar_disagreements": int(planar_errors.min()),
                "foreground_voxels": int((label != 0).sum()),
                "reference_reader_equal": reference_match,
                "export_ground_truth": {},
            }
            for directory in args.export_dir:
                exported_path = directory / f"{name}.npz"
                if not exported_path.exists():
                    continue
                with np.load(exported_path) as archive:
                    exported = archive["ground_truth"]
                padding = [(max(0, 64 - s) // 2, max(0, 64 - s) - max(0, 64 - s) // 2)
                           for s in label.shape]
                padded = np.pad(label, padding)
                crop = tuple(slice((s - 64) // 2, (s - 64) // 2 + 64) if s > 64 else slice(None)
                             for s in padded.shape)
                padded = padded[crop]
                matches = bool(np.array_equal(exported, padded))
                ex1 = (exported == 1).sum(axis=(0, 2))
                ex2 = (exported == 2).sum(axis=(0, 2))
                export_nonplanar = bool(((ex1 > 0) & (ex2 > 0)).any())
                row["export_ground_truth"][str(directory)] = {
                    "equal_after_symmetric_pad_crop_64": matches,
                    "nonplanar": export_nonplanar,
                    "sha256": hashlib.sha256(exported_path.read_bytes()).hexdigest(),
                }
                export_summary = report["export_verification"][str(directory)]
                export_summary["available"] += 1
                export_summary["matched"] += int(matches)
                export_summary["nonplanar"] += int(export_nonplanar)
            report["cases"].append(row)
            summary["case_count"] += 1
            summary["exact_planar_unique_cut"] += int(unique_cut)
            summary["invalid_cases"] += int(not unique_cut)
            summary["mixed_slices"] += len(mixed)
            summary["minority_voxels"] += row["minority_voxels"]
            summary["best_planar_disagreements"] += row["best_planar_disagreements"]
            summary["foreground_voxels"] += row["foreground_voxels"]
        report["summary"][cohort] = dict(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": report["summary"],
                      "export_verification": report["export_verification"]}, indent=2))


if __name__ == "__main__":
    main()
