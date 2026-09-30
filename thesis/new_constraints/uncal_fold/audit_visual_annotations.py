"""Compare frozen provisional image annotations with MSD's label partition.

This is label agreement on a training pilot, not anatomical validation.
Run only after raw-image review and freezing annotations_frozen.json.
"""
import hashlib
import json
from datetime import datetime, timezone

import nibabel as nib
import numpy as np

from .foldedness import best_fit_first_anterior_slice
from .prepare_visual_review import ROOT, OUTPUT


def main():
    annotation_path = OUTPUT / "annotations_frozen.json"
    freeze = json.loads((OUTPUT / "freeze.json").read_text())
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == freeze["annotations_sha256"]
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert hashlib.sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest() == freeze["manifest_sha256"]
    records = json.loads(annotation_path.read_text())["records"]
    assert [r["case"] for r in records] == manifest["cases"]
    dataset = ROOT / "datasets/Dataset101_MSD"
    split_path = dataset / "splits_final.json"
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == manifest["split_sha256"]
    train = set(json.loads(split_path.read_text())[0]["train"])
    results = []
    for record in records:
        name = record["case"]
        assert name in train
        image_path = ROOT / manifest["input_images"][name]["path"]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == manifest["input_images"][name]["sha256"]
        image = nib.load(image_path)
        label = nib.load(dataset / "labelsTr" / f"{name}.nii.gz")
        assert np.allclose(image.affine, label.affine)
        assert np.allclose(label.affine[:3, :3], np.eye(3))
        assert image.shape == label.shape
        c = record["candidate_last_visible_y"]
        lo, hi = record["candidate_interval_y"]
        assert 0 <= lo <= hi < label.shape[1]
        assert c is None or lo <= c <= hi
        target, cost, gap = best_fit_first_anterior_slice(np.asarray(label.dataobj))
        results.append(dict(case=name,visual_candidate_y=c,visual_interval_y=[lo, hi],
                            msd_label_cut_y=target,label_plane_misclassified_voxels=cost,
                            label_second_best_gap=gap,signed_error_mm=None if c is None else c-target,
                            interval_contains_label_cut=None if c is None else lo <= target <= hi))
    accepted = [r for r in results if r["visual_candidate_y"] is not None]
    errors = np.array([r["signed_error_mm"] for r in accepted])
    summary = dict(n_reviewed=len(results),n_candidate=len(accepted),n_abstained=len(results)-len(accepted),
                   mean_absolute_error_mm=float(np.abs(errors).mean()),
                   median_absolute_error_mm=float(np.median(np.abs(errors))),
                   mean_signed_error_mm=float(errors.mean()),within_1mm=int((np.abs(errors)<=1).sum()),
                   within_2mm=int((np.abs(errors)<=2).sum()),
                   intervals_cover_label_cut=sum(r["interval_contains_label_cut"] for r in accepted),
                   mean_interval_width_mm=float(np.mean([r["visual_interval_y"][1]-r["visual_interval_y"][0] for r in accepted])))
    output = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),freeze=freeze,
                  interpretation="Training-pilot label agreement only; not independent anatomical validation or generalization evidence.",
                  summary=summary,cases=results)
    (OUTPUT / "label_comparison.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
