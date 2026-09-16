"""Checks for mask alignment, complete error accounting, and cut detection."""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_fold0_3d import align_audit, label_cut, load_case, mask_stats, rle


def test_padding_crop_preserves_errors_outside_acquired_mri():
    image = np.ones((3, 4, 5), dtype=np.uint8)
    gt = np.zeros_like(image)
    gt[1, 2, 2] = 1
    audit_gt = np.pad(gt, ((1, 2), (1, 1), (0, 1)))
    prediction = audit_gt.copy()
    prediction[0, 0, 0] = 2  # Outside original MRI; must remain an error.
    im, g, p, origin, bounds = align_audit(image, gt, audit_gt, prediction)
    assert origin == [-1, -1, 0]
    assert bounds == [[1, 1, 0], [4, 5, 5]]
    assert im[0, 0, 0] == 0
    assert mask_stats(g, p) == mask_stats(audit_gt, prediction)
    assert mask_stats(g, p)["fp"] == 1


def test_alignment_checks_padding_as_well_as_the_native_crop():
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[1, 1, 1] = 1
    audit_gt = np.pad(gt, 1)
    audit_gt[0, 0, 0] = 1  # A crop-only equality check would miss this.
    with pytest.raises(ValueError, match="does not exactly match"):
        align_audit(gt, gt, audit_gt, audit_gt)


def test_rle_roundtrip_preserves_order_and_long_background_runs():
    array = np.zeros((51, 53, 57), dtype=np.uint8)
    array[4:12, 22:34, 14:29] = 2
    array[8, 25, 20] = 1
    runs = np.asarray(rle(array)).reshape(-1, 2)
    decoded = np.repeat(runs[:, 0], runs[:, 1]).reshape(array.shape)
    assert np.array_equal(decoded, array)


def test_cut_is_not_invented_for_a_mixed_slice():
    gt = np.zeros((3, 6, 3), dtype=np.uint8)
    gt[1, 1:3, 1] = 2
    gt[1, 3:5, 1] = 1
    assert label_cut(gt) == 2.5
    gt[2, 3, 1] = 2
    assert label_cut(gt) is None


def test_directed_error_counts_partition_all_errors():
    gt = np.array([0, 0, 1, 2, 1, 2, 1, 2]).reshape(2, 2, 2)
    pred = np.array([1, 2, 0, 0, 2, 1, 1, 2]).reshape(2, 2, 2)
    stats = mask_stats(gt, pred)
    assert (stats["fp"], stats["fn"], stats["swaps"]) == (2, 2, 2)
    assert stats["total"] == stats["fp"] + stats["fn"] + stats["swaps"] == 6
    assert (stats["a_to_p"], stats["p_to_a"]) == (1, 1)


def test_nifti_prediction_affine_is_verified(tmp_path):
    for folder in ["imagesTr", "labelsTr", "predictions"]:
        (tmp_path / folder).mkdir()
    gt = np.zeros((3, 6, 3), dtype=np.uint8)
    gt[1, 1:3, 1] = 2
    gt[1, 3:5, 1] = 1
    for folder, name in [("imagesTr", "case_0000"), ("labelsTr", "case")]:
        nib.save(nib.Nifti1Image(gt, np.eye(4)), tmp_path / folder / f"{name}.nii.gz")
    affine = np.eye(4)
    affine[0, 3] = 1
    nib.save(nib.Nifti1Image(gt, affine), tmp_path / "predictions/case.nii.gz")
    with pytest.raises(ValueError, match="shape and affine"):
        load_case(tmp_path, "case", tmp_path / "predictions")
    nib.save(nib.Nifti1Image(gt, np.eye(4)), tmp_path / "predictions/case.nii.gz")
    result = load_case(tmp_path, "case", tmp_path / "predictions")
    assert result["stats"]["total"] == 0
    assert result["cut"] == 2.5


def test_ground_truth_only_has_no_synthetic_prediction(tmp_path):
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[1, 1, 1] = 1
    for folder, name in [("imagesTr", "case_0000"), ("labelsTr", "case")]:
        nib.save(nib.Nifti1Image(gt, np.eye(4)), tmp_path / folder / f"{name}.nii.gz")
    result = load_case(tmp_path, "case", None)
    assert result["pred"] is None
    assert "total" not in result["stats"]
    assert result["cut"] is None
