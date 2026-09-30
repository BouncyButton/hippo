"""Bind calibrated geometry and shared translation views to their run settings."""

import sys
from types import SimpleNamespace

import pytest
import torch

from thesis.new_constraints import train_swinunetr_constraints as trainer


@pytest.fixture
def plane_report(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"run": {"spatial_size": (64, 64, 64), "resize": False}}, checkpoint_path)
    args = SimpleNamespace(
        constraint_set="ap_plane", dataset="MSD", fold=0,
        ap_plane_axis=1, ap_plane_anterior_side="high", ap_plane_margin=0.0,
        amp=False, ap_plane_weight=0.1, spatial_size=(64, 64, 64), resize=False,
        pkl=tmp_path / "data.pkl.gz", splits_json=tmp_path / "splits.json",
    )
    report = {
        "status": "complete", "constraint_set": "ap_plane", "dataset": "MSD", "fold": 0,
        "ap_plane": {"axis": 1, "anterior_side": "high", "margin": 0.0, "require_both": True},
        "amp": False, "recommended_ap_plane_weight": 0.1,
        "target_ratio": 0.1, "safety_ratio": 0.5,
        "seed": 0, "max_cases": 1, "training_case_ids": ["case"],
        "cases": [{"case_name": "case", "valid": True, "finite_positive_gradients": True,
                   "dice_gradient_rms": 1.0, "plane_gradient_rms": 1.0}],
        "valid_case_count": 1, "skipped_case_count": 0,
        "dice_gradient_rms": trainer._calibration_summary([1.0]),
        "plane_gradient_rms": trainer._calibration_summary([1.0]),
        "target_weight": 0.1, "cap_weight": 0.5,
        "source_provenance": {}, "runtime_provenance": {}, "execution_provenance": {},
        "pkl": str(args.pkl), "pkl_sha256": "a" * 64,
        "splits_json": str(args.splits_json), "splits_json_sha256": "b" * 64,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": trainer.file_sha256(checkpoint_path),
    }
    monkeypatch.setattr(trainer, "_load_splits_json", lambda path: [{"train": ["case"]}])
    kwargs = dict(
        report_snapshot=SimpleNamespace(sha256="c" * 64),
        pkl_digest="a" * 64, splits_digest="b" * 64,
        source_provenance={}, runtime_provenance={}, execution_provenance={},
    )
    return report, args, kwargs


@pytest.mark.parametrize("legacy", [False, True])
def test_plane_calibration_accepts_matching_preprocessing(plane_report, legacy):
    report, args, kwargs = plane_report
    if not legacy:
        report.update(spatial_size=[64, 64, 64], resize=False)
    trainer.validate_ap_plane_calibration_report(report, args, **kwargs)


@pytest.mark.parametrize("field,value", [("spatial_size", (96, 96, 96)), ("resize", True)])
@pytest.mark.parametrize("legacy", [False, True])
def test_plane_calibration_rejects_changed_preprocessing(plane_report, field, value, legacy):
    report, args, kwargs = plane_report
    if not legacy:
        report.update(spatial_size=[64, 64, 64], resize=False)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        trainer.validate_ap_plane_calibration_report(report, args, **kwargs)


@pytest.mark.parametrize("field,value", [("spatial_size", [96, 96, 96]), ("resize", True)])
def test_plane_report_preprocessing_must_match_verified_checkpoint(plane_report, field, value):
    report, args, kwargs = plane_report
    report[field] = value
    with pytest.raises(ValueError, match=field):
        trainer.validate_ap_plane_calibration_report(report, args, **kwargs)


def test_augmented_run_rejects_identity_calibration(plane_report):
    report, args, kwargs = plane_report
    args.training_augmentation = 'mild_v1'
    with pytest.raises(ValueError, match='training_augmentation'):
        trainer.validate_ap_plane_calibration_report(report, args, **kwargs)


def test_augmented_calibration_binds_policy_and_rng(plane_report):
    report, args, kwargs = plane_report
    args.training_augmentation = 'mild_v1'
    report.update(training_augmentation='mild_v1', augmentation_policy=dict(trainer.MILD_V1),
                  augmentation_seed=1)
    trainer.validate_ap_plane_calibration_report(report, args, **kwargs)
    report['augmentation_seed'] = 2
    with pytest.raises(ValueError, match='policy/RNG'):
        trainer.validate_ap_plane_calibration_report(report, args, **kwargs)


@pytest.mark.parametrize("preset", ["equivariance", "translation"])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_shared_translation_view_requires_matching_shift(monkeypatch, tmp_path, preset, size):
    monkeypatch.setattr(sys, "argv", [
        "train", "--pkl", str(tmp_path / "data.pkl.gz"),
        "--splits-json", str(tmp_path / "splits.json"),
        "--output-dir", str(tmp_path / "run"), "--constraint-set", preset,
        "--translation-augmentation", "--translation-size", str(size),
    ])
    args = trainer.parse_args()
    if size == 2:
        assert trainer.resolve_constraint_config(args).translation_size == 2
    else:
        with pytest.raises(ValueError, match="requires --translation-size 2"):
            trainer.resolve_constraint_config(args)
