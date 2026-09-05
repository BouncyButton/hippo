"""CPU integration coverage for the Dice+CE and experimental constraint routes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.new_constraints import train_swinunetr_constraints as trainer
from thesis.new_constraints.bands.calibrate_weight import validate_checkpoint_provenance
from thesis.new_constraints.objective import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.supervised import CalibrationDiagnostics


def _parse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> argparse.Namespace:
    monkeypatch.setattr(sys, "argv", [
        "train_swinunetr_constraints.py",
        "--pkl", str(tmp_path / "dataset.pkl.gz"),
        "--splits-json", str(tmp_path / "splits.json"),
        "--output-dir", str(tmp_path / "run"),
        *extra,
    ])
    return trainer.parse_args()


def test_parser_keeps_dice_default_and_opt_in_diagnostics(monkeypatch, tmp_path) -> None:
    args = _parse(monkeypatch, tmp_path)
    assert args.supervised_loss == "dice"
    assert args.ce_weight == 1.0
    assert args.calibration_diagnostics is False
    assert args.constraint_set == "equivariance"
    assert args.teacher_views == 2
    assert args.teacher_weight is None and args.ap_cut_weight is None
    assert args.ap_axis is None and args.ap_anterior_side is None


@pytest.mark.parametrize("preset, extra", [
    ("teacher", ("--teacher-weight", "0.1", "--ap-cut-weight", "0.1")),
    ("ap_cut", ("--ap-cut-weight", "0.1", "--teacher-weight", "0.1")),
    ("none", ("--teacher-weight", "0.1")),
    ("none", ("--ap-cut-weight", "0.1")),
    ("teacher", ("--teacher-weight", "0.1", "--equivariance-weight", "0.1")),
    ("ap_cut", ("--ap-cut-weight", "0.1", "--bands-weight", "0.1",
                "--ap-axis", "1", "--ap-anterior-side", "high")),
])
def test_constraint_config_rejects_mixed_presets(monkeypatch, tmp_path, preset, extra) -> None:
    args = _parse(monkeypatch, tmp_path, "--constraint-set", preset, *extra)
    with pytest.raises(ValueError):
        trainer.resolve_constraint_config(args)


@pytest.mark.parametrize("preset, flag", [("teacher", "--teacher-weight"), ("ap_cut", "--ap-cut-weight")])
@pytest.mark.parametrize("weight", [None, "0", "-0.1", "nan", "inf"])
def test_new_presets_require_positive_finite_explicit_weight(monkeypatch, tmp_path, preset, flag, weight) -> None:
    extra = (flag, weight) if weight is not None else ()
    args = _parse(monkeypatch, tmp_path, "--constraint-set", preset, *extra)
    with pytest.raises(ValueError, match="finite positive"):
        trainer.resolve_constraint_config(args)


@pytest.mark.parametrize("orientation", [(), ("--ap-axis", "1"), ("--ap-anterior-side", "high")])
def test_ap_cut_requires_both_orientation_arguments(monkeypatch, tmp_path, orientation) -> None:
    args = _parse(monkeypatch, tmp_path, "--constraint-set", "ap_cut", "--ap-cut-weight", "0.1", *orientation)
    with pytest.raises(ValueError, match="explicit --ap-axis and --ap-anterior-side"):
        trainer.resolve_constraint_config(args)


def test_ap_high_orientation_and_teacher_configuration_reach_objective(monkeypatch, tmp_path) -> None:
    args = _parse(monkeypatch, tmp_path, "--constraint-set", "ap_cut", "--ap-cut-weight", "0.1",
                  "--ap-axis", "1", "--ap-anterior-side", "high", "--supervised-loss", "dice_ce")
    config = trainer.resolve_constraint_config(args)
    assert config.ap_axis == 1 and config.ap_anterior_low is False
    assert config.ap_cut_weight == 0.1 and config.equivariance_weight == 0.0
    objective = NewConstraintObjective(config)
    assert objective.ap_cut.config.anterior_low is False
    with pytest.raises(ValueError, match="Only one auxiliary"):
        NewConstraintObjective(replace(config, teacher_weight=0.1))
    args = _parse(monkeypatch, tmp_path, "--constraint-set", "teacher", "--teacher-weight", "0.03",
                  "--teacher-views", "3", "--teacher-temperature", "1.5")
    config = trainer.resolve_constraint_config(args)
    objective = NewConstraintObjective(config)
    assert objective.teacher.num_views == 3 and objective.teacher.temperature == 1.5


@pytest.mark.parametrize("preset", ["bands", "onecut"])
def test_dice_ce_cannot_reuse_legacy_calibration(monkeypatch, tmp_path, preset) -> None:
    args = _parse(monkeypatch, tmp_path, "--constraint-set", preset, "--supervised-loss", "dice_ce",
                  f"--{preset}-weight", "0.1", f"--{preset}-calibration-json", str(tmp_path / "legacy.json"))
    with pytest.raises(ValueError, match="Dice-only gradients"):
        trainer.resolve_constraint_config(args)


@pytest.mark.parametrize("loss, weight", [("dice_ce", 1.0), ("dice", 1.0)])
def test_calibration_checkpoint_validator_rejects_non_dice_source(tmp_path, loss, weight) -> None:
    checkpoint_path = tmp_path / "checkpoint_latest.pt"
    checkpoint_path.write_bytes(b"must reject provenance before trying to deserialize")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run": {
        "constraint_set": "none", "supervised_loss": loss, "ce_weight": weight,
    }}), encoding="utf-8")
    args = argparse.Namespace(checkpoint=checkpoint_path, checkpoint_config=config_path)
    with pytest.raises(ValueError, match="Dice-only supervised checkpoint"):
        validate_checkpoint_provenance(
            args, pkl_digest="a" * 64, splits_digest="b" * 64,
            checkpoint_path=checkpoint_path, source_provenance={"sha256": "c" * 64},
        )


class _TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.gradient_checks: list[bool] = []
        for parameter in self.parameters():
            parameter.register_hook(self._record_gradient)
        self.initial_parameters = {name: value.detach().clone() for name, value in self.named_parameters()}

    def _record_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        self.gradient_checks.append(bool(torch.isfinite(gradient).all()))
        return gradient

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(image)


def _records(prefix: str) -> list[dict[str, Any]]:
    # Spatial tensor axis 1 is posterior-low/anterior-high with an adjacent,
    # planar cut. The surrounding background also exercises calibration strata.
    labels = torch.zeros((1, 8, 8, 8), dtype=torch.long)
    labels[:, 1:7, 2:4, 1:7] = 2
    labels[:, 1:7, 4:6, 1:7] = 1
    image = torch.linspace(-1.0, 1.0, 8**3).reshape(1, 8, 8, 8)
    return [
        {"image": image * (1.0 + 0.2 * index) + 0.1 * index,
         "label": labels.clone(), "case_name": f"{prefix}_{index}"}
        for index in range(2)
    ]


@pytest.mark.parametrize("preset", ["none", "teacher", "ap_cut"])
def test_metric_headers_cover_evaluation_outputs_and_have_no_duplicates(preset) -> None:
    config = NewConstraintConfig(
        equivariance_weight=0.0,
        teacher_weight=0.03 if preset == "teacher" else 0.0,
        ap_cut_weight=0.05 if preset == "ap_cut" else 0.0,
        ap_axis=1, ap_anterior_low=False,
    )
    objective = NewConstraintObjective(config)
    model = _TinySegmentationModel()
    loader = DataLoader(_records("val"), batch_size=1)
    details: list[dict[str, Any]] = []
    evaluation = trainer.evaluate_validation_metrics(
        model, loader, objective, torch.device("cpu"), num_classes=3,
        amp=False, evaluate_constraints=True, all_translation_shifts=True,
        detail_rows=details, calibration_diagnostics=True,
    )
    fields = trainer.metric_fieldnames(calibration_diagnostics=True)
    assert len(fields) == len(set(fields))
    assert set(evaluation) <= set(fields)
    assert set(CalibrationDiagnostics(3).summary()) <= set(fields)
    assert evaluation["calibration/case_count"] == 2
    if preset != "none":
        name = "translation_teacher_kl" if preset == "teacher" else "ap_cut_posterior"
        assert f"val_{name}_raw_loss" in evaluation
        assert len(details) == 2
        assert {row["constraint_name"] for row in details} == {name}
    assert all(parameter.grad is None for parameter in model.parameters())


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise AssertionError(f"Nonfinite JSON constant {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


@pytest.mark.parametrize("preset", ["none", "teacher", "ap_cut"])
@pytest.mark.parametrize("augmentation", [False, True])
def test_small_cpu_training_checkpoint_metrics_and_resume(monkeypatch, tmp_path, preset, augmentation) -> None:
    if augmentation and preset == "ap_cut":
        pytest.skip("A/P cut geometry is not an augmentation experiment.")
    pkl = tmp_path / "dataset.pkl.gz"
    pkl.write_bytes(b"synthetic dataset bytes: hash and snapshot remain real")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"train": ["train_0", "train_1"], "val": ["val_0", "val_1"]}]), encoding="utf-8")
    created_models: list[_TinySegmentationModel] = []

    def build_small_model(spatial_size, num_classes, device):
        assert spatial_size == (8, 8, 8) and num_classes == 3
        model = _TinySegmentationModel().to(device)
        created_models.append(model)
        return model

    def build_small_data(args, train_generator, *, pkl_path, splits_path):
        assert pkl_path != args.pkl and pkl_path.read_bytes() == args.pkl.read_bytes()
        assert splits_path != args.splits_json and _strict_json(splits_path) == _strict_json(args.splits_json)
        train = DataLoader(_records("train"), batch_size=1, shuffle=True, generator=train_generator)
        validation = DataLoader(_records("val"), batch_size=1, shuffle=False)
        return train, validation, 3, 2, 2

    monkeypatch.setattr(trainer, "build_swinunetr", build_small_model)
    monkeypatch.setattr(trainer, "build_data", build_small_data)
    # Other agents can edit source during this test. All other provenance,
    # snapshot, optimizer, scheduler, checkpoint, and CSV code stays production.
    source = {"sha256": "f" * 64, "files": {"synthetic_test_source.py": "e" * 64}}
    monkeypatch.setattr(trainer, "collect_source_provenance", lambda: source)
    monkeypatch.setattr(trainer, "validate_source_provenance_unchanged", lambda expected: None)
    extra = [
        "--constraint-set", preset, "--supervised-loss", "dice_ce", "--ce-weight", "0.7",
        "--epochs", "2", "--batch-size", "1", "--spatial-size", "8", "8", "8",
        "--device", "cpu", "--learning-rate", "0.01", "--step-size", "1",
        "--constraint-warmup-epochs", "0", "--constraint-eval-every", "1",
    ]
    if preset == "none":
        extra.append("--calibration-diagnostics")
    elif preset == "teacher":
        extra += ["--teacher-weight", "0.03", "--teacher-views", "2"]
    else:
        extra += ["--ap-cut-weight", "0.05", "--ap-axis", "1", "--ap-anterior-side", "high"]
    if augmentation:
        extra.append("--translation-augmentation")
        if preset == "teacher":
            extra += ["--teacher-support", "common"]
    _parse(monkeypatch, tmp_path, *extra)
    with ExitStack() as stack:
        trainer._main(stack)

    output = tmp_path / "run"
    config = _strict_json(output / "config.json")
    final_metrics = _strict_json(output / "final_metrics.json")
    completion = _strict_json(output / "completion_manifest.json")
    checkpoint = torch.load(output / "checkpoint_latest.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["run"]["translation_augmentation"] == augmentation
    assert json.loads(json.dumps(checkpoint["run"])) == config["run"] == completion["run"]
    assert checkpoint["run"]["supervised_loss"] == "dice_ce"
    assert checkpoint["run"]["ce_weight"] == 0.7
    assert config["supervised_loss"]["name"] == "dice_ce"
    assert config["supervised_loss"]["ce_weight"] == 0.7
    assert config["pkl_sha256"] == trainer.file_sha256(pkl)
    assert config["splits_json_sha256"] == trainer.file_sha256(splits)
    assert completion["status"] == "complete"
    for relative_path, digest in completion["artifacts"].items():
        assert trainer.file_sha256(output / relative_path) == digest
    model = created_models[0]
    assert len(model.gradient_checks) == 2 * 2 * len(list(model.parameters()))
    assert all(model.gradient_checks)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(not torch.equal(value, model.initial_parameters[name]) for name, value in model.named_parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(checkpoint["model"][name], value)
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None and len(fieldnames) == len(set(fieldnames))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert all(None not in row for row in rows)
    assert all(math.isfinite(float(value)) for row in rows for value in row.values() if value != "")
    assert all(float(row["train_supervised_loss"]) > 0.0 for row in rows)
    if preset == "none":
        assert config["calibration_diagnostics"]["num_bins"] == 15
        assert final_metrics["calibration/case_count"] == 2
        assert float(rows[-1]["calibration/whole/nll"]) > 0.0
    else:
        assert config["calibration_diagnostics"] is None
        assert all(float(row["train_constraint_loss"]) >= 0.0 for row in rows)
    with (output / "validation_constraint_details.csv").open(newline="", encoding="utf-8") as handle:
        detail_rows = list(csv.DictReader(handle))
    assert all(None not in row for row in detail_rows)
    assert len(detail_rows) == (0 if preset == "none" else 2)
    if preset == "teacher":
        assert all(row["teacher_views"] == "12" for row in detail_rows)
    if preset == "ap_cut":
        assert checkpoint["run"]["constraint_config"]["ap_anterior_low"] is False

    # A completed-run resume still restores real model/optimizer/scheduler/RNG
    # state, validates the two CSV rows, and regenerates final artifacts.
    _parse(monkeypatch, tmp_path, *extra, "--resume")
    with ExitStack() as stack:
        trainer._main(stack)
    assert len(created_models) == 2
    assert created_models[-1].gradient_checks == []
    for name, value in created_models[-1].state_dict().items():
        assert torch.equal(value, checkpoint["model"][name])
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == rows
    assert _strict_json(output / "final_metrics.json") == final_metrics

    # The CE coefficient is immutable run identity, not a silently mutable knob.
    digests_before = {path.name: trainer.file_sha256(path) for path in output.iterdir() if path.is_file()}
    _parse(monkeypatch, tmp_path, *extra, "--ce-weight", "0.8", "--resume")
    with pytest.raises(ValueError, match="Resume configuration does not match"):
        with ExitStack() as stack:
            trainer._main(stack)
    assert {path.name: trainer.file_sha256(path) for path in output.iterdir() if path.is_file()} == digests_before
    if not augmentation and preset == "none":
        _parse(monkeypatch, tmp_path, *extra, "--translation-augmentation", "--resume")
        with pytest.raises(ValueError, match="Resume configuration does not match"):
            with ExitStack() as stack:
                trainer._main(stack)
