"""Morphology, numerical, and gradient tests for the outer-boundary bands."""

import csv
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss

from thesis.new_constraints import NewConstraintConfig, NewConstraintObjective
from thesis.new_constraints.bands import (
    OuterBoundaryBandLoss,
    build_boundary_bands,
    foreground_log_odds,
)
from thesis.new_constraints.bands.calibrate_weight import (
    CALIBRATION_CHECKPOINT_EPOCH,
    _summary,
    calibrated_weight,
    compute_logit_gradients,
    finalize_calibration_report,
    validate_calibration_protocol,
    validate_checkpoint_provenance,
    validate_split_integrity,
)
from thesis.new_constraints.train_swinunetr_constraints import resolve_constraint_config
from thesis.new_constraints.train_swinunetr_constraints import (
    RunSpec,
    collect_source_provenance,
    evaluate_constraint_metrics,
    evaluate_validation_metrics,
    file_sha256,
    reconcile_metrics_for_resume,
    reconcile_best_checkpoint,
    snapshot_file,
    validate_optimizer_hyperparameters,
    validate_epoch_metrics,
    validate_input_file_provenance,
    validate_resume_checkpoint,
    validate_three_class_labels,
)


class UnusedModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The bands constraint must not perform another forward pass.")


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.expand(images.shape[0], -1, -1, -1, -1)


class CountingPointwiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv3d(1, 3, kernel_size=1)
        self.forward_count = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        return self.projection(images)


def _cube_labels(size: int = 13) -> torch.Tensor:
    labels = torch.zeros(1, 1, size, size, size, dtype=torch.long)
    labels[:, :, 2:-2, 2:-2, 2:-2] = 1
    return labels


def _test_run_spec() -> RunSpec:
    return RunSpec(
        dataset="MSD",
        fold=0,
        seed=0,
        translation_seed=1,
        epochs=5,
        batch_size=1,
        spatial_size=(64, 64, 64),
        resize=False,
        optimizer_mode="adamw_0.01",
        learning_rate=1e-4,
        weight_decay=1e-5,
        step_size=20,
        adamw_gamma=0.5,
        constraint_set="none",
        constraint_config={"equivariance_weight": 0.0, "bands_weight": 0.0},
        constraint_warmup_epochs=5,
        constraint_eval_every=5,
        amp=False,
        initial_checkpoint=None,
        initial_checkpoint_sha256=None,
        pkl_sha256="a" * 64,
        splits_json_sha256="b" * 64,
        source_sha256="c" * 64,
    )


def test_public_band_loss_rejects_noncanonical_step_count() -> None:
    try:
        OuterBoundaryBandLoss(steps=1)
    except ValueError as error:
        assert "exactly 2 steps" in str(error)
    else:
        raise AssertionError("Direct construction bypassed the two-step contract.")


def test_file_snapshot_binds_digest_to_consumed_bytes(tmp_path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"version-one")
    snapshot = snapshot_file(source)
    source.write_bytes(b"version-two")
    try:
        assert snapshot.path.read_bytes() == b"version-one"
        assert snapshot.sha256 == file_sha256(snapshot.path)
        assert snapshot.sha256 != file_sha256(source)
    finally:
        snapshot.cleanup()


def test_three_class_schema_rejects_corrupt_labels() -> None:
    invalid_values = (-1, 0.5, np.nan, np.inf, 3, 255)
    for value in invalid_values:
        try:
            validate_three_class_labels(
                [{"case_name": f"case-{value}", "label": np.asarray([0, 1, value])}]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid label {value} was silently accepted.")

    validate_three_class_labels(
        [{"case_name": "valid", "label": np.asarray([0.0, 1.0, 2.0])}]
    )


def test_initializer_snapshot_changes_when_same_path_is_overwritten(tmp_path) -> None:
    initializer = tmp_path / "model.pt"
    initializer.write_bytes(b"weights-a")
    first = snapshot_file(initializer)
    initializer.write_bytes(b"weights-b")
    second = snapshot_file(initializer)
    try:
        assert first.original_path == second.original_path
        assert first.sha256 != second.sha256
    finally:
        first.cleanup()
        second.cleanup()


def test_source_provenance_records_manifest_commit_and_aggregate_digest() -> None:
    provenance = collect_source_provenance()

    assert len(provenance["sha256"]) == 64
    assert "thesis/new_constraints/train_swinunetr_constraints.py" in provenance[
        "files"
    ]
    assert "baselines/swin_unetr/swin_unetr.py" in provenance["files"]
    assert provenance["git_commit"] is None or len(provenance["git_commit"]) == 40


def test_two_step_bands_are_disjoint_and_on_the_correct_gt_side() -> None:
    labels = _cube_labels()
    foreground = labels > 0

    inner, outer = build_boundary_bands(foreground, steps=2)

    assert torch.all(inner <= foreground)
    assert torch.all(outer <= ~foreground)
    assert not torch.any(inner & outer)
    assert inner.any()
    assert outer.any()
    assert not inner[0, 0, 6, 6, 6]


def test_two_dilations_use_manhattan_not_euclidean_distance() -> None:
    foreground = torch.zeros(1, 1, 9, 9, 9, dtype=torch.bool)
    foreground[0, 0, 4, 4, 4] = True

    _, outer = build_boundary_bands(foreground, steps=2)

    assert outer[0, 0, 6, 4, 4]  # displacement (2, 0, 0), L1 distance 2
    assert outer[0, 0, 5, 5, 4]  # displacement (1, 1, 0), L1 distance 2
    assert not outer[0, 0, 5, 5, 5]  # displacement (1, 1, 1), L1 distance 3
    assert not outer[0, 0, 6, 6, 4]  # displacement (2, 2, 0), L1 distance 4


def test_foreground_log_odds_matches_summed_softmax_probability() -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 3, 4, 5, 6)

    log_odds = foreground_log_odds(logits, (1, 2), (0,))
    foreground_probability = torch.softmax(logits, dim=1)[:, 1:].sum(dim=1)

    assert torch.allclose(torch.sigmoid(log_odds), foreground_probability, atol=1e-6)


def test_band_gradients_have_the_correct_direction_and_locality() -> None:
    labels = _cube_labels()
    logits = torch.zeros(1, 3, 13, 13, 13, requires_grad=True)
    constraint = OuterBoundaryBandLoss(steps=2)

    result = constraint(logits, labels)
    result.loss.backward()
    gradient = logits.grad
    assert gradient is not None

    foreground = labels > 0
    inner, outer = build_boundary_bands(foreground, steps=2)
    inner_index = tuple(torch.nonzero(inner[0, 0], as_tuple=False)[0].tolist())
    outer_index = tuple(torch.nonzero(outer[0, 0], as_tuple=False)[0].tolist())
    deep_index = (6, 6, 6)

    assert gradient[(0, 1, *inner_index)] < 0
    assert gradient[(0, 2, *inner_index)] < 0
    assert gradient[(0, 0, *inner_index)] > 0
    assert gradient[(0, 1, *outer_index)] > 0
    assert gradient[(0, 2, *outer_index)] > 0
    assert gradient[(0, 0, *outer_index)] < 0
    assert torch.equal(gradient[(0, slice(None), *deep_index)], torch.zeros(3))


def test_band_loss_is_invariant_to_swapping_anterior_and_posterior() -> None:
    torch.manual_seed(12)
    labels = _cube_labels()
    logits = torch.randn(1, 3, 13, 13, 13)
    swapped = logits[:, (0, 2, 1)]
    constraint = OuterBoundaryBandLoss()

    first = constraint(logits, labels).loss
    second = constraint(swapped, labels).loss

    assert torch.allclose(first, second, atol=1e-7)


def test_invalid_batch_returns_finite_connected_float32_zero() -> None:
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    logits = torch.full(
        (1, 3, 8, 8, 8),
        10.0,
        dtype=torch.float16,
        requires_grad=True,
    )

    result = OuterBoundaryBandLoss()(logits, labels)
    result.loss.backward()

    assert result.loss.dtype == torch.float32
    assert result.loss.item() == 0.0
    assert torch.isfinite(result.loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0
    assert result.details["valid"].tolist() == [False]


def test_full_foreground_is_invalid_and_reported_as_edge_touching() -> None:
    labels = torch.ones(1, 1, 8, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 3, 8, 8, 8)

    result = OuterBoundaryBandLoss()(logits, labels)

    assert result.loss.item() == 0.0
    assert result.details["inner_voxels"].item() > 0
    assert result.details["outer_voxels"].item() == 0
    assert result.details["valid"].tolist() == [False]
    assert result.details["edge_touching"].tolist() == [True]


def test_invalid_patients_do_not_dilute_valid_patient_loss() -> None:
    valid_labels = _cube_labels()
    empty_labels = torch.zeros_like(valid_labels)
    batch_labels = torch.cat((valid_labels, empty_labels), dim=0)
    single_logits = torch.zeros(1, 3, 13, 13, 13)
    batch_logits = torch.zeros(2, 3, 13, 13, 13)
    constraint = OuterBoundaryBandLoss()

    single = constraint(single_logits, valid_labels)
    batch = constraint(batch_logits, batch_labels)

    assert torch.allclose(batch.loss, single.loss)
    assert batch.details["valid"].tolist() == [True, False]


def test_large_logits_remain_finite() -> None:
    labels = _cube_labels()
    logits = torch.full((1, 3, 13, 13, 13), 10_000.0)
    logits[:, 0] = -10_000.0

    result = OuterBoundaryBandLoss()(logits, labels)

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.details["inner_loss"]).all()
    assert torch.isfinite(result.details["outer_loss"]).all()


def test_band_numerics_remain_float32_under_autocast() -> None:
    labels = _cube_labels()
    logits = torch.zeros(1, 3, 13, 13, 13, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = OuterBoundaryBandLoss()(logits, labels)

    assert result.loss.dtype == torch.float32
    assert result.details["inner_loss"].dtype == torch.float32
    assert result.details["outer_loss"].dtype == torch.float32


def test_patient_losses_balance_inner_and_outer_sides() -> None:
    labels = _cube_labels()
    logits = torch.zeros(1, 3, 13, 13, 13)

    result = OuterBoundaryBandLoss()(logits, labels)

    assert result.details["inner_voxels"].item() != result.details["outer_voxels"].item()
    expected = 0.5 * (
        result.details["inner_loss"] + result.details["outer_loss"]
    )
    assert torch.allclose(result.loss, expected.mean())


def test_side_reductions_use_count_plus_epsilon() -> None:
    labels = _cube_labels()
    logits = torch.zeros(1, 3, 13, 13, 13)
    result = OuterBoundaryBandLoss(epsilon=1.0)(logits, labels)
    inner_count = result.details["inner_voxels"]
    outer_count = result.details["outer_voxels"]
    expected_inner = torch.log(torch.tensor(1.5)) * inner_count / (inner_count + 1.0)
    expected_outer = torch.log(torch.tensor(3.0)) * outer_count / (outer_count + 1.0)

    assert torch.allclose(result.details["inner_loss"], expected_inner)
    assert torch.allclose(result.details["outer_loss"], expected_outer)


def test_bands_objective_requires_labels_and_applies_weight() -> None:
    labels = _cube_labels()
    logits = torch.zeros(1, 3, 13, 13, 13, requires_grad=True)
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.0, bands_weight=0.25)
    )

    try:
        objective(UnusedModel(), torch.empty(1), logits)
    except ValueError as error:
        assert "requires transformed labels" in str(error)
    else:
        raise AssertionError("Missing labels were accepted.")

    output = objective(UnusedModel(), torch.empty(1), logits, labels)
    raw = output["results"]["outer_boundary_band"].loss
    assert torch.allclose(output["loss"], 0.25 * raw)


def test_constraint_flags_resolve_none_equivariance_and_bands() -> None:
    base = {
        "translation_size": 2,
        "equivariance_max_samples": 1,
        "band_steps": 2,
        "foreground_class_ids": (1, 2),
        "complement_class_ids": (0,),
    }
    none = resolve_constraint_config(
        SimpleNamespace(
            **base,
            constraint_set="none",
            equivariance_weight=None,
            bands_weight=None,
        )
    )
    equivariance = resolve_constraint_config(
        SimpleNamespace(
            **base,
            constraint_set="equivariance",
            equivariance_weight=None,
            bands_weight=None,
        )
    )
    alias = resolve_constraint_config(
        SimpleNamespace(
            **base,
            constraint_set="translation",
            equivariance_weight=None,
            bands_weight=None,
        )
    )
    bands = resolve_constraint_config(
        SimpleNamespace(
            **base,
            constraint_set="bands",
            equivariance_weight=None,
            bands_weight=0.04,
        )
    )

    assert none.equivariance_weight == none.bands_weight == 0.0
    assert equivariance == alias
    assert equivariance.equivariance_weight == 0.10
    assert bands.equivariance_weight == 0.0
    assert bands.bands_weight == 0.04


def test_canonical_bands_preset_rejects_non_two_step_geometry() -> None:
    arguments = SimpleNamespace(
        constraint_set="bands",
        equivariance_weight=None,
        bands_weight=0.04,
        translation_size=2,
        equivariance_max_samples=1,
        band_steps=1,
        foreground_class_ids=(1, 2),
        complement_class_ids=(0,),
    )

    try:
        resolve_constraint_config(arguments)
    except ValueError as error:
        assert "exactly --band-steps 2" in str(error)
    else:
        raise AssertionError("A noncanonical band radius was accepted.")

    try:
        NewConstraintObjective(
            NewConstraintConfig(
                equivariance_weight=0.0,
                bands_weight=0.04,
                band_steps=3,
            )
        )
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("Direct configuration bypassed the two-step contract.")


def test_gradient_calibration_uses_target_and_safety_cap() -> None:
    final, target, cap = calibrated_weight(
        [2.0, 2.0, 2.0],
        [4.0, 4.0, 100.0],
        target_ratio=0.10,
        safety_ratio=0.50,
    )

    assert target == 0.05
    assert cap < target
    assert final == cap


def test_calibration_rejects_nonfinite_ratios_and_noncanonical_scope() -> None:
    for value in (float("nan"), float("inf"), -float("inf")):
        try:
            calibrated_weight([1.0], [1.0], target_ratio=value)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("A nonfinite calibration ratio was accepted.")

    for max_cases, steps in ((33, 2), (1, 1), (0, 2)):
        try:
            validate_calibration_protocol(max_cases, steps)
        except ValueError:
            pass
        else:
            raise AssertionError("A noncanonical calibration protocol was accepted.")


def test_calibration_split_checks_patient_overlap_and_duplicates() -> None:
    try:
        validate_split_integrity(
            ["hippocampus_mni_s001_left"],
            ["hippocampus_mni_s001_right"],
            "MNI",
        )
    except ValueError as error:
        assert "patient IDs overlap" in str(error)
    else:
        raise AssertionError("Patient-level leakage was accepted.")

    try:
        validate_split_integrity(
            ["hippocampus_001", "hippocampus_001"],
            ["hippocampus_002"],
            "MSD",
        )
    except ValueError as error:
        assert "duplicate case names" in str(error)
    else:
        raise AssertionError("Duplicate split cases were accepted.")


def test_checkpoint_provenance_requires_matching_dice_only_epoch_five(tmp_path) -> None:
    pkl_path = tmp_path / "dataset.pkl"
    split_path = tmp_path / "splits.json"
    checkpoint_path = tmp_path / "checkpoint_latest.pt"
    config_path = tmp_path / "config.json"
    pkl_path.write_bytes(b"placeholder")
    split_path.write_text("[]", encoding="utf-8")
    source_provenance = collect_source_provenance()
    run = {
        "dataset": "MSD",
        "fold": 0,
        "epochs": CALIBRATION_CHECKPOINT_EPOCH,
        "spatial_size": [64, 64, 64],
        "resize": False,
        "constraint_set": "none",
        "initial_checkpoint": None,
        "pkl_sha256": file_sha256(pkl_path),
        "splits_json_sha256": file_sha256(split_path),
        "source_sha256": source_provenance["sha256"],
        "constraint_config": {
            "equivariance_weight": 0.0,
            "bands_weight": 0.0,
        },
    }
    config = {
        "run": run,
        "pkl": str(pkl_path.resolve()),
        "pkl_sha256": file_sha256(pkl_path),
        "splits_json": str(split_path.resolve()),
        "splits_json_sha256": file_sha256(split_path),
        "source_provenance": source_provenance,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint_run = {**run, "spatial_size": (64, 64, 64)}
    torch.save(
        {
            "epoch": CALIBRATION_CHECKPOINT_EPOCH,
            "model": {},
            "optimizer": {"unneeded": torch.ones(1)},
            "run": checkpoint_run,
        },
        checkpoint_path,
    )
    arguments = SimpleNamespace(
        checkpoint=checkpoint_path,
        checkpoint_config=config_path,
        dataset="MSD",
        fold=0,
        spatial_size=(64, 64, 64),
        resize=False,
        pkl=pkl_path,
        splits_json=split_path,
    )

    def validate() -> tuple[dict, object, dict]:
        pkl_snapshot = snapshot_file(pkl_path)
        splits_snapshot = snapshot_file(split_path)
        checkpoint_snapshot = snapshot_file(checkpoint_path)
        try:
            return validate_checkpoint_provenance(
                arguments,
                pkl_digest=pkl_snapshot.sha256,
                splits_digest=splits_snapshot.sha256,
                checkpoint_path=checkpoint_snapshot.path,
                source_provenance=source_provenance,
            )
        finally:
            pkl_snapshot.cleanup()
            splits_snapshot.cleanup()
            checkpoint_snapshot.cleanup()

    checkpoint, resolved_config, _ = validate()

    assert checkpoint["epoch"] == 5
    assert "optimizer" not in checkpoint
    assert resolved_config == config_path.resolve()

    initialized_run = {**run, "initial_checkpoint": "/models/unverified.pt"}
    config["run"] = initialized_run
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {
            "epoch": CALIBRATION_CHECKPOINT_EPOCH,
            "model": {},
            "run": initialized_run,
        },
        checkpoint_path,
    )
    try:
        validate()
    except ValueError as error:
        assert "without an initial checkpoint" in str(error)
    else:
        raise AssertionError("An unverified initializer was accepted.")

    config["run"] = run
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.save(checkpoint, checkpoint_path)
    pkl_path.write_bytes(b"changed in place")
    try:
        validate()
    except ValueError as error:
        assert "contents changed" in str(error)
    else:
        raise AssertionError("In-place dataset changes were accepted.")
    pkl_path.write_bytes(b"placeholder")

    checkpoint["epoch"] = 6
    torch.save(checkpoint, checkpoint_path)
    try:
        validate()
    except ValueError as error:
        assert "epoch-5" in str(error)
    else:
        raise AssertionError("A wrong-epoch checkpoint was accepted.")

    equivariance_run = {
        **run,
        "constraint_set": "equivariance",
        "constraint_config": {
            "equivariance_weight": 0.10,
            "bands_weight": 0.0,
        },
    }
    config["run"] = equivariance_run
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {
            "epoch": CALIBRATION_CHECKPOINT_EPOCH,
            "model": {},
            "run": equivariance_run,
        },
        checkpoint_path,
    )
    try:
        validate()
    except ValueError as error:
        assert "constraint-set none" in str(error)
    else:
        raise AssertionError("An equivariance checkpoint was accepted.")


def test_resume_checkpoint_must_match_embedded_run_provenance() -> None:
    run_spec = _test_run_spec()
    checkpoint = {
        "epoch": 3,
        "model": {},
        "optimizer": {},
        "scheduler": {"last_epoch": 3},
        "scaler": {},
        "best_hard_dice": 0.5,
        "best_epoch": 2,
        "rng": {},
        "run": asdict(run_spec),
    }

    validate_resume_checkpoint(checkpoint, run_spec)
    checkpoint["run"] = {**asdict(run_spec), "fold": 1}
    try:
        validate_resume_checkpoint(checkpoint, run_spec)
    except ValueError as error:
        assert "run provenance" in str(error)
    else:
        raise AssertionError("A checkpoint from another run was accepted.")


def test_resume_rejects_scheduler_chronology_and_nonfinite_metrics() -> None:
    run_spec = _test_run_spec()
    base = {
        "epoch": 3,
        "model": {},
        "optimizer": {},
        "scheduler": {"last_epoch": 3},
        "scaler": {},
        "best_hard_dice": 0.5,
        "best_epoch": 2,
        "rng": {},
        "run": asdict(run_spec),
    }
    invalid_cases = (
        ({**base, "scheduler": None}, "scheduler"),
        ({**base, "epoch": 999}, "outside"),
        ({**base, "best_epoch": 4}, "best_epoch"),
        ({**base, "best_hard_dice": float("nan")}, "finite"),
    )
    for checkpoint, message in invalid_cases:
        try:
            validate_resume_checkpoint(checkpoint, run_spec)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Malformed resume checkpoint passed: {message}")


def test_optimizer_hyperparameters_fail_before_training() -> None:
    valid = {
        "step_size": 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "adamw_gamma": 0.5,
    }
    invalid_cases = (
        {**valid, "step_size": 0},
        {**valid, "learning_rate": float("nan")},
        {**valid, "learning_rate": 0.0},
        {**valid, "weight_decay": float("inf")},
        {**valid, "weight_decay": -1.0},
        {**valid, "adamw_gamma": float("nan")},
        {**valid, "adamw_gamma": 0.0},
    )
    for arguments in invalid_cases:
        try:
            validate_optimizer_hyperparameters(**arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid optimizer arguments passed: {arguments}")

    validate_optimizer_hyperparameters(**valid)


def test_nonfinite_epoch_metrics_are_rejected_before_checkpointing() -> None:
    validate_epoch_metrics({"epoch": 1, "train_loss": 0.5})
    try:
        validate_epoch_metrics({"epoch": 1, "train_loss": float("nan")})
    except FloatingPointError as error:
        assert "train_loss" in str(error)
    else:
        raise AssertionError("A nonfinite epoch metric was accepted.")


def test_resume_repairs_best_checkpoint_from_durable_latest(tmp_path) -> None:
    run_spec = _test_run_spec()
    latest = {
        "epoch": 3,
        "model": {},
        "optimizer": {},
        "scheduler": {"last_epoch": 3},
        "scaler": {},
        "best_hard_dice": 0.7,
        "best_epoch": 3,
        "rng": {},
        "run": asdict(run_spec),
    }

    reconcile_best_checkpoint(tmp_path, latest, run_spec)

    repaired = torch.load(
        tmp_path / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert repaired["epoch"] == 3
    assert repaired["best_epoch"] == 3


def test_resume_input_provenance_checks_file_contents(tmp_path) -> None:
    pkl_path = tmp_path / "dataset.pkl"
    splits_path = tmp_path / "splits.json"
    pkl_path.write_bytes(b"dataset-v1")
    splits_path.write_text("[]", encoding="utf-8")
    config = {
        "pkl": str(pkl_path.resolve()),
        "pkl_sha256": file_sha256(pkl_path),
        "splits_json": str(splits_path.resolve()),
        "splits_json_sha256": file_sha256(splits_path),
    }

    validate_input_file_provenance(config, pkl_path, splits_path)
    splits_path.write_text('[{"changed": true}]', encoding="utf-8")
    try:
        validate_input_file_provenance(config, pkl_path, splits_path)
    except ValueError as error:
        assert "contents changed" in str(error)
    else:
        raise AssertionError("Resume accepted a split file changed in place.")


def test_resume_reconciles_uncheckpointed_metric_row(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    fieldnames = ["epoch", "train_loss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 1, "train_loss": 1.0},
                {"epoch": 2, "train_loss": 0.8},
                {"epoch": 3, "train_loss": 0.7},
            ]
        )

    reconcile_metrics_for_resume(path, checkpoint_epoch=2, fieldnames=fieldnames)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2]


def test_degenerate_calibration_writes_failure_report(tmp_path) -> None:
    output = tmp_path / "calibration.json"
    payload = {
        "status": "complete",
        "dice_gradient_rms": _summary([]),
        "band_gradient_rms_unconditional": _summary([]),
    }

    try:
        finalize_calibration_report(output, payload, [], [])
    except RuntimeError as error:
        assert "diagnostic report written" in str(error)
    else:
        raise AssertionError("Degenerate calibration did not fail.")

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["recommended_bands_weight"] is None
    assert report["dice_gradient_rms"]["count"] == 0


def test_calibration_detaches_model_graph_before_logit_gradients() -> None:
    model = nn.Conv3d(1, 3, kernel_size=1)
    images = torch.randn(1, 1, 13, 13, 13)
    labels = _cube_labels()

    logits, dice_gradient, band_gradient, _ = compute_logit_gradients(
        model,
        images,
        labels,
        DiceLoss(to_onehot_y=True, softmax=True),
        OuterBoundaryBandLoss(),
    )

    assert logits.is_leaf
    assert logits.grad_fn is None
    assert torch.isfinite(dice_gradient).all()
    assert torch.isfinite(band_gradient).all()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_validation_pipeline_reports_band_diagnostics_without_extra_forward() -> None:
    labels = _cube_labels()
    objective = NewConstraintObjective(
        NewConstraintConfig(equivariance_weight=0.0, bands_weight=0.04)
    )
    details: list[dict[str, object]] = []
    model = CountingPointwiseModel()

    metrics = evaluate_validation_metrics(
        model,
        [{"image": torch.zeros(1, 1, 13, 13, 13), "label": labels, "case_name": ["case-a"]}],
        objective,
        torch.device("cpu"),
        num_classes=3,
        amp=False,
        evaluate_constraints=True,
        all_translation_shifts=False,
        detail_rows=details,
    )

    assert model.forward_count == 1
    assert "val_dice_soft" in metrics
    assert "val_dice_hard" in metrics
    assert metrics["val_outer_boundary_band_valid_patient"] == 1.0
    assert metrics["val_outer_boundary_band_inner_voxels"] > 0
    assert metrics["val_outer_boundary_band_outer_voxels"] > 0
    assert "val_outer_boundary_band_truth" not in metrics
    assert "val_outer_boundary_band_confidence_adherent" not in metrics
    assert details[0]["constraint_name"] == "outer_boundary_band"
    assert details[0]["band_valid"] == 1


def test_three_pipeline_presets_have_expected_forward_counts_and_gradients() -> None:
    images = torch.randn(1, 1, 13, 13, 13)
    labels = _cube_labels()
    supervised_loss = DiceLoss(to_onehot_y=True, softmax=True)
    configurations = (
        (NewConstraintConfig(equivariance_weight=0.0, bands_weight=0.0), 1),
        (NewConstraintConfig(equivariance_weight=0.10, bands_weight=0.0), 2),
        (NewConstraintConfig(equivariance_weight=0.0, bands_weight=0.04), 1),
    )

    for config, expected_forwards in configurations:
        model = CountingPointwiseModel()
        objective = NewConstraintObjective(config)
        logits = model(images)
        auxiliary = objective(
            model,
            images,
            logits,
            labels,
            shift=(2, 0, 0),
        )["loss"]
        total = supervised_loss(logits, labels) + auxiliary
        total.backward()

        assert model.forward_count == expected_forwards
        assert model.projection.weight.grad is not None
        assert torch.isfinite(model.projection.weight.grad).all()
